"""Wan video generation via Alibaba DashScope (Model Studio).

Covers both text-to-video (``wan2.7-t2v``, ``wan2.5-t2v-plus``,
``wan2.2-t2v-plus``) and image-to-video (``wan2.7-i2v``, ``wan2.6-i2v-flash``,
``wan2.5-i2v-plus``) against DashScope's async video-synthesis endpoints.

Named ``wan_video_api`` to coexist with the local-GPU ``wan_video`` tool.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tools._dashscope import (
    DashScopeError,
    collect_asset_urls,
    download_asset,
    get_api_key,
    install_instructions,
    poll_task,
    submit_task,
)
from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


WAN_VIDEO_T2V_MODELS: dict[str, dict[str, Any]] = {
    "wan2.7-t2v": {"name": "Wan 2.7 Text-to-Video", "cost_per_second": 0.40, "quality": "highest"},
    "wan2.5-t2v-plus": {"name": "Wan 2.5 T2V Plus", "cost_per_second": 0.30, "quality": "high"},
    "wan2.2-t2v-plus": {"name": "Wan 2.2 T2V Plus", "cost_per_second": 0.20, "quality": "high"},
}

WAN_VIDEO_I2V_MODELS: dict[str, dict[str, Any]] = {
    "wan2.7-i2v": {"name": "Wan 2.7 Image-to-Video", "cost_per_second": 0.45, "quality": "highest"},
    "wan2.6-i2v-flash": {"name": "Wan 2.6 I2V Flash", "cost_per_second": 0.20, "quality": "high"},
    "wan2.5-i2v-plus": {"name": "Wan 2.5 I2V Plus", "cost_per_second": 0.30, "quality": "high"},
}

T2V_ENDPOINT = "services/aigc/video-generation/video-synthesis"
I2V_ENDPOINT = "services/aigc/image2video/video-synthesis"


class WanVideoAPI(BaseTool):
    name = "wan_video_api"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "wan"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies: list[str] = []
    install_instructions = install_instructions()
    agent_skills = ["ai-video-gen"]
    fallback_tools = ["wan_video", "kling_video", "minimax_video", "veo_video"]

    capabilities = ["text_to_video", "image_to_video"]
    supports = {
        "text_to_video": True,
        "image_to_video": True,
        "reference_image": True,
        "offline": False,
        "native_audio": False,
    }
    best_for = [
        "Wan 2.7 t2v and i2v via Alibaba DashScope (no local GPU needed)",
        "strong motion coherence and physics",
        "Chinese prompts and localized cultural content",
    ]
    not_good_for = ["offline generation", "regions without DashScope access"]
    provider_matrix = {
        **{mid: {"tool": "wan_video_api", "mode": "dashscope", "operation": "text_to_video", **meta}
           for mid, meta in WAN_VIDEO_T2V_MODELS.items()},
        **{mid: {"tool": "wan_video_api", "mode": "dashscope", "operation": "image_to_video", **meta}
           for mid, meta in WAN_VIDEO_I2V_MODELS.items()},
    }

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "operation": {
                "type": "string",
                "enum": ["text_to_video", "image_to_video"],
                "default": "text_to_video",
            },
            "model": {
                "type": "string",
                "enum": sorted({*WAN_VIDEO_T2V_MODELS, *WAN_VIDEO_I2V_MODELS}),
            },
            "img_url": {"type": "string", "description": "Reference image URL (required for image_to_video)"},
            "negative_prompt": {"type": "string"},
            "size": {"type": "string"},
            "duration": {"type": "integer", "minimum": 3, "maximum": 10, "default": 5},
            "prompt_extend": {"type": "boolean", "default": True},
            "seed": {"type": "integer"},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "model", "operation", "img_url", "seed"]
    side_effects = ["writes video file to output_path", "calls DashScope API"]
    user_visible_verification = ["Watch generated clip for motion coherence and artifacts"]

    def get_status(self) -> ToolStatus:
        if get_api_key():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def _resolve_model(self, inputs: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        """Return (operation, model, meta). Raises ValueError on mismatch."""
        operation = inputs.get("operation", "text_to_video")
        if operation not in {"text_to_video", "image_to_video"}:
            raise ValueError(f"operation must be text_to_video or image_to_video, got {operation!r}")

        if operation == "text_to_video":
            model = inputs.get("model", "wan2.7-t2v")
            if model not in WAN_VIDEO_T2V_MODELS:
                raise ValueError(
                    f"Model {model!r} is not a Wan text-to-video model. "
                    f"Available: {', '.join(sorted(WAN_VIDEO_T2V_MODELS))}"
                )
            return operation, model, WAN_VIDEO_T2V_MODELS[model]

        model = inputs.get("model", "wan2.7-i2v")
        if model not in WAN_VIDEO_I2V_MODELS:
            raise ValueError(
                f"Model {model!r} is not a Wan image-to-video model. "
                f"Available: {', '.join(sorted(WAN_VIDEO_I2V_MODELS))}"
            )
        return operation, model, WAN_VIDEO_I2V_MODELS[model]

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        try:
            _operation, _model, meta = self._resolve_model(inputs)
        except ValueError:
            return 0.0
        duration = int(inputs.get("duration", 5) or 5)
        return round(meta["cost_per_second"] * max(1, duration), 4)

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 120.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = get_api_key()
        if not api_key:
            return ToolResult(success=False, error="DASHSCOPE_API_KEY not set. " + self.install_instructions)

        try:
            operation, model, meta = self._resolve_model(inputs)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))

        prompt = inputs["prompt"]
        parameters: dict[str, Any] = {
            "duration": int(inputs.get("duration", 5) or 5),
            "prompt_extend": bool(inputs.get("prompt_extend", True)),
        }
        if inputs.get("size"):
            parameters["size"] = inputs["size"]
        if inputs.get("seed") is not None:
            parameters["seed"] = int(inputs["seed"])

        input_block: dict[str, Any] = {"prompt": prompt}
        if inputs.get("negative_prompt"):
            input_block["negative_prompt"] = inputs["negative_prompt"]

        if operation == "image_to_video":
            img_url = inputs.get("img_url") or inputs.get("reference_image_url")
            if not img_url:
                return ToolResult(
                    success=False,
                    error="image_to_video requires img_url (or reference_image_url).",
                )
            input_block["img_url"] = img_url
            endpoint = I2V_ENDPOINT
        else:
            endpoint = T2V_ENDPOINT

        body = {"model": model, "input": input_block, "parameters": parameters}

        start = time.time()
        try:
            task_id = submit_task(endpoint, body, api_key)
            output = poll_task(task_id, api_key, timeout=900)
            urls = collect_asset_urls(output)
            if not urls:
                return ToolResult(success=False, error=f"DashScope returned no video URL: {output}")

            output_path = Path(inputs.get("output_path", f"wan_api_{model.replace('.', '_')}_{task_id}.mp4"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(download_asset(urls[0], timeout=300))
        except DashScopeError as exc:
            return ToolResult(success=False, error=f"Wan video API generation failed: {exc}")
        except Exception as exc:
            return ToolResult(success=False, error=f"Wan video API generation failed: {exc}")

        from tools.video._shared import probe_output

        probed = probe_output(output_path)
        return ToolResult(
            success=True,
            data={
                "provider": "wan",
                "provider_name": meta["name"],
                "mode": "dashscope",
                "model": model,
                "operation": operation,
                "prompt": prompt,
                "task_id": task_id,
                "output": str(output_path),
                "output_path": str(output_path),
                "format": "mp4",
                **probed,
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            seed=inputs.get("seed"),
            model=model,
        )
