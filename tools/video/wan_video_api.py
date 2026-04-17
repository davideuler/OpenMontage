"""Wan video generation via Alibaba DashScope API (text-to-video + image-to-video).

Supports the Wan video family served through Alibaba Model Studio (Bailian):

- Text-to-video:   ``wan2.7-t2v``, ``wan2.5-t2v-plus``, ``wan2.2-t2v-plus``
- Image-to-video:  ``wan2.7-i2v``, ``wan2.6-i2v-flash``, ``wan2.5-i2v-plus``

See:
  - https://help.aliyun.com/zh/model-studio/text-to-video-api-reference
  - https://help.aliyun.com/zh/model-studio/image-to-video-api-reference/
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tools import _dashscope as dashscope
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


WAN_T2V_MODELS: dict[str, dict[str, Any]] = {
    "wan2.7-t2v": {"cost_per_clip_usd": 0.70, "tier": "pro", "max_duration": 10},
    "wan2.5-t2v-plus": {"cost_per_clip_usd": 0.50, "tier": "plus", "max_duration": 10},
    "wan2.2-t2v-plus": {"cost_per_clip_usd": 0.40, "tier": "plus", "max_duration": 10},
}

WAN_I2V_MODELS: dict[str, dict[str, Any]] = {
    "wan2.7-i2v": {"cost_per_clip_usd": 0.70, "tier": "pro", "max_duration": 10},
    "wan2.6-i2v-flash": {"cost_per_clip_usd": 0.25, "tier": "flash", "max_duration": 5},
    "wan2.5-i2v-plus": {"cost_per_clip_usd": 0.50, "tier": "plus", "max_duration": 10},
}

ALL_MODELS: dict[str, dict[str, Any]] = {**WAN_T2V_MODELS, **WAN_I2V_MODELS}

WAN_VIDEO_SIZES = [
    "1280*720",
    "720*1280",
    "960*960",
    "1920*1080",
    "1080*1920",
]


class WanVideoApi(BaseTool):
    name = "wan_video_api"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "wan"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.ASYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []  # checked dynamically via env var
    install_instructions = dashscope.INSTALL_INSTRUCTIONS
    agent_skills = ["dashscope", "ai-video-gen"]

    capabilities = [
        "text_to_video",
        "image_to_video",
        "model_selection",
    ]
    supports = {
        "text_to_video": True,
        "image_to_video": True,
        "reference_image": True,
        "offline": False,
        "native_audio": False,
        "chinese_prompt": True,
    }
    best_for = [
        "cloud Wan video generation without local GPU",
        "image-to-video shots driven by Wan 2.7 / 2.6 models",
        "pipelines already on the Alibaba Model Studio stack",
    ]
    not_good_for = ["offline pipelines", "budget-constrained projects"]
    fallback = "wan_video"
    fallback_tools = ["wan_video", "kling_video", "minimax_video", "ltx_video_local"]

    provider_matrix = {
        variant: {
            "tool": "wan_video_api",
            "mode": "api",
            "tier": meta["tier"],
            "cost_per_clip_usd": meta["cost_per_clip_usd"],
            "max_duration": meta["max_duration"],
            "operation": "image_to_video" if variant in WAN_I2V_MODELS else "text_to_video",
        }
        for variant, meta in ALL_MODELS.items()
    }

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "negative_prompt": {"type": "string"},
            "operation": {
                "type": "string",
                "enum": ["text_to_video", "image_to_video"],
                "default": "text_to_video",
            },
            "model": {
                "type": "string",
                "enum": sorted(ALL_MODELS),
                "default": "wan2.7-t2v",
            },
            "img_url": {
                "type": "string",
                "description": "Reference image URL (required for image_to_video)",
            },
            "image_path": {
                "type": "string",
                "description": "Alias for local reference image path; must be pre-uploaded to a URL",
            },
            "size": {
                "type": "string",
                "enum": WAN_VIDEO_SIZES,
                "default": "1280*720",
            },
            "duration": {
                "type": "integer",
                "minimum": 3,
                "maximum": 10,
                "default": 5,
                "description": "Clip duration in seconds",
            },
            "seed": {"type": "integer"},
            "prompt_extend": {"type": "boolean", "default": True},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "model", "operation", "size", "duration", "seed"]
    side_effects = [
        "writes video file to output_path",
        "calls Alibaba DashScope API",
    ]
    user_visible_verification = ["Watch generated clip for motion coherence and artifacts"]

    def get_status(self) -> ToolStatus:
        if dashscope.get_api_key():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def _resolve_operation(self, inputs: dict[str, Any]) -> str:
        """Infer operation from model name when the user does not specify it."""
        explicit = inputs.get("operation")
        if explicit in ("text_to_video", "image_to_video"):
            return explicit
        model = inputs.get("model", "wan2.7-t2v")
        return "image_to_video" if model in WAN_I2V_MODELS else "text_to_video"

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        model = inputs.get("model", "wan2.7-t2v")
        meta = ALL_MODELS.get(model)
        if meta is None:
            return 0.5
        duration = max(3, min(int(inputs.get("duration", 5)), meta["max_duration"]))
        return meta["cost_per_clip_usd"] * (duration / 5.0)

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 90.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = dashscope.get_api_key()
        if not api_key:
            return ToolResult(
                success=False,
                error="DASHSCOPE_API_KEY not set. " + self.install_instructions,
            )

        model = inputs.get("model", "wan2.7-t2v")
        if model not in ALL_MODELS:
            return ToolResult(
                success=False,
                error=f"Unknown wan_video_api model {model!r}. "
                f"Allowed: {sorted(ALL_MODELS)}",
            )

        operation = self._resolve_operation(inputs)
        if operation == "image_to_video" and model not in WAN_I2V_MODELS:
            return ToolResult(
                success=False,
                error=f"Model {model!r} does not support image_to_video. "
                f"Use one of: {sorted(WAN_I2V_MODELS)}",
            )
        if operation == "text_to_video" and model not in WAN_T2V_MODELS:
            return ToolResult(
                success=False,
                error=f"Model {model!r} does not support text_to_video. "
                f"Use one of: {sorted(WAN_T2V_MODELS)}",
            )

        meta = ALL_MODELS[model]
        duration = max(3, min(int(inputs.get("duration", 5)), meta["max_duration"]))
        parameters: dict[str, Any] = {
            "size": inputs.get("size", "1280*720"),
            "duration": duration,
            "prompt_extend": bool(inputs.get("prompt_extend", True)),
        }
        if inputs.get("seed") is not None:
            parameters["seed"] = int(inputs["seed"])

        input_block: dict[str, Any] = {"prompt": inputs["prompt"]}
        if inputs.get("negative_prompt"):
            input_block["negative_prompt"] = inputs["negative_prompt"]
        if operation == "image_to_video":
            img_url = inputs.get("img_url") or inputs.get("image_url")
            if not img_url:
                return ToolResult(
                    success=False,
                    error="image_to_video requires an 'img_url' (or 'image_url') pointing to a publicly reachable image.",
                )
            input_block["img_url"] = img_url

        payload = {"model": model, "input": input_block, "parameters": parameters}

        start = time.time()
        try:
            task_id = dashscope.submit_async_task(
                dashscope.VIDEO_SYNTHESIS_ENDPOINT, payload, api_key
            )
            output = dashscope.poll_task(task_id, api_key, timeout_seconds=1800)
            urls = dashscope.extract_result_urls(output, "video_url")
            if not urls:
                return ToolResult(
                    success=False,
                    error=f"DashScope returned no video URLs: {output!r}",
                )
            output_path = Path(inputs.get("output_path", "wan_video.mp4"))
            dashscope.download_asset(urls[0], output_path)
        except Exception as exc:
            return ToolResult(
                success=False, error=f"Wan video generation failed: {exc}"
            )

        from tools.video._shared import probe_output

        probed = probe_output(output_path)
        return ToolResult(
            success=True,
            data={
                "provider": "wan",
                "model": f"dashscope/{model}",
                "prompt": inputs["prompt"],
                "operation": operation,
                "size": parameters["size"],
                "duration": duration,
                "output": str(output_path),
                "output_path": str(output_path),
                "task_id": task_id,
                "format": "mp4",
                "seed": parameters.get("seed"),
                **probed,
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            seed=parameters.get("seed"),
            model=f"dashscope/{model}",
        )
