"""Qwen-Image text-to-image generation via Alibaba DashScope (Model Studio).

Covers the Qwen-Image 2.0 family. Uses the same async DashScope protocol as
``wan_image`` but with Qwen-specific model IDs and cost profile.
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


QWEN_IMAGE_MODELS: dict[str, dict[str, Any]] = {
    "qwen-image-2.0-pro": {"name": "Qwen-Image 2.0 Pro", "cost_per_image": 0.10, "quality": "highest"},
    "qwen-image-2.0": {"name": "Qwen-Image 2.0", "cost_per_image": 0.05, "quality": "high"},
    "qwen-image-plus": {"name": "Qwen-Image Plus", "cost_per_image": 0.04, "quality": "high"},
    "qwen-image": {"name": "Qwen-Image", "cost_per_image": 0.02, "quality": "medium"},
}


class QwenImage(BaseTool):
    name = "qwen_image"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "qwen"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.API

    dependencies: list[str] = []
    install_instructions = install_instructions()
    agent_skills = ["flux-best-practices"]
    fallback_tools = ["wan_image", "flux_image", "google_imagen"]

    capabilities = ["text_to_image", "render_text_in_image"]
    supports = {
        "negative_prompt": True,
        "seed": True,
        "custom_size": True,
        "batch": True,
        "chinese_text_rendering": True,
    }
    best_for = [
        "renders readable text inside images (CN + EN)",
        "Chinese cultural imagery and poster design",
        "infographics that need legible captions",
    ]
    not_good_for = ["offline generation", "regions without DashScope access"]
    provider_matrix = {
        model_id: {"tool": "qwen_image", "mode": "dashscope", **meta}
        for model_id, meta in QWEN_IMAGE_MODELS.items()
    }

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "model": {"type": "string", "enum": sorted(QWEN_IMAGE_MODELS), "default": "qwen-image-2.0-pro"},
            "negative_prompt": {"type": "string"},
            "size": {"type": "string", "default": "1328*1328"},
            "n": {"type": "integer", "minimum": 1, "maximum": 4, "default": 1},
            "seed": {"type": "integer"},
            "prompt_extend": {"type": "boolean", "default": True},
            "watermark": {"type": "boolean", "default": False},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=100, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "model", "size", "seed", "n"]
    side_effects = ["writes image file(s) to output_path", "calls DashScope API"]
    user_visible_verification = ["Inspect generated image for relevance and quality"]

    def get_status(self) -> ToolStatus:
        if get_api_key():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        meta = QWEN_IMAGE_MODELS.get(
            inputs.get("model", "qwen-image-2.0-pro"), QWEN_IMAGE_MODELS["qwen-image-2.0-pro"]
        )
        n = int(inputs.get("n", 1) or 1)
        return round(meta["cost_per_image"] * max(1, n), 4)

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 20.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = get_api_key()
        if not api_key:
            return ToolResult(success=False, error="DASHSCOPE_API_KEY not set. " + self.install_instructions)

        model = inputs.get("model", "qwen-image-2.0-pro")
        if model not in QWEN_IMAGE_MODELS:
            return ToolResult(
                success=False,
                error=f"Unknown model: {model}. Available: {', '.join(sorted(QWEN_IMAGE_MODELS))}",
            )

        prompt = inputs["prompt"]
        parameters: dict[str, Any] = {
            "size": inputs.get("size", "1328*1328"),
            "n": int(inputs.get("n", 1) or 1),
            "prompt_extend": bool(inputs.get("prompt_extend", True)),
            "watermark": bool(inputs.get("watermark", False)),
        }
        if inputs.get("seed") is not None:
            parameters["seed"] = int(inputs["seed"])

        body: dict[str, Any] = {
            "model": model,
            "input": {"prompt": prompt},
            "parameters": parameters,
        }
        if inputs.get("negative_prompt"):
            body["input"]["negative_prompt"] = inputs["negative_prompt"]

        start = time.time()
        try:
            task_id = submit_task(
                "services/aigc/text2image/image-synthesis",
                body,
                api_key,
            )
            output = poll_task(task_id, api_key, timeout=300)
            urls = collect_asset_urls(output)
            if not urls:
                return ToolResult(success=False, error=f"DashScope returned no image URLs: {output}")

            base_output = Path(inputs.get("output_path", f"{model.replace('.', '_')}_{task_id}.png"))
            base_output.parent.mkdir(parents=True, exist_ok=True)
            written: list[str] = []
            for idx, url in enumerate(urls):
                if len(urls) == 1:
                    target = base_output
                else:
                    stem, suffix = base_output.stem, base_output.suffix or ".png"
                    target = base_output.with_name(f"{stem}_{idx}{suffix}")
                target.write_bytes(download_asset(url))
                written.append(str(target))
        except DashScopeError as exc:
            return ToolResult(success=False, error=f"Qwen image generation failed: {exc}")
        except Exception as exc:
            return ToolResult(success=False, error=f"Qwen image generation failed: {exc}")

        meta = QWEN_IMAGE_MODELS[model]
        return ToolResult(
            success=True,
            data={
                "provider": "qwen",
                "provider_name": meta["name"],
                "mode": "dashscope",
                "model": model,
                "prompt": prompt,
                "output": written[0],
                "outputs": written,
                "task_id": task_id,
                "size": parameters["size"],
                "n": parameters["n"],
            },
            artifacts=written,
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            seed=inputs.get("seed"),
            model=model,
        )
