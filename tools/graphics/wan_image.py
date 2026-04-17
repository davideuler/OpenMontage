"""Wan text-to-image generation via Alibaba DashScope API.

Supports the Wan text-to-image family, including the 2.7 generation:

- ``wan2.7-image-pro`` — highest quality
- ``wan2.7-image`` — balanced quality / cost
- ``wan2.2-t2i-plus`` — previous-generation plus variant
- ``wan2.2-t2i-flash`` — faster / cheaper preview variant

See: https://help.aliyun.com/zh/model-studio/text-to-image
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


WAN_IMAGE_MODELS: dict[str, dict[str, Any]] = {
    "wan2.7-image-pro": {"cost_per_image_usd": 0.06, "tier": "pro"},
    "wan2.7-image": {"cost_per_image_usd": 0.04, "tier": "standard"},
    "wan2.2-t2i-plus": {"cost_per_image_usd": 0.03, "tier": "plus"},
    "wan2.2-t2i-flash": {"cost_per_image_usd": 0.02, "tier": "flash"},
}

WAN_IMAGE_SIZES = [
    "512*512",
    "768*768",
    "1024*1024",
    "1024*768",
    "768*1024",
    "1440*810",
    "810*1440",
    "1664*928",
    "928*1664",
]


class WanImage(BaseTool):
    name = "wan_image"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "wan"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.ASYNC
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.API

    dependencies = []  # checked dynamically via env var
    install_instructions = dashscope.INSTALL_INSTRUCTIONS
    agent_skills = []

    capabilities = [
        "generate_image",
        "generate_illustration",
        "text_to_image",
        "model_selection",
    ]
    supports = {
        "negative_prompt": True,
        "seed": True,
        "custom_size": True,
        "chinese_prompt": True,
    }
    best_for = [
        "photorealistic and illustrative images via Alibaba Model Studio",
        "strong Chinese-language prompting",
        "pipelines that want the Wan 2.7 family alongside Qwen-Image",
    ]
    not_good_for = ["offline generation", "deterministic pixel-perfect output"]
    fallback_tools = ["qwen_image", "flux_image", "google_imagen"]

    provider_matrix = {
        variant: {
            "tool": "wan_image",
            "mode": "api",
            "tier": meta["tier"],
            "cost_per_image_usd": meta["cost_per_image_usd"],
        }
        for variant, meta in WAN_IMAGE_MODELS.items()
    }

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "negative_prompt": {"type": "string"},
            "model": {
                "type": "string",
                "enum": sorted(WAN_IMAGE_MODELS),
                "default": "wan2.7-image",
            },
            "size": {
                "type": "string",
                "enum": WAN_IMAGE_SIZES,
                "default": "1024*1024",
            },
            "n": {"type": "integer", "minimum": 1, "maximum": 4, "default": 1},
            "seed": {"type": "integer"},
            "prompt_extend": {"type": "boolean", "default": True},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "model", "size", "seed"]
    side_effects = [
        "writes image file to output_path",
        "calls Alibaba DashScope API",
    ]
    user_visible_verification = ["Inspect generated image for relevance and quality"]

    def get_status(self) -> ToolStatus:
        if dashscope.get_api_key():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        model = inputs.get("model", "wan2.7-image")
        n = max(1, int(inputs.get("n", 1)))
        meta = WAN_IMAGE_MODELS.get(model, WAN_IMAGE_MODELS["wan2.7-image"])
        return meta["cost_per_image_usd"] * n

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 25.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = dashscope.get_api_key()
        if not api_key:
            return ToolResult(
                success=False,
                error="DASHSCOPE_API_KEY not set. " + self.install_instructions,
            )

        model = inputs.get("model", "wan2.7-image")
        if model not in WAN_IMAGE_MODELS:
            return ToolResult(
                success=False,
                error=f"Unknown wan_image model {model!r}. "
                f"Allowed: {sorted(WAN_IMAGE_MODELS)}",
            )

        parameters: dict[str, Any] = {
            "size": inputs.get("size", "1024*1024"),
            "n": max(1, int(inputs.get("n", 1))),
            "prompt_extend": bool(inputs.get("prompt_extend", True)),
        }
        if inputs.get("seed") is not None:
            parameters["seed"] = int(inputs["seed"])

        input_block: dict[str, Any] = {"prompt": inputs["prompt"]}
        if inputs.get("negative_prompt"):
            input_block["negative_prompt"] = inputs["negative_prompt"]

        payload = {"model": model, "input": input_block, "parameters": parameters}

        start = time.time()
        try:
            task_id = dashscope.submit_async_task(
                dashscope.TEXT_TO_IMAGE_ENDPOINT, payload, api_key
            )
            output = dashscope.poll_task(task_id, api_key)
            urls = dashscope.extract_result_urls(output, "url")
            if not urls:
                return ToolResult(
                    success=False,
                    error=f"DashScope returned no image URLs: {output!r}",
                )

            output_path = Path(inputs.get("output_path", "wan_image.png"))
            written_paths: list[Path] = []
            if len(urls) == 1:
                dashscope.download_asset(urls[0], output_path)
                written_paths.append(output_path)
            else:
                stem, suffix = output_path.stem, output_path.suffix or ".png"
                for idx, url in enumerate(urls):
                    target = output_path.with_name(f"{stem}_{idx}{suffix}")
                    dashscope.download_asset(url, target)
                    written_paths.append(target)
        except Exception as exc:
            return ToolResult(success=False, error=f"Wan image generation failed: {exc}")

        primary = written_paths[0]
        return ToolResult(
            success=True,
            data={
                "provider": "wan",
                "model": model,
                "prompt": inputs["prompt"],
                "size": parameters["size"],
                "output": str(primary),
                "output_path": str(primary),
                "outputs": [str(p) for p in written_paths],
                "task_id": task_id,
                "seed": parameters.get("seed"),
            },
            artifacts=[str(p) for p in written_paths],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            seed=parameters.get("seed"),
            model=f"dashscope/{model}",
        )
