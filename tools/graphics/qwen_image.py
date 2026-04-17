"""Qwen-Image text-to-image generation via Alibaba DashScope API.

Supports the Qwen-Image family, including the 2.0 generation:

- ``qwen-image-2.0-pro`` — highest quality
- ``qwen-image-2.0`` — balanced
- ``qwen-image-plus`` — previous-generation plus variant
- ``qwen-image`` — base variant

See: https://help.aliyun.com/zh/model-studio/qwen-image-api
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


QWEN_IMAGE_MODELS: dict[str, dict[str, Any]] = {
    "qwen-image-2.0-pro": {"cost_per_image_usd": 0.08, "tier": "pro"},
    "qwen-image-2.0": {"cost_per_image_usd": 0.05, "tier": "standard"},
    "qwen-image-plus": {"cost_per_image_usd": 0.04, "tier": "plus"},
    "qwen-image": {"cost_per_image_usd": 0.03, "tier": "base"},
}

QWEN_IMAGE_SIZES = [
    "512*512",
    "768*768",
    "1024*1024",
    "1024*768",
    "768*1024",
    "1328*1328",
    "1664*928",
    "928*1664",
]


class QwenImage(BaseTool):
    name = "qwen_image"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "qwen"
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
        "chinese_text_rendering",
        "model_selection",
    ]
    supports = {
        "negative_prompt": True,
        "seed": True,
        "custom_size": True,
        "chinese_prompt": True,
        "text_in_image": True,
    }
    best_for = [
        "rendering legible text inside images (especially Chinese)",
        "illustrations and posters via Alibaba Model Studio",
        "prompts mixing Chinese and English",
    ]
    not_good_for = ["offline generation", "photorealistic portrait detail"]
    fallback_tools = ["wan_image", "flux_image", "google_imagen"]

    provider_matrix = {
        variant: {
            "tool": "qwen_image",
            "mode": "api",
            "tier": meta["tier"],
            "cost_per_image_usd": meta["cost_per_image_usd"],
        }
        for variant, meta in QWEN_IMAGE_MODELS.items()
    }

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "negative_prompt": {"type": "string"},
            "model": {
                "type": "string",
                "enum": sorted(QWEN_IMAGE_MODELS),
                "default": "qwen-image-2.0",
            },
            "size": {
                "type": "string",
                "enum": QWEN_IMAGE_SIZES,
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
        model = inputs.get("model", "qwen-image-2.0")
        n = max(1, int(inputs.get("n", 1)))
        meta = QWEN_IMAGE_MODELS.get(model, QWEN_IMAGE_MODELS["qwen-image-2.0"])
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

        model = inputs.get("model", "qwen-image-2.0")
        if model not in QWEN_IMAGE_MODELS:
            return ToolResult(
                success=False,
                error=f"Unknown qwen_image model {model!r}. "
                f"Allowed: {sorted(QWEN_IMAGE_MODELS)}",
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

            output_path = Path(inputs.get("output_path", "qwen_image.png"))
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
            return ToolResult(success=False, error=f"Qwen image generation failed: {exc}")

        primary = written_paths[0]
        return ToolResult(
            success=True,
            data={
                "provider": "qwen",
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
