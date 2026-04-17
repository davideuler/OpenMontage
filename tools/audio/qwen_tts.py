"""Qwen-TTS text-to-speech provider tool via Alibaba DashScope API.

Qwen-TTS is Alibaba's text-to-speech family served through Model Studio
(Bailian). Voices include neutral English/Chinese personas (Cherry, Ethan,
Chelsie, Serena) and Chinese dialect personas exposed by the ``-latest``
model (Dylan/Beijing, Sunny/Sichuan, Jada/Shanghai).

The endpoint returns an audio URL synchronously (no async polling needed):

  POST https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
    body: {"model": "qwen-tts", "input": {"text": "...", "voice": "Cherry"}}
    -> output.audio.url -> download

See: https://help.aliyun.com/zh/model-studio/qwen-tts-api
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


QWEN_TTS_ENDPOINT = (
    f"{dashscope.DASHSCOPE_BASE_URL}/services/aigc/multimodal-generation/generation"
)

QWEN_TTS_MODELS: dict[str, dict[str, Any]] = {
    "qwen-tts": {"cost_per_1k_chars_usd": 0.0014, "supports_dialects": False},
    "qwen-tts-latest": {"cost_per_1k_chars_usd": 0.0014, "supports_dialects": True},
    "qwen-tts-2025-05-22": {"cost_per_1k_chars_usd": 0.0014, "supports_dialects": False},
}

QWEN_TTS_VOICES = [
    "Cherry",
    "Ethan",
    "Chelsie",
    "Serena",
    "Dylan",
    "Sunny",
    "Jada",
]


class QwenTTS(BaseTool):
    name = "qwen_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "qwen"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = dashscope.INSTALL_INSTRUCTIONS
    fallback = "openai_tts"
    fallback_tools = ["openai_tts", "elevenlabs_tts", "google_tts", "piper_tts"]
    agent_skills = ["text-to-speech"]

    capabilities = [
        "text_to_speech",
        "voice_selection",
        "multilingual",
        "chinese_native",
        "chinese_dialects",
    ]
    supports = {
        "voice_cloning": False,
        "multilingual": True,
        "offline": False,
        "native_audio": True,
        "chinese_prompt": True,
        "chinese_dialects": True,
    }
    best_for = [
        "native Chinese narration via Alibaba Model Studio",
        "Chinese dialect voiceovers (Beijing, Sichuan, Shanghai)",
        "bilingual Chinese/English delivery",
    ]
    not_good_for = [
        "voice cloning",
        "fully offline production",
    ]

    provider_matrix = {
        variant: {
            "tool": "qwen_tts",
            "mode": "api",
            "cost_per_1k_chars_usd": meta["cost_per_1k_chars_usd"],
            "supports_dialects": meta["supports_dialects"],
        }
        for variant, meta in QWEN_TTS_MODELS.items()
    }

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "description": "Text to convert to speech"},
            "model": {
                "type": "string",
                "enum": sorted(QWEN_TTS_MODELS),
                "default": "qwen-tts-latest",
            },
            "voice": {
                "type": "string",
                "enum": QWEN_TTS_VOICES,
                "default": "Cherry",
                "description": (
                    "Voice persona. Cherry/Ethan/Chelsie/Serena are bilingual "
                    "EN/ZH; Dylan/Sunny/Jada are Chinese dialect voices "
                    "(qwen-tts-latest only)."
                ),
            },
            "format": {
                "type": "string",
                "default": "mp3",
                "enum": ["mp3", "wav"],
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["text", "voice", "model", "format"]
    side_effects = ["writes audio file to output_path", "calls Alibaba DashScope API"]
    user_visible_verification = ["Listen to generated audio for natural speech quality"]

    _DIALECT_VOICES = {"Dylan", "Sunny", "Jada"}

    def get_status(self) -> ToolStatus:
        if dashscope.get_api_key():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        text = inputs.get("text", "")
        model = inputs.get("model", "qwen-tts-latest")
        meta = QWEN_TTS_MODELS.get(model, QWEN_TTS_MODELS["qwen-tts-latest"])
        return round(len(text) / 1000.0 * meta["cost_per_1k_chars_usd"], 4)

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = dashscope.get_api_key()
        if not api_key:
            return ToolResult(
                success=False,
                error="DASHSCOPE_API_KEY not set. " + self.install_instructions,
            )

        model = inputs.get("model", "qwen-tts-latest")
        if model not in QWEN_TTS_MODELS:
            return ToolResult(
                success=False,
                error=(
                    f"Unknown qwen_tts model {model!r}. "
                    f"Allowed: {sorted(QWEN_TTS_MODELS)}"
                ),
            )

        voice = inputs.get("voice", "Cherry")
        if voice not in QWEN_TTS_VOICES:
            return ToolResult(
                success=False,
                error=(
                    f"Unknown qwen_tts voice {voice!r}. "
                    f"Allowed: {QWEN_TTS_VOICES}"
                ),
            )
        if voice in self._DIALECT_VOICES and not QWEN_TTS_MODELS[model]["supports_dialects"]:
            return ToolResult(
                success=False,
                error=(
                    f"Voice {voice!r} requires a dialect-capable model "
                    "(use 'qwen-tts-latest')."
                ),
            )

        start = time.time()
        try:
            result = self._generate(inputs, api_key, model, voice)
        except Exception as exc:
            return ToolResult(success=False, error=f"Qwen TTS failed: {exc}")

        result.duration_seconds = round(time.time() - start, 2)
        result.cost_usd = self.estimate_cost(inputs)
        return result

    def _generate(
        self,
        inputs: dict[str, Any],
        api_key: str,
        model: str,
        voice: str,
    ) -> ToolResult:
        import requests

        text = inputs["text"]
        fmt = inputs.get("format", "mp3")

        payload = {
            "model": model,
            "input": {"text": text, "voice": voice},
        }

        response = requests.post(
            QWEN_TTS_ENDPOINT,
            headers=dashscope.build_headers(api_key, async_task=False),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        body = response.json()

        audio_url = self._extract_audio_url(body)
        if not audio_url:
            raise RuntimeError(f"Qwen TTS response missing audio URL: {body!r}")

        output_path = Path(inputs.get("output_path", f"qwen_tts.{fmt}"))
        dashscope.download_asset(audio_url, output_path)

        request_id = body.get("request_id")
        usage = body.get("usage") or {}

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": model,
                "voice": voice,
                "format": fmt,
                "text_length": len(text),
                "output": str(output_path),
                "audio_url": audio_url,
                "request_id": request_id,
                "usage": usage,
            },
            artifacts=[str(output_path)],
            model=f"dashscope/{model}",
        )

    @staticmethod
    def _extract_audio_url(body: dict[str, Any]) -> str | None:
        """Pull the audio URL out of a Qwen-TTS response.

        DashScope multimodal-generation responses place the result under
        ``output.audio.url``; some variants nest it inside
        ``output.choices[0].message.content[*].audio.url``.
        """
        output = body.get("output") or {}
        audio = output.get("audio")
        if isinstance(audio, dict) and isinstance(audio.get("url"), str):
            return audio["url"]
        for choice in output.get("choices") or []:
            message = choice.get("message") if isinstance(choice, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        nested = part.get("audio")
                        if isinstance(nested, dict) and isinstance(nested.get("url"), str):
                            return nested["url"]
        return None
