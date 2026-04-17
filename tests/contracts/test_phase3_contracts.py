"""Phase 3 contract tests — instruction-driven architecture.

Tests the new tools (TTS, music gen), pipeline manifests, style playbooks,
stage director skills, meta skills, and the animated-explainer pipeline.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline_loader import (
    load_pipeline,
    get_stage_order,
    get_required_tools,
    get_stage_skill,
    get_stage_review_focus,
    list_pipelines,
)
from lib.checkpoint import STAGES
from schemas.artifacts import list_schemas
from styles.playbook_loader import load_playbook, list_playbooks, validate_playbook
from tools.base_tool import ToolTier
from tools.audio.music_gen import MusicGen
from tools.tool_registry import ToolRegistry
from tools.audio.elevenlabs_tts import ElevenLabsTTS
from tools.audio.openai_tts import OpenAITTS
from tools.audio.piper_tts import PiperTTS
from tools.audio.qwen_tts import QWEN_TTS_MODELS, QWEN_TTS_VOICES, QwenTTS
from tools.audio.tts_selector import TTSSelector


# ---- TTS Provider Tools ----

class TestElevenLabsTTS:
    def test_identity(self):
        tool = ElevenLabsTTS()
        info = tool.get_info()
        assert info["name"] == "elevenlabs_tts"
        assert info["tier"] == "voice"
        assert info["capability"] == "tts"
        assert info["provider"] == "elevenlabs"

    def test_cost_estimate(self):
        tool = ElevenLabsTTS()
        cost = tool.estimate_cost({"text": "Hello world, this is a test."})
        assert cost > 0
        assert cost < 0.01  # short text should be cheap

    def test_capabilities(self):
        tool = ElevenLabsTTS()
        assert "text_to_speech" in tool.capabilities
        assert "voice_selection" in tool.capabilities


class TestPiperTTS:
    def test_identity(self):
        tool = PiperTTS()
        info = tool.get_info()
        assert info["name"] == "piper_tts"
        assert info["tier"] == "voice"
        assert info["capability"] == "tts"
        assert info["provider"] == "piper"

    def test_cost_is_free(self):
        tool = PiperTTS()
        assert tool.estimate_cost({"text": "anything"}) == 0.0

    def test_capabilities(self):
        tool = PiperTTS()
        assert "text_to_speech" in tool.capabilities
        assert "offline_generation" in tool.capabilities


class TestMusicGen:
    def test_identity(self):
        tool = MusicGen()
        info = tool.get_info()
        assert info["name"] == "music_gen"
        assert info["tier"] == "generate"

    def test_cost_estimate_scales_with_duration(self):
        tool = MusicGen()
        cost_30 = tool.estimate_cost({"prompt": "ambient", "duration_seconds": 30})
        cost_60 = tool.estimate_cost({"prompt": "ambient", "duration_seconds": 60})
        assert cost_60 > cost_30

    def test_capabilities(self):
        tool = MusicGen()
        assert "generate_background_music" in tool.capabilities


class TestQwenTTS:
    def test_identity(self):
        tool = QwenTTS()
        info = tool.get_info()
        assert info["name"] == "qwen_tts"
        assert info["tier"] == "voice"
        assert info["capability"] == "tts"
        assert info["provider"] == "qwen"

    def test_capabilities(self):
        tool = QwenTTS()
        assert "text_to_speech" in tool.capabilities
        assert "voice_selection" in tool.capabilities
        assert "chinese_dialects" in tool.capabilities

    def test_cost_scales_with_text_length(self):
        tool = QwenTTS()
        short = tool.estimate_cost({"text": "Hi."})
        long = tool.estimate_cost({"text": "x" * 1000})
        assert long > short
        assert short >= 0

    def test_input_schema_voices_match_module(self):
        tool = QwenTTS()
        voices = tool.input_schema["properties"]["voice"]["enum"]
        assert set(voices) == set(QWEN_TTS_VOICES)
        assert "Cherry" in voices

    def test_models_enumerated(self):
        tool = QwenTTS()
        models = tool.input_schema["properties"]["model"]["enum"]
        assert set(models) == set(QWEN_TTS_MODELS)
        assert "qwen-tts" in models
        assert "qwen-tts-latest" in models

    def test_status_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.delenv("ALIYUN_DASHSCOPE_API_KEY", raising=False)
        from tools.base_tool import ToolStatus
        assert QwenTTS().get_status() == ToolStatus.UNAVAILABLE

    def test_execute_without_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.delenv("ALIYUN_DASHSCOPE_API_KEY", raising=False)
        result = QwenTTS().execute({"text": "hello"})
        assert result.success is False
        assert "DASHSCOPE_API_KEY" in result.error

    def test_execute_rejects_unknown_model(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
        result = QwenTTS().execute({"text": "hi", "model": "nope"})
        assert result.success is False
        assert "Unknown qwen_tts model" in result.error

    def test_execute_rejects_unknown_voice(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
        result = QwenTTS().execute({"text": "hi", "voice": "Bogus"})
        assert result.success is False
        assert "Unknown qwen_tts voice" in result.error

    def test_dialect_voice_requires_latest_model(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
        result = QwenTTS().execute({"text": "hi", "voice": "Dylan", "model": "qwen-tts"})
        assert result.success is False
        assert "dialect" in result.error.lower()

    def test_extract_audio_url_top_level(self):
        body = {"output": {"audio": {"url": "https://example.com/a.mp3"}}}
        assert QwenTTS._extract_audio_url(body) == "https://example.com/a.mp3"

    def test_extract_audio_url_nested_choices(self):
        body = {
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"audio": {"url": "https://example.com/b.mp3"}}
                            ]
                        }
                    }
                ]
            }
        }
        assert QwenTTS._extract_audio_url(body) == "https://example.com/b.mp3"

    def test_extract_audio_url_missing(self):
        assert QwenTTS._extract_audio_url({"output": {}}) is None

    def test_execute_happy_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")

        class FakeResponse:
            def __init__(self, body):
                self._body = body
            def raise_for_status(self):
                pass
            def json(self):
                return self._body

        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            return FakeResponse({
                "request_id": "req-1",
                "output": {"audio": {"url": "https://example.com/audio.mp3"}},
                "usage": {"audio_tokens": 7},
            })

        def fake_download(url, output_path, timeout=120):
            from pathlib import Path
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"FAKE_AUDIO")
            return p

        import requests
        from tools import _dashscope as dashscope_mod
        monkeypatch.setattr(requests, "post", fake_post)
        monkeypatch.setattr(dashscope_mod, "download_asset", fake_download)

        out_path = tmp_path / "qwen.mp3"
        result = QwenTTS().execute({
            "text": "Hello world",
            "voice": "Cherry",
            "model": "qwen-tts-latest",
            "output_path": str(out_path),
        })

        assert result.success is True, result.error
        assert result.data["provider"] == "qwen"
        assert result.data["model"] == "qwen-tts-latest"
        assert result.data["voice"] == "Cherry"
        assert result.data["request_id"] == "req-1"
        assert result.data["audio_url"] == "https://example.com/audio.mp3"
        assert result.artifacts == [str(out_path)]
        assert out_path.read_bytes() == b"FAKE_AUDIO"
        assert captured["payload"] == {
            "model": "qwen-tts-latest",
            "input": {"text": "Hello world", "voice": "Cherry"},
        }
        assert captured["headers"]["Authorization"] == "Bearer fake-key"
        assert "X-DashScope-Async" not in captured["headers"]


class TestNewToolsRegistry:
    def test_all_register(self):
        reg = ToolRegistry()
        reg.register(ElevenLabsTTS())
        reg.register(PiperTTS())
        reg.register(MusicGen())
        assert len(reg.list_all()) == 3

    def test_voice_tier_tools(self):
        reg = ToolRegistry()
        reg.register(ElevenLabsTTS())
        reg.register(OpenAITTS())
        reg.register(PiperTTS())
        voice_tools = reg.get_by_tier(ToolTier.VOICE)
        assert len(voice_tools) == 3
        names = {t.name for t in voice_tools}
        assert names == {"elevenlabs_tts", "openai_tts", "piper_tts"}


class TestCapabilityMetadata:
    def test_tts_tools_expose_capability_provider_and_location(self):
        tool = ElevenLabsTTS()
        info = tool.get_info()
        assert info["capability"] == "tts"
        assert info["provider"] == "elevenlabs"
        assert info["usage_location"].endswith("tools\\audio\\elevenlabs_tts.py") or info["usage_location"].endswith("tools/audio/elevenlabs_tts.py")
        assert "related_skills" in info
        assert "fallback_tools" in info

    def test_provider_specific_tts_tools_register(self):
        reg = ToolRegistry()
        reg.register(ElevenLabsTTS())
        reg.register(OpenAITTS())
        reg.register(PiperTTS())
        reg.register(TTSSelector())
        assert {tool.name for tool in reg.get_by_capability("tts")} == {
            "elevenlabs_tts",
            "openai_tts",
            "piper_tts",
            "tts_selector",
        }
        assert {tool.name for tool in reg.get_by_provider("elevenlabs")} == {"elevenlabs_tts"}

    def test_registry_catalog_views(self):
        reg = ToolRegistry()
        reg.register(ElevenLabsTTS())
        reg.register(OpenAITTS())
        reg.register(PiperTTS())
        catalog = reg.capability_catalog()
        assert "tts" in catalog
        providers = {item["provider"] for item in catalog["tts"] if item["provider"] != "selector"}
        assert providers == {"elevenlabs", "google_tts", "openai", "piper", "qwen"}


# ---- Animated Explainer Pipeline ----

class TestAnimatedExplainerManifest:
    def test_loads(self):
        manifest = load_pipeline("animated-explainer")
        assert manifest["name"] == "animated-explainer"
        assert manifest["version"] == "2.0"

    def test_all_stages_present(self):
        manifest = load_pipeline("animated-explainer")
        stage_names = get_stage_order(manifest)
        expected = ["research", "proposal", "script", "scene_plan", "assets", "edit", "compose", "publish"]
        assert stage_names == expected

    def test_every_stage_has_skill(self):
        manifest = load_pipeline("animated-explainer")
        for stage in manifest["stages"]:
            assert "skill" in stage, f"Stage {stage['name']} missing skill"
            skill = get_stage_skill(manifest, stage["name"])
            assert skill is not None
            assert skill.startswith("pipelines/explainer/")

    def test_every_stage_has_review_focus(self):
        manifest = load_pipeline("animated-explainer")
        for stage in manifest["stages"]:
            focus = get_stage_review_focus(manifest, stage["name"])
            assert len(focus) >= 3, f"Stage {stage['name']} needs more review focus items"

    def test_required_tools_complete(self):
        manifest = load_pipeline("animated-explainer")
        tools = get_required_tools(manifest)
        expected = {"tts_selector", "image_selector", "video_compose", "audio_mixer"}
        for t in expected:
            assert t in tools, f"Missing required tool: {t}"

    def test_creative_stages_require_human_approval(self):
        manifest = load_pipeline("animated-explainer")
        approval_stages = {"proposal", "script", "scene_plan", "publish"}
        for stage in manifest["stages"]:
            if stage["name"] in approval_stages:
                assert stage.get("human_approval_default") is True, (
                    f"Stage {stage['name']} should require human approval"
                )

    def test_listed(self):
        assert "animated-explainer" in list_pipelines()


# ---- Style Playbooks ----

class TestStylePlaybooks:
    def test_all_listed(self):
        playbooks = list_playbooks()
        assert "clean-professional" in playbooks
        assert "flat-motion-graphics" in playbooks
        assert "minimalist-diagram" in playbooks

    @pytest.mark.parametrize("name", ["clean-professional", "flat-motion-graphics", "minimalist-diagram"])
    def test_loads_and_validates(self, name):
        pb = load_playbook(name)
        assert pb["identity"]["name"]
        assert pb["identity"]["category"]

    @pytest.mark.parametrize("name", ["clean-professional", "flat-motion-graphics", "minimalist-diagram"])
    def test_has_required_sections(self, name):
        pb = load_playbook(name)
        assert "visual_language" in pb
        assert "typography" in pb
        assert "motion" in pb
        assert "audio" in pb
        assert "asset_generation" in pb
        assert "quality_rules" in pb
        assert len(pb["quality_rules"]) >= 3

    @pytest.mark.parametrize("name", ["clean-professional", "flat-motion-graphics", "minimalist-diagram"])
    def test_color_palette_complete(self, name):
        pb = load_playbook(name)
        palette = pb["visual_language"]["color_palette"]
        assert "primary" in palette
        assert "accent" in palette
        assert "background" in palette
        assert "text" in palette

    @pytest.mark.parametrize("name", ["clean-professional", "flat-motion-graphics", "minimalist-diagram"])
    def test_pacing_rules_present(self, name):
        pb = load_playbook(name)
        pacing = pb["motion"]["pacing_rules"]
        assert "min_scene_hold_seconds" in pacing
        assert "max_scene_hold_seconds" in pacing

    def test_compatible_with_manifest(self):
        manifest = load_pipeline("animated-explainer")
        available = list_playbooks()
        compat = manifest.get("compatible_playbooks", {})
        # compatible_playbooks is a dict with recommended/also_works lists
        playbook_names = compat.get("recommended", []) + compat.get("also_works", [])
        for name in playbook_names:
            assert name in available, f"Manifest references unavailable playbook: {name}"


# ---- Skills Existence ----

class TestSkillsExist:
    SKILLS_DIR = PROJECT_ROOT / "skills"

    @pytest.mark.parametrize("skill_path", [
        "pipelines/explainer/idea-director.md",
        "pipelines/explainer/script-director.md",
        "pipelines/explainer/scene-director.md",
        "pipelines/explainer/asset-director.md",
        "pipelines/explainer/edit-director.md",
        "pipelines/explainer/compose-director.md",
        "pipelines/explainer/publish-director.md",
    ])
    def test_director_skills_exist(self, skill_path):
        full_path = self.SKILLS_DIR / skill_path
        assert full_path.exists(), f"Missing director skill: {skill_path}"
        content = full_path.read_text(encoding="utf-8")
        assert len(content) > 500, f"Skill too short to be useful: {skill_path}"

    @pytest.mark.parametrize("skill_path", [
        "meta/reviewer.md",
        "meta/checkpoint-protocol.md",
        "meta/skill-creator.md",
    ])
    def test_meta_skills_exist(self, skill_path):
        full_path = self.SKILLS_DIR / skill_path
        assert full_path.exists(), f"Missing meta skill: {skill_path}"
        content = full_path.read_text(encoding="utf-8")
        assert len(content) > 500, f"Skill too short to be useful: {skill_path}"

    @pytest.mark.parametrize("skill_path", [
        "pipelines/explainer/idea-director.md",
        "pipelines/explainer/script-director.md",
        "pipelines/explainer/scene-director.md",
        "pipelines/explainer/asset-director.md",
        "pipelines/explainer/edit-director.md",
        "pipelines/explainer/compose-director.md",
        "pipelines/explainer/publish-director.md",
    ])
    def test_director_skills_have_required_sections(self, skill_path):
        content = (self.SKILLS_DIR / skill_path).read_text(encoding="utf-8")
        assert "## When to Use" in content
        assert "## Process" in content or "## Protocol" in content
        assert "Self-Evaluate" in content or "self-evaluate" in content.lower()

    @pytest.mark.parametrize("skill_path", [
        "meta/reviewer.md",
        "meta/checkpoint-protocol.md",
        "meta/skill-creator.md",
    ])
    def test_meta_skills_have_required_sections(self, skill_path):
        content = (self.SKILLS_DIR / skill_path).read_text(encoding="utf-8")
        assert "## When to Use" in content
        assert "## Protocol" in content or "## Process" in content


# ---- Remotion Scaffold ----

class TestRemotionScaffold:
    REMOTION_DIR = PROJECT_ROOT / "remotion-composer"

    def test_package_json_exists(self):
        assert (self.REMOTION_DIR / "package.json").exists()

    def test_entry_point_exists(self):
        assert (self.REMOTION_DIR / "src" / "index.tsx").exists()

    def test_root_composition_exists(self):
        assert (self.REMOTION_DIR / "src" / "Root.tsx").exists()

    def test_explainer_component_exists(self):
        assert (self.REMOTION_DIR / "src" / "Explainer.tsx").exists()

    def test_text_card_component_exists(self):
        assert (self.REMOTION_DIR / "src" / "components" / "TextCard.tsx").exists()

    def test_stat_card_component_exists(self):
        assert (self.REMOTION_DIR / "src" / "components" / "StatCard.tsx").exists()


# ---- Video Compose Operations ----

class TestVideoComposeOperations:
    def test_render_operation_exists(self):
        from tools.video.video_compose import VideoCompose
        tool = VideoCompose()
        ops = tool.input_schema["properties"]["operation"]["enum"]
        assert "render" in ops
        assert "remotion_render" in ops

    def test_render_rejects_missing_inputs(self):
        from tools.video.video_compose import VideoCompose
        tool = VideoCompose()
        result = tool.execute({"operation": "render"})
        assert not result.success
        assert "edit_decisions" in result.error
