"""Tests for the Alibaba DashScope provider tools (wan/qwen image + wan video).

The tests exercise everything that does not require a live API: registry
discovery, metadata shape, cost/runtime estimation, unavailable-path handling,
input validation, and the async submit + poll + download pipeline stubbed at
the ``requests`` boundary.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import _dashscope as dashscope
from tools.base_tool import ToolStatus
from tools.graphics.qwen_image import QWEN_IMAGE_MODELS, QwenImage
from tools.graphics.wan_image import WAN_IMAGE_MODELS, WanImage
from tools.tool_registry import ToolRegistry
from tools.video.wan_video_api import (
    ALL_MODELS as WAN_VIDEO_MODELS,
    WAN_I2V_MODELS,
    WAN_T2V_MODELS,
    WanVideoApi,
)


# ---------------------------------------------------------------------------
# Fake HTTP plumbing — we want to exercise submit/poll/download end-to-end
# without going near the real DashScope endpoints.
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, *, json_body: Any = None, content: bytes = b"") -> None:
        self._json_body = json_body
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._json_body


class FakeRequests:
    """Replaces the ``requests`` module imported inside ``tools._dashscope``."""

    def __init__(
        self,
        *,
        task_id: str = "task-123",
        result_urls: list[str] | None = None,
        result_key: str = "url",
        poll_sequence: list[str] | None = None,
        asset_bytes: bytes = b"BINARY",
    ) -> None:
        self.task_id = task_id
        self.result_urls = result_urls or ["https://example.com/a.png"]
        self.result_key = result_key
        self.poll_sequence = poll_sequence or ["RUNNING", "SUCCEEDED"]
        self.asset_bytes = asset_bytes
        self.post_calls: list[dict[str, Any]] = []
        self.get_calls: list[str] = []
        self._poll_index = 0

    # ---- POST: submit task ----
    def post(self, url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> FakeResponse:
        self.post_calls.append({"url": url, "headers": headers, "payload": json})
        return FakeResponse(json_body={"output": {"task_id": self.task_id}})

    # ---- GET: poll task or download asset ----
    def get(self, url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> FakeResponse:
        self.get_calls.append(url)
        if url.startswith(dashscope.TASKS_ENDPOINT):
            if self._poll_index >= len(self.poll_sequence):
                status = self.poll_sequence[-1]
            else:
                status = self.poll_sequence[self._poll_index]
                self._poll_index += 1
            if status.upper() == "SUCCEEDED":
                return FakeResponse(
                    json_body={
                        "output": {
                            "task_status": "SUCCEEDED",
                            "results": [{self.result_key: u} for u in self.result_urls],
                        }
                    }
                )
            return FakeResponse(json_body={"output": {"task_status": status}})
        # treat any other GET as an asset download
        return FakeResponse(content=self.asset_bytes)


@pytest.fixture
def fake_requests(monkeypatch) -> FakeRequests:
    fake = FakeRequests()
    monkeypatch.setitem(sys.modules, "requests", fake)
    return fake


@pytest.fixture
def with_api_key(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key-abc")


@pytest.fixture
def no_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("ALIYUN_DASHSCOPE_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# Registry discovery + metadata
# ---------------------------------------------------------------------------


class TestDashScopeRegistryDiscovery:
    def test_all_three_tools_discovered(self):
        reg = ToolRegistry()
        reg.discover()
        for name in ("wan_image", "qwen_image", "wan_video_api"):
            assert reg.get(name) is not None, f"{name} not registered"

    def test_capabilities_are_correct(self):
        reg = ToolRegistry()
        reg.discover()
        image_tools = {t.name: t for t in reg.get_by_capability("image_generation")}
        assert "wan_image" in image_tools
        assert "qwen_image" in image_tools
        video_tools = {t.name: t for t in reg.get_by_capability("video_generation")}
        assert "wan_video_api" in video_tools

    def test_provider_menu_groups_by_capability(self, with_api_key):
        reg = ToolRegistry()
        reg.discover()
        menu = reg.provider_menu()
        image_group = menu["image_generation"]
        names = {p["name"] for p in image_group["available"] + image_group["unavailable"]}
        assert {"wan_image", "qwen_image"} <= names

    def test_wan_video_api_does_not_collide_with_local_wan_video(self):
        reg = ToolRegistry()
        reg.discover()
        local = reg.get("wan_video")
        api = reg.get("wan_video_api")
        assert local is not None and api is not None
        assert local.runtime.value == "local_gpu"
        assert api.runtime.value == "api"


# ---------------------------------------------------------------------------
# Metadata sanity checks on each tool
# ---------------------------------------------------------------------------


class TestToolMetadata:
    def test_wan_image_model_matrix(self):
        tool = WanImage()
        info = tool.get_info()
        assert set(info["provider_matrix"]) == set(WAN_IMAGE_MODELS)
        # Issue #1 explicitly requires 2.7 variants to be supported.
        assert "wan2.7-image-pro" in WAN_IMAGE_MODELS
        assert "wan2.7-image" in WAN_IMAGE_MODELS

    def test_qwen_image_model_matrix(self):
        tool = QwenImage()
        info = tool.get_info()
        assert set(info["provider_matrix"]) == set(QWEN_IMAGE_MODELS)
        assert "qwen-image-2.0-pro" in QWEN_IMAGE_MODELS
        assert "qwen-image-2.0" in QWEN_IMAGE_MODELS

    def test_wan_video_api_model_matrix(self):
        tool = WanVideoApi()
        info = tool.get_info()
        assert set(info["provider_matrix"]) == set(WAN_VIDEO_MODELS)
        # The issue requires wan2.7-i2v and wan2.6-i2v-flash for image-to-video.
        assert "wan2.7-i2v" in WAN_I2V_MODELS
        assert "wan2.6-i2v-flash" in WAN_I2V_MODELS
        # And t2v family exists.
        assert WAN_T2V_MODELS, "expected at least one text-to-video model"

    def test_status_reflects_api_key_presence(self, with_api_key):
        for tool in (WanImage(), QwenImage(), WanVideoApi()):
            assert tool.get_status() == ToolStatus.AVAILABLE

    def test_status_unavailable_without_key(self, no_api_key):
        for tool in (WanImage(), QwenImage(), WanVideoApi()):
            assert tool.get_status() == ToolStatus.UNAVAILABLE

    def test_cost_scales_with_n_images(self):
        tool = WanImage()
        base = tool.estimate_cost({"model": "wan2.7-image", "n": 1})
        doubled = tool.estimate_cost({"model": "wan2.7-image", "n": 2})
        assert doubled == pytest.approx(base * 2)

    def test_video_cost_scales_with_duration(self):
        tool = WanVideoApi()
        base = tool.estimate_cost({"model": "wan2.7-t2v", "duration": 5})
        longer = tool.estimate_cost({"model": "wan2.7-t2v", "duration": 10})
        assert longer > base


# ---------------------------------------------------------------------------
# Graceful-failure paths without API key
# ---------------------------------------------------------------------------


class TestUnavailableBehaviour:
    def test_wan_image_fails_clean_without_key(self, no_api_key):
        result = WanImage().execute({"prompt": "a red fox"})
        assert result.success is False
        assert "DASHSCOPE_API_KEY" in (result.error or "")

    def test_qwen_image_fails_clean_without_key(self, no_api_key):
        result = QwenImage().execute({"prompt": "a red fox"})
        assert result.success is False
        assert "DASHSCOPE_API_KEY" in (result.error or "")

    def test_wan_video_api_fails_clean_without_key(self, no_api_key):
        result = WanVideoApi().execute({"prompt": "a red fox"})
        assert result.success is False
        assert "DASHSCOPE_API_KEY" in (result.error or "")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_wan_image_rejects_unknown_model(self, with_api_key):
        result = WanImage().execute({"prompt": "x", "model": "nope"})
        assert result.success is False
        assert "Unknown" in (result.error or "")

    def test_qwen_image_rejects_unknown_model(self, with_api_key):
        result = QwenImage().execute({"prompt": "x", "model": "nope"})
        assert result.success is False

    def test_wan_video_rejects_unknown_model(self, with_api_key):
        result = WanVideoApi().execute({"prompt": "x", "model": "nope"})
        assert result.success is False

    def test_wan_video_rejects_i2v_model_in_t2v_mode(self, with_api_key):
        result = WanVideoApi().execute(
            {"prompt": "x", "model": "wan2.6-i2v-flash", "operation": "text_to_video"}
        )
        assert result.success is False
        assert "text_to_video" in (result.error or "")

    def test_wan_video_rejects_t2v_model_in_i2v_mode(self, with_api_key):
        result = WanVideoApi().execute(
            {"prompt": "x", "model": "wan2.7-t2v", "operation": "image_to_video"}
        )
        assert result.success is False
        assert "image_to_video" in (result.error or "")

    def test_wan_video_requires_image_url_for_i2v(self, with_api_key):
        result = WanVideoApi().execute(
            {"prompt": "x", "model": "wan2.7-i2v", "operation": "image_to_video"}
        )
        assert result.success is False
        assert "img_url" in (result.error or "")


# ---------------------------------------------------------------------------
# End-to-end execution with fake HTTP
# ---------------------------------------------------------------------------


class TestHappyPaths:
    def test_wan_image_submit_poll_download(self, tmp_path, with_api_key, fake_requests):
        fake_requests.result_urls = ["https://example.com/generated.png"]
        fake_requests.result_key = "url"
        fake_requests.asset_bytes = b"PNGDATA"

        output_path = tmp_path / "out.png"
        result = WanImage().execute(
            {
                "prompt": "a cinematic sunset over a canyon",
                "model": "wan2.7-image-pro",
                "size": "1024*1024",
                "output_path": str(output_path),
                "seed": 7,
            }
        )
        assert result.success, result.error
        assert output_path.exists()
        assert output_path.read_bytes() == b"PNGDATA"
        assert result.data["task_id"] == "task-123"
        assert result.data["model"] == "wan2.7-image-pro"
        assert result.model == "dashscope/wan2.7-image-pro"
        # Request targeted the text2image endpoint.
        assert fake_requests.post_calls[0]["url"] == dashscope.TEXT_TO_IMAGE_ENDPOINT

    def test_qwen_image_happy_path(self, tmp_path, with_api_key, fake_requests):
        fake_requests.result_urls = ["https://example.com/qwen.png"]
        output_path = tmp_path / "qwen.png"
        result = QwenImage().execute(
            {
                "prompt": "一只可爱的熊猫",
                "model": "qwen-image-2.0-pro",
                "output_path": str(output_path),
            }
        )
        assert result.success, result.error
        assert output_path.exists()
        assert result.data["model"] == "qwen-image-2.0-pro"
        assert result.model == "dashscope/qwen-image-2.0-pro"

    def test_wan_video_api_happy_path_i2v(self, tmp_path, with_api_key, fake_requests, monkeypatch):
        fake_requests.result_urls = ["https://example.com/clip.mp4"]
        fake_requests.result_key = "video_url"
        fake_requests.asset_bytes = b"MP4"

        # probe_output shells out to ffprobe which isn't present in CI env;
        # stub it to a simple dict.
        from tools.video import _shared as video_shared

        monkeypatch.setattr(
            video_shared, "probe_output", lambda path: {"resolution": "1280x720"}
        )

        output_path = tmp_path / "clip.mp4"
        result = WanVideoApi().execute(
            {
                "prompt": "gentle zoom over a mountain lake",
                "model": "wan2.7-i2v",
                "operation": "image_to_video",
                "img_url": "https://example.com/source.jpg",
                "duration": 5,
                "output_path": str(output_path),
            }
        )
        assert result.success, result.error
        assert output_path.exists()
        assert output_path.read_bytes() == b"MP4"
        assert result.data["operation"] == "image_to_video"
        assert fake_requests.post_calls[0]["url"] == dashscope.VIDEO_SYNTHESIS_ENDPOINT
        payload = fake_requests.post_calls[0]["payload"]
        assert payload["input"]["img_url"] == "https://example.com/source.jpg"

    def test_failed_task_returns_error_result(self, tmp_path, with_api_key, monkeypatch):
        fake = FakeRequests(poll_sequence=["RUNNING", "FAILED"])
        monkeypatch.setitem(sys.modules, "requests", fake)
        result = WanImage().execute(
            {"prompt": "x", "output_path": str(tmp_path / "x.png")}
        )
        assert result.success is False
        assert "failed" in (result.error or "").lower()

    def test_poll_aliases_recognised(self, monkeypatch):
        """``ALIYUN_DASHSCOPE_API_KEY`` should be accepted as a fallback name."""
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.setenv("ALIYUN_DASHSCOPE_API_KEY", "alt-key")
        assert dashscope.get_api_key() == "alt-key"
        assert WanImage().get_status() == ToolStatus.AVAILABLE
