"""Tests for DashScope-backed providers: wan_image, qwen_image, wan_video_api.

The DashScope API is never actually called — ``requests.post`` / ``requests.get``
are replaced with fakes scoped to the module each tool imports from. Every
happy path walks the full submit -> poll -> download sequence so regressions in
the shared ``tools/_dashscope.py`` helper surface immediately.
"""
from __future__ import annotations

from typing import Any

import pytest

from tools import _dashscope
from tools._dashscope import DashScopeError, collect_asset_urls, poll_task, submit_task
from tools.base_tool import ToolStatus
from tools.graphics.qwen_image import QWEN_IMAGE_MODELS, QwenImage
from tools.graphics.wan_image import WAN_IMAGE_MODELS, WanImage
from tools.tool_registry import ToolRegistry
from tools.video.wan_video_api import (
    WAN_VIDEO_I2V_MODELS,
    WAN_VIDEO_T2V_MODELS,
    WanVideoAPI,
)


# ---------------------------------------------------------------------------
# Fake HTTP primitives
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, payload: Any = None, *, content: bytes = b"", status: int = 200):
        self._payload = payload
        self.content = content
        self.status_code = status

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeRequests:
    """Records calls and serves queued responses."""

    def __init__(self):
        self.post_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self._post_queue: list[FakeResponse] = []
        self._get_queue: list[FakeResponse] = []

    def enqueue_post(self, response: FakeResponse) -> None:
        self._post_queue.append(response)

    def enqueue_get(self, response: FakeResponse) -> None:
        self._get_queue.append(response)

    def post(self, url, *, headers=None, json=None, timeout=None):
        self.post_calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if not self._post_queue:
            raise AssertionError(f"Unexpected POST to {url}")
        return self._post_queue.pop(0)

    def get(self, url, *, headers=None, timeout=None):
        self.get_calls.append({"url": url, "headers": headers, "timeout": timeout})
        if not self._get_queue:
            raise AssertionError(f"Unexpected GET to {url}")
        return self._get_queue.pop(0)


@pytest.fixture
def fake_requests(monkeypatch):
    fake = FakeRequests()

    class _RequestsModule:
        post = staticmethod(fake.post)
        get = staticmethod(fake.get)
        RequestException = RuntimeError  # satisfy ``requests.RequestException`` imports

    monkeypatch.setitem(__import__("sys").modules, "requests", _RequestsModule)
    return fake


@pytest.fixture
def api_key_env(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-123")
    # Ensure the alt var does not also leak in from the real env
    monkeypatch.delenv("ALIBABA_DASHSCOPE_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# Registry discovery
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_registry():
    reg = ToolRegistry()
    reg.discover("tools")
    return reg


def test_registry_registers_three_new_tools(fresh_registry):
    for name in ("wan_image", "qwen_image", "wan_video_api"):
        assert fresh_registry.get(name) is not None, f"{name} missing from registry"


def test_registry_does_not_collide_with_local_wan_video(fresh_registry):
    assert fresh_registry.get("wan_video") is not None
    assert fresh_registry.get("wan_video_api") is not None
    assert fresh_registry.get("wan_video") is not fresh_registry.get("wan_video_api")


def test_image_capability_includes_wan_and_qwen(fresh_registry):
    names = {t.name for t in fresh_registry.get_by_capability("image_generation")}
    assert {"wan_image", "qwen_image"}.issubset(names)


def test_video_capability_includes_wan_video_api(fresh_registry):
    names = {t.name for t in fresh_registry.get_by_capability("video_generation")}
    assert "wan_video_api" in names


# ---------------------------------------------------------------------------
# Metadata / status / cost
# ---------------------------------------------------------------------------


def test_wan_image_metadata_exposes_all_four_models():
    info = WanImage().get_info()
    assert set(info["provider_matrix"].keys()) == set(WAN_IMAGE_MODELS.keys())
    assert "wan2.7-image-pro" in info["input_schema"]["properties"]["model"]["enum"]


def test_qwen_image_metadata_exposes_all_four_models():
    info = QwenImage().get_info()
    assert set(info["provider_matrix"].keys()) == set(QWEN_IMAGE_MODELS.keys())
    assert "qwen-image-2.0-pro" in info["input_schema"]["properties"]["model"]["enum"]


def test_wan_video_api_covers_t2v_and_i2v_models():
    info = WanVideoAPI().get_info()
    matrix_keys = set(info["provider_matrix"].keys())
    assert set(WAN_VIDEO_T2V_MODELS.keys()).issubset(matrix_keys)
    assert set(WAN_VIDEO_I2V_MODELS.keys()).issubset(matrix_keys)
    # Matrix entries carry the operation they belong to so agents can route.
    assert info["provider_matrix"]["wan2.7-t2v"]["operation"] == "text_to_video"
    assert info["provider_matrix"]["wan2.7-i2v"]["operation"] == "image_to_video"


def test_status_is_unavailable_without_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("ALIBABA_DASHSCOPE_API_KEY", raising=False)
    assert WanImage().get_status() == ToolStatus.UNAVAILABLE
    assert QwenImage().get_status() == ToolStatus.UNAVAILABLE
    assert WanVideoAPI().get_status() == ToolStatus.UNAVAILABLE


def test_status_is_available_with_api_key(api_key_env):
    assert WanImage().get_status() == ToolStatus.AVAILABLE
    assert QwenImage().get_status() == ToolStatus.AVAILABLE
    assert WanVideoAPI().get_status() == ToolStatus.AVAILABLE


def test_wan_image_cost_scales_with_n():
    tool = WanImage()
    assert tool.estimate_cost({"model": "wan2.7-image-pro", "n": 1}) == pytest.approx(0.12)
    assert tool.estimate_cost({"model": "wan2.7-image-pro", "n": 3}) == pytest.approx(0.36)


def test_qwen_image_cost_uses_model_table():
    tool = QwenImage()
    assert tool.estimate_cost({"model": "qwen-image-2.0-pro"}) == pytest.approx(0.10)
    assert tool.estimate_cost({"model": "qwen-image"}) == pytest.approx(0.02)


def test_wan_video_api_cost_scales_with_duration():
    tool = WanVideoAPI()
    assert tool.estimate_cost({"operation": "text_to_video", "model": "wan2.7-t2v", "duration": 5}) == pytest.approx(2.00)
    assert tool.estimate_cost({"operation": "image_to_video", "model": "wan2.6-i2v-flash", "duration": 6}) == pytest.approx(1.20)


def test_wan_video_api_cost_tolerates_invalid_model():
    # Cost estimation should not raise if the caller hasn't validated yet.
    assert WanVideoAPI().estimate_cost({"operation": "text_to_video", "model": "bogus"}) == 0.0


# ---------------------------------------------------------------------------
# DashScope helper: happy path, terminal failure, timeout
# ---------------------------------------------------------------------------


def test_submit_task_includes_async_header(fake_requests):
    fake_requests.enqueue_post(FakeResponse({"output": {"task_id": "t-1", "task_status": "PENDING"}}))
    task_id = submit_task("services/aigc/text2image/image-synthesis", {"model": "wan2.7-image"}, "sk-x")
    assert task_id == "t-1"
    call = fake_requests.post_calls[0]
    assert call["headers"]["Authorization"] == "Bearer sk-x"
    assert call["headers"]["X-DashScope-Async"] == "enable"
    assert call["url"].endswith("/services/aigc/text2image/image-synthesis")


def test_submit_task_raises_when_no_task_id(fake_requests):
    fake_requests.enqueue_post(FakeResponse({"output": {}}))
    with pytest.raises(DashScopeError):
        submit_task("services/aigc/text2image/image-synthesis", {}, "sk-x")


def test_poll_task_transitions_running_to_succeeded(fake_requests):
    fake_requests.enqueue_get(FakeResponse({"output": {"task_status": "RUNNING"}}))
    fake_requests.enqueue_get(
        FakeResponse({"output": {"task_status": "SUCCEEDED", "results": [{"url": "https://asset/one.png"}]}})
    )
    sleeps: list[float] = []
    output = poll_task("t-1", "sk-x", timeout=60, initial_interval=0.0, max_interval=0.0, sleep=sleeps.append)
    assert output["task_status"] == "SUCCEEDED"
    assert collect_asset_urls(output) == ["https://asset/one.png"]
    assert sleeps  # polled at least once


def test_poll_task_raises_on_failure(fake_requests):
    fake_requests.enqueue_get(FakeResponse({"output": {"task_status": "FAILED", "message": "bad prompt"}}))
    with pytest.raises(DashScopeError) as exc_info:
        poll_task("t-1", "sk-x", timeout=10, initial_interval=0.0, sleep=lambda _s: None)
    assert "FAILED" in str(exc_info.value)
    assert "bad prompt" in str(exc_info.value)


def test_collect_asset_urls_handles_single_video_url():
    assert collect_asset_urls({"video_url": "https://v/one.mp4"}) == ["https://v/one.mp4"]
    assert collect_asset_urls({"results": [{"url": "a"}, {"url": "b"}]}) == ["a", "b"]


# ---------------------------------------------------------------------------
# End-to-end tool execution with the fake HTTP layer
# ---------------------------------------------------------------------------


def _submit_response(task_id: str = "t-1"):
    return FakeResponse({"output": {"task_id": task_id, "task_status": "PENDING"}})


def _succeeded_image(url: str = "https://cdn/img.png"):
    return FakeResponse({"output": {"task_status": "SUCCEEDED", "results": [{"url": url}]}})


def _succeeded_video(url: str = "https://cdn/clip.mp4"):
    return FakeResponse({"output": {"task_status": "SUCCEEDED", "video_url": url}})


def test_wan_image_happy_path_submits_polls_downloads(
    fake_requests, api_key_env, tmp_path, monkeypatch
):
    monkeypatch.setattr(_dashscope.time, "sleep", lambda _s: None)
    fake_requests.enqueue_post(_submit_response("task-img-1"))
    fake_requests.enqueue_get(FakeResponse({"output": {"task_status": "RUNNING"}}))
    fake_requests.enqueue_get(_succeeded_image())
    fake_requests.enqueue_get(FakeResponse(content=b"PNG_BYTES"))

    output_path = tmp_path / "out.png"
    result = WanImage().execute(
        {
            "prompt": "A red panda coding in neon light",
            "model": "wan2.7-image-pro",
            "n": 1,
            "size": "1024*1024",
            "seed": 42,
            "output_path": str(output_path),
        }
    )
    assert result.success, result.error
    assert output_path.read_bytes() == b"PNG_BYTES"
    assert result.data["task_id"] == "task-img-1"
    assert result.data["provider"] == "wan"
    assert result.data["model"] == "wan2.7-image-pro"
    assert result.cost_usd == pytest.approx(0.12)

    submit = fake_requests.post_calls[0]
    body = submit["json"]
    assert body["model"] == "wan2.7-image-pro"
    assert body["input"]["prompt"].startswith("A red panda")
    assert body["parameters"]["seed"] == 42


def test_wan_image_multi_image_batch_writes_all_files(
    fake_requests, api_key_env, tmp_path, monkeypatch
):
    monkeypatch.setattr(_dashscope.time, "sleep", lambda _s: None)
    fake_requests.enqueue_post(_submit_response("batch-1"))
    fake_requests.enqueue_get(
        FakeResponse({"output": {"task_status": "SUCCEEDED", "results": [
            {"url": "https://cdn/1.png"},
            {"url": "https://cdn/2.png"},
        ]}})
    )
    fake_requests.enqueue_get(FakeResponse(content=b"A"))
    fake_requests.enqueue_get(FakeResponse(content=b"B"))

    base = tmp_path / "img.png"
    result = WanImage().execute(
        {"prompt": "x", "n": 2, "output_path": str(base)}
    )
    assert result.success, result.error
    assert (tmp_path / "img_0.png").read_bytes() == b"A"
    assert (tmp_path / "img_1.png").read_bytes() == b"B"
    assert result.data["outputs"] == [str(tmp_path / "img_0.png"), str(tmp_path / "img_1.png")]


def test_wan_image_returns_error_when_key_missing(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("ALIBABA_DASHSCOPE_API_KEY", raising=False)
    result = WanImage().execute({"prompt": "x"})
    assert not result.success
    assert "DASHSCOPE_API_KEY" in (result.error or "")


def test_wan_image_rejects_unknown_model(api_key_env):
    result = WanImage().execute({"prompt": "x", "model": "not-a-real-model"})
    assert not result.success
    assert "Unknown model" in (result.error or "")


def test_qwen_image_happy_path(fake_requests, api_key_env, tmp_path, monkeypatch):
    monkeypatch.setattr(_dashscope.time, "sleep", lambda _s: None)
    fake_requests.enqueue_post(_submit_response("task-q-1"))
    fake_requests.enqueue_get(_succeeded_image("https://cdn/q.png"))
    fake_requests.enqueue_get(FakeResponse(content=b"QIMG"))

    out = tmp_path / "q.png"
    result = QwenImage().execute(
        {"prompt": "poster with 中文 caption", "model": "qwen-image-2.0-pro", "output_path": str(out)}
    )
    assert result.success, result.error
    assert out.read_bytes() == b"QIMG"
    assert result.data["provider"] == "qwen"
    assert result.cost_usd == pytest.approx(0.10)


def test_qwen_image_returns_error_when_key_missing(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("ALIBABA_DASHSCOPE_API_KEY", raising=False)
    result = QwenImage().execute({"prompt": "x"})
    assert not result.success
    assert "DASHSCOPE_API_KEY" in (result.error or "")


def test_wan_video_api_text_to_video_happy_path(
    fake_requests, api_key_env, tmp_path, monkeypatch
):
    monkeypatch.setattr(_dashscope.time, "sleep", lambda _s: None)
    fake_requests.enqueue_post(_submit_response("task-t2v-1"))
    fake_requests.enqueue_get(_succeeded_video())
    fake_requests.enqueue_get(FakeResponse(content=b"MP4_BYTES"))

    out = tmp_path / "clip.mp4"
    result = WanVideoAPI().execute(
        {
            "prompt": "A drone shot over misty mountains at dawn",
            "operation": "text_to_video",
            "model": "wan2.7-t2v",
            "duration": 5,
            "output_path": str(out),
        }
    )
    assert result.success, result.error
    assert out.read_bytes() == b"MP4_BYTES"
    assert result.data["provider"] == "wan"
    assert result.data["model"] == "wan2.7-t2v"
    assert result.data["operation"] == "text_to_video"
    assert result.data["format"] == "mp4"

    submit_url = fake_requests.post_calls[0]["url"]
    assert "text2video" in submit_url or "video-generation" in submit_url


def test_wan_video_api_image_to_video_requires_img_url(api_key_env):
    result = WanVideoAPI().execute(
        {
            "prompt": "dolphin leaps",
            "operation": "image_to_video",
            "model": "wan2.7-i2v",
        }
    )
    assert not result.success
    assert "img_url" in (result.error or "")


def test_wan_video_api_image_to_video_happy_path(
    fake_requests, api_key_env, tmp_path, monkeypatch
):
    monkeypatch.setattr(_dashscope.time, "sleep", lambda _s: None)
    fake_requests.enqueue_post(_submit_response("task-i2v-1"))
    fake_requests.enqueue_get(_succeeded_video("https://cdn/i2v.mp4"))
    fake_requests.enqueue_get(FakeResponse(content=b"I2V_BYTES"))

    out = tmp_path / "i2v.mp4"
    result = WanVideoAPI().execute(
        {
            "prompt": "The subject walks forward",
            "operation": "image_to_video",
            "model": "wan2.6-i2v-flash",
            "img_url": "https://example.com/ref.png",
            "output_path": str(out),
        }
    )
    assert result.success, result.error
    assert out.read_bytes() == b"I2V_BYTES"
    assert result.data["operation"] == "image_to_video"
    # Submit hit the image2video endpoint
    assert "image2video" in fake_requests.post_calls[0]["url"]
    # And the image URL was threaded through
    assert fake_requests.post_calls[0]["json"]["input"]["img_url"] == "https://example.com/ref.png"


def test_wan_video_api_rejects_t2v_model_on_i2v_operation(api_key_env):
    result = WanVideoAPI().execute(
        {
            "prompt": "x",
            "operation": "image_to_video",
            "model": "wan2.7-t2v",  # t2v model on i2v op
            "img_url": "https://example.com/ref.png",
        }
    )
    assert not result.success
    assert "image-to-video" in (result.error or "")


def test_wan_video_api_rejects_i2v_model_on_t2v_operation(api_key_env):
    result = WanVideoAPI().execute(
        {
            "prompt": "x",
            "operation": "text_to_video",
            "model": "wan2.7-i2v",  # i2v model on t2v op
        }
    )
    assert not result.success
    assert "text-to-video" in (result.error or "")


def test_wan_video_api_returns_error_when_key_missing(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("ALIBABA_DASHSCOPE_API_KEY", raising=False)
    result = WanVideoAPI().execute({"prompt": "x"})
    assert not result.success
    assert "DASHSCOPE_API_KEY" in (result.error or "")


def test_wan_video_api_bubbles_up_dashscope_failure(
    fake_requests, api_key_env, tmp_path, monkeypatch
):
    monkeypatch.setattr(_dashscope.time, "sleep", lambda _s: None)
    fake_requests.enqueue_post(_submit_response("task-fail"))
    fake_requests.enqueue_get(
        FakeResponse({"output": {"task_status": "FAILED", "message": "prompt rejected"}})
    )
    result = WanVideoAPI().execute(
        {"prompt": "x", "operation": "text_to_video", "model": "wan2.7-t2v", "output_path": str(tmp_path / "x.mp4")}
    )
    assert not result.success
    assert "prompt rejected" in (result.error or "")
