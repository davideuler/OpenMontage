"""Shared helper for Alibaba DashScope async generation APIs.

DashScope (Aliyun Model Studio) exposes text-to-image, text-to-video, and
image-to-video via a three-step async protocol:

1. POST the generation request with header ``X-DashScope-Async: enable`` ->
   returns a ``task_id``.
2. GET ``/api/v1/tasks/{task_id}`` until the task reaches a terminal state
   (``SUCCEEDED``, ``FAILED``, ``CANCELED``, ``UNKNOWN``).
3. Download the produced image/video asset from the URLs in the result.

This module centralizes the HTTP + polling mechanics so the per-model tool
classes (``wan_image``, ``qwen_image``, ``wan_video_api``) stay thin.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"

TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"}


class DashScopeError(RuntimeError):
    """Raised for any non-recoverable DashScope API failure."""


def get_api_key() -> Optional[str]:
    """Return the DashScope API key from environment, if set."""
    return os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("ALIBABA_DASHSCOPE_API_KEY")


def install_instructions() -> str:
    return (
        "Set DASHSCOPE_API_KEY to your Alibaba Cloud Model Studio API key.\n"
        "  Get one at https://bailian.console.aliyun.com/?apiKey=1"
    )


def submit_task(
    endpoint: str,
    payload: dict[str, Any],
    api_key: str,
    *,
    timeout: int = 60,
) -> str:
    """Submit an async DashScope task and return its ``task_id``.

    ``endpoint`` is the path relative to ``DASHSCOPE_BASE_URL`` — for example
    ``services/aigc/text2image/image-synthesis``.
    """
    import requests

    url = f"{DASHSCOPE_BASE_URL}/{endpoint.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-DashScope-Async": "enable",
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    task_id = (data.get("output") or {}).get("task_id")
    if not task_id:
        raise DashScopeError(f"No task_id in DashScope response: {data}")
    return task_id


def poll_task(
    task_id: str,
    api_key: str,
    *,
    timeout: int = 600,
    initial_interval: float = 3.0,
    max_interval: float = 15.0,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """Poll a DashScope task until it reaches a terminal state.

    Returns the full ``output`` block. Raises :class:`DashScopeError` when the
    task ends in a non-``SUCCEEDED`` terminal state. ``sleep`` is injectable
    to keep unit tests fast.
    """
    import requests

    url = f"{DASHSCOPE_BASE_URL}/tasks/{task_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.time() + timeout
    interval = initial_interval

    while True:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        output = payload.get("output") or {}
        status = (output.get("task_status") or "").upper()

        if status == "SUCCEEDED":
            return output
        if status in TERMINAL_STATUSES:
            message = output.get("message") or output.get("code") or payload
            raise DashScopeError(f"DashScope task {task_id} ended as {status}: {message}")

        if time.time() >= deadline:
            raise DashScopeError(f"DashScope task {task_id} timed out after {timeout}s (last status={status!r})")

        sleep(min(interval, max(0.5, deadline - time.time())))
        interval = min(interval * 1.5, max_interval)


def download_asset(url: str, *, timeout: int = 180) -> bytes:
    """Download the bytes for a produced asset URL."""
    import requests

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def collect_asset_urls(output: dict[str, Any], *, key: str = "url") -> list[str]:
    """Extract asset URLs from a successful DashScope output block.

    DashScope varies across capabilities:
      - text-to-image returns ``output.results = [{"url": "..."}, ...]``
      - text-to-video / image-to-video returns ``output.video_url`` (single)
        or ``output.results = [{"url": "..."}]`` depending on the model.
    """
    urls: list[str] = []
    results = output.get("results")
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            value = item.get(key) or item.get("url")
            if value:
                urls.append(value)
    for single_key in ("video_url", "image_url", "url"):
        value = output.get(single_key)
        if value and value not in urls:
            urls.append(value)
    return urls
