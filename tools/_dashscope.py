"""Shared helpers for Alibaba DashScope (Bailian / Model Studio) API.

DashScope generation endpoints for Wan and Qwen image/video use an async task
pattern:

  1. POST a job with header ``X-DashScope-Async: enable`` -> returns ``task_id``
  2. GET ``/api/v1/tasks/{task_id}`` until ``task_status`` is terminal.

Docs:
  - Wan text-to-image:   https://help.aliyun.com/zh/model-studio/text-to-image
  - Qwen text-to-image:  https://help.aliyun.com/zh/model-studio/qwen-image-api
  - Image-to-video:      https://help.aliyun.com/zh/model-studio/image-to-video-api-reference/
  - Text-to-video:       https://help.aliyun.com/zh/model-studio/text-to-video-api-reference
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"

TEXT_TO_IMAGE_ENDPOINT = f"{DASHSCOPE_BASE_URL}/services/aigc/text2image/image-synthesis"
VIDEO_SYNTHESIS_ENDPOINT = f"{DASHSCOPE_BASE_URL}/services/aigc/video-generation/video-synthesis"
TASKS_ENDPOINT = f"{DASHSCOPE_BASE_URL}/tasks"

DEFAULT_POLL_INTERVAL_SECONDS = 5
DEFAULT_POLL_TIMEOUT_SECONDS = 600

TERMINAL_SUCCESS = {"SUCCEEDED"}
TERMINAL_FAILURE = {"FAILED", "CANCELED", "UNKNOWN"}


def get_api_key() -> str | None:
    """Return the configured DashScope API key, if any.

    Accepts the canonical ``DASHSCOPE_API_KEY`` and the alias
    ``ALIYUN_DASHSCOPE_API_KEY`` used by some examples in Aliyun docs.
    """
    return (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("ALIYUN_DASHSCOPE_API_KEY")
    )


def build_headers(api_key: str, *, async_task: bool = True) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if async_task:
        headers["X-DashScope-Async"] = "enable"
    return headers


def submit_async_task(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    *,
    timeout: int = 60,
) -> str:
    """Submit an async DashScope task and return its task_id."""
    import requests

    resp = requests.post(
        url,
        headers=build_headers(api_key, async_task=True),
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    output = body.get("output") or {}
    task_id = output.get("task_id") or body.get("task_id")
    if not task_id:
        raise RuntimeError(f"DashScope task did not return a task_id: {body!r}")
    return task_id


def poll_task(
    task_id: str,
    api_key: str,
    *,
    poll_interval: int = DEFAULT_POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_POLL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Poll a DashScope task until it reaches a terminal state.

    Returns the full ``output`` dict of the completed task.
    Raises :class:`RuntimeError` on failure or timeout.
    """
    import requests

    deadline = time.monotonic() + timeout_seconds
    url = f"{TASKS_ENDPOINT}/{task_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        output = body.get("output") or {}
        status = output.get("task_status") or body.get("task_status") or "UNKNOWN"
        status = status.upper()
        if status in TERMINAL_SUCCESS:
            return output
        if status in TERMINAL_FAILURE:
            message = output.get("message") or body.get("message") or status
            raise RuntimeError(f"DashScope task {task_id} failed: {message}")
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"DashScope task {task_id} did not complete within {timeout_seconds}s"
            )
        time.sleep(poll_interval)


def download_asset(url: str, output_path: Path, *, timeout: int = 120) -> Path:
    """Download a remote asset (image/video) to ``output_path``."""
    import requests

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
    return output_path


def extract_result_urls(output: dict[str, Any], key: str) -> list[str]:
    """Pull result URLs out of a DashScope response.

    Image responses return ``output.results: [{url: ...}, ...]``; video
    responses return ``output.video_url`` or ``output.results[0].video_url``.
    The caller passes the key it expects (``url`` for images, ``video_url``
    for videos); we also fall back to a flat value when present.
    """
    flat = output.get(key)
    if isinstance(flat, str):
        return [flat]
    urls: list[str] = []
    for item in output.get("results") or []:
        if isinstance(item, dict):
            value = item.get(key) or item.get("url")
            if isinstance(value, str):
                urls.append(value)
    return urls


INSTALL_INSTRUCTIONS = (
    "Set DASHSCOPE_API_KEY to your Alibaba Model Studio (Bailian) API key.\n"
    "  Get one at https://bailian.console.aliyun.com/?apiKey=1#/api-key"
)
