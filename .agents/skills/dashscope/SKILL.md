---
name: dashscope
description: |
  Alibaba DashScope (Bailian / Model Studio) API integration guide for the Wan and Qwen family of generation models. Use when working with: (1) `wan_image` text-to-image, (2) `qwen_image` text-to-image, (3) `wan_video_api` text-to-video or image-to-video (Wan 2.7 / 2.6 / 2.5), (4) `qwen_tts` text-to-speech including Chinese dialect voices. Covers auth, the async task pattern (POST + poll), the synchronous TTS pattern, and the shared helpers in `tools/_dashscope.py`.
metadata:
  openclaw:
    requires:
      env_any:
        - DASHSCOPE_API_KEY
        - ALIYUN_DASHSCOPE_API_KEY
    primaryEnv: DASHSCOPE_API_KEY
---

# Alibaba DashScope (Bailian / Model Studio)

DashScope is Alibaba's Model Studio API that serves the Wan and Qwen
generation families — images, videos, and text-to-speech. A single API key
unlocks all four OpenMontage tools built on it:

| Tool | Capability | Operation pattern |
|------|-----------|-------------------|
| `wan_image` | `image_generation` | Async task (POST + poll) |
| `qwen_image` | `image_generation` | Async task (POST + poll) |
| `wan_video_api` | `video_generation` (text-to-video AND image-to-video) | Async task (POST + poll) |
| `qwen_tts` | `tts` | **Synchronous** (POST → audio URL → download) |

Do not duplicate HTTP plumbing. Always go through `tools/_dashscope.py`,
which centralizes auth, headers, task submission, polling, and asset download.

## Authentication

Set either `DASHSCOPE_API_KEY` (canonical) or `ALIYUN_DASHSCOPE_API_KEY`
(alias used by some Aliyun docs). Get one at
<https://bailian.console.aliyun.com/?apiKey=1#/api-key>.

```python
from tools import _dashscope as dashscope
api_key = dashscope.get_api_key()  # reads both env names
```

In OpenMontage the tools' `get_status()` already checks this; agents should
go through the selector (`image_selector`, `video_selector`, `tts_selector`)
rather than calling the DashScope tools directly.

## Endpoints

| Purpose | URL |
|---------|-----|
| Base | `https://dashscope.aliyuncs.com/api/v1` |
| Text-to-image (Wan, Qwen-Image) | `/services/aigc/text2image/image-synthesis` |
| Text-to-video (Wan T2V) | `/services/aigc/video-generation/video-synthesis` |
| Image-to-video (Wan I2V) | `/services/aigc/video-generation/video-synthesis` with `operation` field |
| TTS (Qwen-TTS) | `/services/aigc/multimodal-generation/generation` |
| Task status | `/tasks/{task_id}` |

## Async Task Pattern (images and videos)

Image and video endpoints are **asynchronous**:

1. POST with header `X-DashScope-Async: enable` → returns `task_id`.
2. GET `/tasks/{task_id}` every 5s until `task_status` is terminal.
3. Terminal success: `SUCCEEDED`. Terminal failure: `FAILED`, `CANCELED`, `UNKNOWN`.
4. Success payload includes result URLs under `output.results[]` (images) or
   `output.video_url` / `output.results[0].video_url` (videos).
5. Download with a plain `GET` (no auth header needed for the signed URL).

The helpers in `tools/_dashscope.py` implement all of this:

```python
from tools import _dashscope as dashscope

task_id = dashscope.submit_async_task(
    dashscope.TEXT_TO_IMAGE_ENDPOINT,
    payload,
    api_key,
)
output = dashscope.poll_task(task_id, api_key)           # raises on failure/timeout
urls = dashscope.extract_result_urls(output, "url")       # "url" for images
# urls = dashscope.extract_result_urls(output, "video_url")  # for videos
dashscope.download_asset(urls[0], Path("out.png"))
```

Use `submit_async_task` + `poll_task` + `extract_result_urls` + `download_asset`.
**Never reimplement the HTTP dance.**

## Synchronous TTS Pattern (Qwen-TTS)

Qwen-TTS does not use the async task pattern. The POST returns the audio
URL in the first response body:

```python
import requests
from tools import _dashscope as dashscope

resp = requests.post(
    f"{dashscope.DASHSCOPE_BASE_URL}/services/aigc/multimodal-generation/generation",
    headers=dashscope.build_headers(api_key, async_task=False),   # no X-DashScope-Async
    json={"model": "qwen-tts-latest", "input": {"text": "...", "voice": "Cherry"}},
    timeout=120,
)
body = resp.json()
# Top-level shape:    body["output"]["audio"]["url"]
# Nested choices:     body["output"]["choices"][0]["message"]["content"][i]["audio"]["url"]
```

`QwenTTS._extract_audio_url(body)` handles both response shapes; reuse it when
rolling anything custom.

## Models

### Wan-Image (`wan_image`)

| Model | Tier | ~$/image |
|-------|------|----------|
| `wan2.7-image-pro` | pro | $0.06 |
| `wan2.7-image` | standard (default) | $0.04 |
| `wan2.2-t2i-plus` | plus | $0.03 |
| `wan2.2-t2i-flash` | flash | $0.02 |

Sizes use the DashScope `W*H` string (e.g. `1024*1024`, `1440*810`,
`810*1440`, `1664*928`, `928*1664`). Not `{width, height}`, not a pixel list.
Accepts `negative_prompt`, `seed`, `n` (1-4), `prompt_extend` (bool).

### Qwen-Image (`qwen_image`)

| Model | Tier | ~$/image | Notable |
|-------|------|----------|---------|
| `qwen-image-2.0-pro` | pro | $0.08 | |
| `qwen-image-2.0` | standard (default) | $0.05 | |
| `qwen-image-plus` | plus | $0.04 | |
| `qwen-image` | base | $0.03 | |

Strongest public model for rendering **legible Chinese characters inside an
image** (posters, signage, overlays). Accepts the same parameters as Wan-Image
plus first-class Chinese prompt support.

### Wan-Video via API (`wan_video_api`)

Text-to-video:

| Model | Tier | ~$/clip | Max duration |
|-------|------|---------|--------------|
| `wan2.7-t2v` | pro (default) | $0.70 | 10s |
| `wan2.5-t2v-plus` | plus | $0.50 | 10s |
| `wan2.2-t2v-plus` | plus | $0.40 | 10s |

Image-to-video (set `operation="image_to_video"` and pass `img_url` — the URL
must be publicly reachable by DashScope's task workers; upload a local image
to a public bucket or a signed URL beforehand):

| Model | Tier | ~$/clip | Max duration |
|-------|------|---------|--------------|
| `wan2.7-i2v` | pro | $0.70 | 10s |
| `wan2.5-i2v-plus` | plus | $0.50 | 10s |
| `wan2.6-i2v-flash` | flash | $0.25 | 5s |

Sizes: `1280*720`, `720*1280`, `960*960`, `1920*1080`, `1080*1920`. Audio is
not generated — handle narration and music in the asset stage.

### Qwen-TTS (`qwen_tts`)

| Model | Notes |
|-------|-------|
| `qwen-tts-latest` | OpenMontage default; dialect voices require this. |
| `qwen-tts` | Stable baseline. |
| `qwen-tts-2025-05-22` | Pinned snapshot for reproducibility. |

Voices:

- Bilingual EN/ZH: `Cherry` (female, default), `Ethan` (male), `Chelsie` (female), `Serena` (female).
- Chinese dialects (`qwen-tts-latest` only): `Dylan` (Beijing), `Sunny` (Sichuan), `Jada` (Shanghai).

Pricing is per 1k characters of input (~$0.0014 / 1k chars) — the cheapest
TTS provider in the OpenMontage stack.

## Prompt Guidance

- Both Chinese and English prompts work natively. **Do not pre-translate
  Chinese briefs into English** — Wan and Qwen handle Chinese first-class.
- Keep `prompt_extend=true` when you want DashScope to enrich short prompts.
  Disable it (`prompt_extend=false`) when you need verbatim control.
- For image-to-video: describe **motion only**. Do not re-describe what is
  already in the reference image; identity and setting carry over from
  `img_url`.
- Follow the universal cinematography vocabulary from
  `skills/creative/video-gen-prompting.md` — Wan responds well to the
  standard shot / camera / lighting / style fields.

## Common Pitfalls

1. **Reimplementing HTTP / polling.** Always use `tools/_dashscope.py`.
2. **Forgetting `X-DashScope-Async: enable`** on image/video POST — without
   it you get a sync response that may time out.
3. **Adding `X-DashScope-Async` to Qwen-TTS** — TTS is synchronous; setting
   the header is harmless but misleading. Pass `async_task=False` to
   `build_headers` for TTS.
4. **Passing a local file path as `img_url`** for I2V. The URL must be
   publicly fetchable; use the `image_path` alias only after uploading.
5. **Pairing a dialect voice with a non-`latest` TTS model.** `Dylan`,
   `Sunny`, and `Jada` require `qwen-tts-latest`. The tool will error before
   calling the API.
6. **Using `{width, height}` for size.** DashScope expects the `W*H` string.
7. **Downloading audio/video with the DashScope `Authorization` header.**
   The result URL is pre-signed; use `dashscope.download_asset(url, path)`
   which omits the bearer token.

## When to Prefer the DashScope Stack

- The project brief is in **Chinese** (or mixes Chinese and English).
- The deliverable requires **legible Chinese text inside images** — Qwen-Image is the strongest option.
- The user wants to **consolidate on a single API key** across image + video + TTS.
- The project needs **Chinese dialect narration** (Beijing / Sichuan / Shanghai).
- A GPU-free cloud path is needed for Wan video generation (use `wan_video_api`, not the local `wan_video`).

## References

- Wan text-to-image: <https://help.aliyun.com/zh/model-studio/text-to-image>
- Qwen-Image: <https://help.aliyun.com/zh/model-studio/qwen-image-api>
- Wan text-to-video: <https://help.aliyun.com/zh/model-studio/text-to-video-api-reference>
- Wan image-to-video: <https://help.aliyun.com/zh/model-studio/image-to-video-api-reference/>
- Qwen-TTS: <https://help.aliyun.com/zh/model-studio/qwen-tts-api>
- OpenMontage helpers: `tools/_dashscope.py`
