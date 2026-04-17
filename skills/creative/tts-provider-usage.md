# TTS Provider Usage for OpenMontage

> How to choose among text-to-speech providers, and how to use each effectively.
> Complements `lip-sync-usage.md`, `talking-head-gen-usage.md`, and `sound-design.md`.
> Layer 3: `.agents/skills/elevenlabs`, `.agents/skills/text-to-speech`,
> `.agents/skills/dashscope` (for Qwen-TTS).

## Provider Landscape

All TTS tools live under `tools/audio/` and register with `capability="tts"`.
The `tts_selector` auto-discovers every provider in the registry — adding a
new provider tool is enough; no selector code changes needed.

| Tool | Provider | Runtime | Cost | Best For |
|------|----------|---------|------|----------|
| `elevenlabs_tts` | ElevenLabs | API | ~$0.30 / 1k chars | Highest-quality expressive English narration, voice cloning, spokesperson videos |
| `openai_tts` | OpenAI (`gpt-4o-mini-tts`) | API | ~$0.015 / 1k chars | Cheap workhorse for English narration, delivery `instructions` parameter |
| `google_tts` | Google Cloud TTS | API | ~$0.004-$0.030 / 1k chars | Localization (700+ voices, 50+ languages), SSML support, Chirp 3 HD |
| `qwen_tts` | Qwen-TTS via Alibaba DashScope | API | ~$0.0014 / 1k chars | Native Chinese narration, Chinese dialect voices (Beijing, Sichuan, Shanghai), bilingual EN/ZH |
| `piper_tts` | Piper (local) | LOCAL | Free | Offline / privacy-sensitive workflows, no API key |

### Selector

| Tool | Purpose |
|------|---------|
| `tts_selector` | Routes to the best available TTS provider based on preference, availability, and scored ranking |

## Provider Selection by Scene Type

| Brief | Primary | Why | Fallback |
|-------|---------|-----|----------|
| **English narration, production quality** | `elevenlabs_tts` | Most expressive English voices, clone support | `openai_tts` → `google_tts` |
| **Cheap English narration at scale** | `openai_tts` | ~20x cheaper than ElevenLabs, still natural | `google_tts` |
| **Chinese narration (Mandarin)** | `qwen_tts` | Native Mandarin prosody, bilingual EN/ZH voices | `google_tts` (Cmn-CN voices) |
| **Chinese dialect narration** (Beijing / Sichuan / Shanghai) | `qwen_tts` with `qwen-tts-latest` | Dedicated dialect voices (Dylan / Sunny / Jada) | None in the registry — plan fallback to standard Mandarin |
| **Localization / many languages** | `google_tts` | 700+ voices across 50+ languages, Chirp 3 HD | `elevenlabs_tts` multilingual |
| **SSML control** (pronunciations, breaks) | `google_tts` | First-class SSML support | `elevenlabs_tts` |
| **Voice cloning** (use an uploaded voice) | `elevenlabs_tts` | Only provider with cloning | — |
| **Offline / air-gapped** | `piper_tts` | No network, no API key | — |
| **Alibaba Model Studio pipelines** | `qwen_tts` | Shares `DASHSCOPE_API_KEY` with `wan_image` / `qwen_image` / `wan_video_api` | `google_tts` |

## Provider-Specific Notes

### ElevenLabs (`elevenlabs_tts`)
- **Env var:** `ELEVENLABS_API_KEY`
- Default model `eleven_multilingual_v2`. Tune `stability` (lower = more expressive) and `style` (higher = more exaggeration).
- Formats: `mp3_44100_128` (default), `mp3_44100_192`, `pcm_16000`, `pcm_24000`.
- Best combined with the `.agents/skills/elevenlabs` and `.agents/skills/text-to-speech` Layer 3 skills.

### OpenAI (`openai_tts`)
- **Env var:** `OPENAI_API_KEY`
- Default model `gpt-4o-mini-tts`. Supports an `instructions` field to shape delivery ("speak warmly, like a documentary narrator").
- Voices: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` (and newer named voices via the API).
- Good default for cost-sensitive production; fallback when ElevenLabs is unavailable.

### Google Cloud TTS (`google_tts`)
- **Env vars:** `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) **or** `GOOGLE_APPLICATION_CREDENTIALS`.
- Voice tiers (most expressive first): Chirp 3 HD → Studio → Neural2 → WaveNet → Standard.
- Chirp 3 HD and Journey voices route through the `v1beta1` endpoint automatically.
- Supports `speakingRate`, `pitch`, and SSML input.

### Qwen-TTS via Alibaba DashScope (`qwen_tts`)
- **Env vars:** `DASHSCOPE_API_KEY` (alias: `ALIYUN_DASHSCOPE_API_KEY`). Shared with `qwen_image`, `wan_image`, and `wan_video_api`.
- **Models:**
  - `qwen-tts` — stable.
  - `qwen-tts-latest` (default in OpenMontage) — latest features including Chinese dialect voices.
  - `qwen-tts-2025-05-22` — pinned snapshot for reproducibility.
- **Voices:**
  - Bilingual EN/ZH: `Cherry` (female, default), `Ethan` (male), `Chelsie` (female), `Serena` (female).
  - Chinese dialects (require `qwen-tts-latest`): `Dylan` (Beijing Mandarin), `Sunny` (Sichuan Mandarin), `Jada` (Shanghainese).
- **Protocol:** synchronous — the API returns `output.audio.url` on the first POST; the tool downloads the audio for you. No async polling, unlike `qwen_image` / `wan_video_api`.
- **Best for:** Chinese-language narration, bilingual content, and any pipeline already on the Alibaba Model Studio stack where consolidating on one API key matters.

### Piper (`piper_tts`)
- **Runtime:** local (no API key). Requires the `piper` CLI and a downloaded voice model.
- **Install:** `pip install piper-tts` + `piper --download-dir ~/.piper/models --model en_US-lessac-medium`.
- Deterministic output given the same text + model + seed.

## Using the TTS Selector

For most cases, call `tts_selector` and let it route:

```python
result = tts_selector.execute({
    "text": "Welcome to the OpenMontage production walkthrough.",
    "preferred_provider": "auto",  # or "elevenlabs", "qwen", "google_tts", "openai", "piper"
    "output_path": "projects/<slug>/assets/audio/narration-01.mp3",
})
```

Restrict the pool when you need to stay inside a particular stack:

```python
# Alibaba Model Studio only — pairs with wan_video_api / qwen_image
result = tts_selector.execute({
    "text": "欢迎来到 OpenMontage 演示。",
    "voice": "Cherry",
    "allowed_providers": ["qwen"],
    "output_path": "projects/<slug>/assets/audio/narration-01.mp3",
})

# Offline / free only
result = tts_selector.execute({
    "text": "Offline narration test.",
    "allowed_providers": ["piper"],
    "output_path": "projects/<slug>/assets/audio/narration-01.wav",
})
```

Use `operation="rank"` to preview provider scoring without generating audio.

## Cost-Quality Tradeoff

```
PRODUCTION PATH: Premium (English)
├── All narration: elevenlabs_tts (~$0.30 / 1k chars)
└── 2k-char script: ~$0.60

PRODUCTION PATH: Standard (English)
├── All narration: openai_tts (~$0.015 / 1k chars)
└── 2k-char script: ~$0.03

PRODUCTION PATH: Localization
├── Any language: google_tts (Chirp 3 HD)
└── 2k-char script: ~$0.06

PRODUCTION PATH: Alibaba Model Studio (Chinese or bilingual)
├── All narration: qwen_tts qwen-tts-latest (~$0.0014 / 1k chars)
├── 2k-char script: ~$0.003
└── Shares DASHSCOPE_API_KEY with qwen_image + wan_image + wan_video_api

PRODUCTION PATH: Offline
├── All narration: piper_tts (free, local)
└── 2k-char script: $0.00
```

## Output Consistency

When a project uses several TTS providers (e.g. Qwen for Mandarin segments
and ElevenLabs for English segments), normalize loudness in the asset stage
using `audio_mixer` / the LUFS targets defined in `sound-design.md`. Do not
ship raw provider output — it will not match across vendors.

## Common Pitfalls

1. **Cross-provider loudness drift.** Every provider masters at a different
   target. Always run the `audio_enhance` / `audio_mixer` normalize step.
2. **Ignoring dialect requirements.** `qwen_tts` dialect voices (Dylan / Sunny / Jada)
   only work with `qwen-tts-latest`. The selector will surface the error if you
   pair a dialect voice with a non-latest model.
3. **Assuming Chinese quality from English-first providers.** ElevenLabs and
   OpenAI can say Mandarin, but prosody is weaker than Qwen-TTS or Google
   Chirp 3 HD Cmn-CN voices. Prefer native providers for native languages.
4. **Hardcoding a provider.** Always go through `tts_selector` so a missing
   API key falls back instead of crashing the pipeline.
5. **Skipping SSML** when breaks, emphasis, or pronunciations matter. Use
   `google_tts` or `elevenlabs_tts` with SSML for production-grade reads.
