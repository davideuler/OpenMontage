# Image Provider Usage for OpenMontage

> How to choose between image generation and stock providers, and how to use each effectively.
> Supplements the existing `image-gen-usage.md` (which covers FLUX prompting in depth).
> Layer 3: `.agents/skills/flux-best-practices`, `.agents/skills/bfl-api`,
> `.agents/skills/grok-media`, `.agents/skills/dashscope` (for Qwen-Image / Wan-Image).

## Provider Landscape

### Generation Providers (AI creates the image)

| Tool | Provider | Cost | Speed | Best For |
|------|----------|------|-------|----------|
| `flux_image` | FLUX 2 Pro via fal.ai | ~$0.03-0.05 | ~5-10s | Photorealism, general purpose, workhorse |
| `grok_image` | Grok Imagine Image (xAI) | $0.02/output + $0.002/input edit image | ~5-15s | Image edits, style transfer, multi-image compositing |
| `openai_image` | GPT Image 1 (OpenAI) | ~$0.01-0.17 | ~5-15s | Complex instructions, text in images, multi-element |
| `recraft_image` | Recraft V4 via fal.ai | ~$0.04-0.25 | ~5-10s | Logos, SVG vectors, brand assets, text rendering (see caveat below) |
| `qwen_image` | Qwen-Image 2.0 via Alibaba DashScope | $0.03-0.08 | ~20-30s | Chinese/English prompts, legible text rendered inside images, posters, bilingual illustrations |
| `wan_image` | Wan 2.7 / 2.2 via Alibaba DashScope | $0.02-0.06 | ~20-30s | Photorealism and illustration with strong Chinese-language prompting, Wan/Qwen stack parity |
| `local_diffusion` | Stable Diffusion (local) | Free | ~30s+ | Offline, privacy, free |
| `image_gen` | Multi (legacy, deprecated) | Varies | Varies | **Deprecated** — use `image_selector` or per-provider tools |

### Stock Providers (search and download existing images)

| Tool | Provider | Cost | Speed | Best For |
|------|----------|------|-------|----------|
| `pexels_image` | Pexels | Free | ~2-5s | High-quality photography, color filtering |
| `pixabay_image` | Pixabay | Free | ~2-5s | Large library, category filtering, illustrations |

### Selector

| Tool | Purpose |
|------|---------|
| `image_selector` | Routes to the best available provider based on preference and availability |

## Provider Selection by Scene Type

| Scene Type | Primary Provider | Why | Fallback |
|-----------|-----------------|-----|----------|
| **Real-world photo** (city, nature, people) | `pexels_image` | Real photos > AI for realism | `pixabay_image` → `flux_image` |
| **Technical diagram** | `diagram_gen` | Structured, editable | `flux_image` with diagram prompt |
| **Abstract/conceptual illustration** | `flux_image` | AI excels at custom concepts | `openai_image` |
| **Style transfer / repaint of an existing image** | `grok_image` | Native edit flow, strong promptable transforms | `openai_image` |
| **Multi-image merge / composite** | `grok_image` | Can combine multiple source images into one scene | `openai_image` |
| **Logo or brand asset** | `recraft_image` | SVG support, text accuracy | `openai_image` |
| **Image with text/labels** | `openai_image` | Best text rendering (GPT Image 1) | `qwen_image` → `recraft_image` |
| **Chinese text inside the image** | `qwen_image` | Qwen is the strongest at legible Chinese characters | `openai_image` |
| **Chinese / bilingual prompts** | `qwen_image` or `wan_image` | Native Chinese prompting via Alibaba Model Studio | `flux_image` |
| **Alibaba Model Studio pipelines** | `wan_image` | Keeps image + video stack on Wan 2.7 | `qwen_image` |
| **Complex multi-element composition** | `openai_image` | Best instruction following | `flux_image` |
| **Hero image (key visual)** | `flux_image` | Highest visual quality | `openai_image` |
| **Thumbnail** | `flux_image` or `recraft_image` | Needs to be eye-catching | — |
| **Budget/free project** | `pexels_image` or `pixabay_image` | Free, immediate | `local_diffusion` |
| **Offline/air-gapped** | `local_diffusion` | No network needed | — |

## Provider-Specific Caveats

### Recraft V4 via fal.ai
- **`style` parameter causes 422 errors** (as of 2026-04). The `style` enum values (`digital_illustration`, `realistic_image`, etc.) are rejected by fal.ai's Recraft V4 endpoint. **Workaround:** encode style direction in the prompt text instead (e.g. "digital illustration of a tooth cross-section" rather than `style="digital_illustration"`). The `image_size` and `colors` parameters work fine.
- **Text rendering is unreliable for exact business names.** Recraft (like all AI image models) may hallucinate wrong text. For any scene where text must be verbatim (CTA screens, business names, phone numbers), use Remotion `text_card` instead of generating an image with text.

### Qwen-Image via Alibaba DashScope (`qwen_image`)
- **Models:** `qwen-image-2.0-pro` (~$0.08), `qwen-image-2.0` (~$0.05, default), `qwen-image-plus` (~$0.04), `qwen-image` (~$0.03).
- **Sizes:** use the DashScope `W*H` format (e.g. `1024*1024`, `1664*928`, `928*1664`). Not pixels-list — literally `width*height` with a `*`.
- **Chinese in images:** Qwen is the strongest mainstream model for rendering legible Chinese characters inside an image. Prefer it over FLUX or DALL-E when the brief requires Chinese signage, posters, or overlays.
- **Async API:** submission returns a `task_id`; the tool polls for you. Expect 20-30s latency.
- **Auth:** set `DASHSCOPE_API_KEY` (alias: `ALIYUN_DASHSCOPE_API_KEY`).

### Wan-Image via Alibaba DashScope (`wan_image`)
- **Models:** `wan2.7-image-pro` (~$0.06), `wan2.7-image` (~$0.04, default), `wan2.2-t2i-plus` (~$0.03), `wan2.2-t2i-flash` (~$0.02).
- **Sizes:** DashScope `W*H` format including widescreen `1440*810` and portrait `810*1440` for video frames.
- **Best for** photoreal and illustrative images when the pipeline is already using Wan 2.7 video; keeps the image+video stack on one provider.
- **Chinese prompts** work natively and often better than English translations.
- **Auth:** set `DASHSCOPE_API_KEY` (shared with `qwen_image`, `qwen_tts`, and `wan_video_api`).

## Cost-Quality Tradeoff

```

PRODUCTION PATH: Premium
├── Hero images: flux_image ($0.05/img)
├── Supporting visuals: flux_image ($0.03/img)
├── Text overlays: openai_image ($0.04/img)
├── B-roll stills: pexels_image ($0.00)
└── Total for 10 images: ~$0.35

PRODUCTION PATH: Standard
├── All generated: flux_image ($0.03/img)
├── B-roll stills: pexels_image ($0.00)
└── Total for 10 images: ~$0.25

PRODUCTION PATH: Alibaba Model Studio (Chinese-native or DashScope-only)
├── Hero + supporting visuals: wan_image wan2.7-image ($0.04/img)
├── Text-inside-image scenes: qwen_image qwen-image-2.0 ($0.05/img)
├── B-roll stills: pexels_image ($0.00)
└── Total for 10 images: ~$0.30 (single API key, pairs with wan_video_api + qwen_tts)

PRODUCTION PATH: Budget
├── All stock: pexels_image + pixabay_image ($0.00)
├── Diagrams: diagram_gen ($0.00)
└── Total: $0.00

PRODUCTION PATH: Offline
├── All generated: local_diffusion ($0.00)
├── Diagrams: diagram_gen ($0.00)
└── Total: $0.00 (but slower, lower quality)
```

Use `generation_mode="edit"` when the task starts from an existing image and should route only to edit-capable providers.

## Using the Image Selector

For most cases, use `image_selector` and let it route:

```python
# The selector finds the best available provider
result = image_selector.execute({
    "prompt": "aerial view of a modern data center",
    "preferred_provider": "auto",  # or "flux", "pexels", etc.
    "output_path": "assets/images/scene-3.png"
})
```

Override with `preferred_provider` when you know which provider is best for the scene type.
Use `allowed_providers` to restrict to free or local options:

```python
# Budget mode: only free providers
result = image_selector.execute({
    "prompt": "server room interior",
    "allowed_providers": ["pexels", "pixabay", "local_diffusion"],
    "output_path": "assets/images/scene-3.jpg"
})
```

## Consistency Across Mixed Sources

When mixing stock and generated images in the same video, visual consistency is the challenge.

### Strategy: Color Grade Everything
Apply the playbook's color grading LUT to both stock and generated images in the compose stage.
This unifies the look. The `color_grade` enhancement tool handles this.

### Strategy: Match The Visual Identity, Not A Copied Prefix
When generating images, adapt the playbook's mood, palette, texture, and medium
into a short scene-specific anchor. When searching stock, filter for the same
emotional and visual qualities: color, lighting, environment, composition, era,
and texture. Use the playbook as a consistency source, not a script to paste.

### Strategy: Avoid Mixing Styles Within a Scene
Don't use a stock photo for one element and an AI illustration for another in the same scene.
Keep each scene internally consistent — all stock or all generated.
