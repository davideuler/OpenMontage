# Video Generation Prompting — Universal Guide

## When to Use

When writing prompts for the video generation family (`video_selector`, `heygen_video`,
`wan_video`, `wan_video_api`, `hunyuan_video`, `ltx_video_local`, `ltx_video_modal`,
`cogvideo_video`).
This skill covers the universal prompt vocabulary that works across all video generation models.

For model-specific tips, see the linked guides below.

Layer 3: `.agents/skills/ai-video-gen` (HeyGen/fal.ai gateways),
`.agents/skills/ltx2` (LTX-2), `.agents/skills/dashscope` (Wan T2V / I2V via
Alibaba DashScope).

## Model-Specific Guides

| Model | Guide | Key Insight |
|-------|-------|-------------|
| **Sora 2 / Sora 2 Pro** | [OpenAI Sora 2 Cookbook](https://developers.openai.com/cookbook/examples/sora/sora2_prompting_guide) | Richest structured template. Advanced fields: lenses, filtration, grade, diegetic sound, wardrobe, finishing. |
| **VEO 3.1 / VEO 3** | [Vertex AI Prompt Guide](https://cloud.google.com/vertex-ai/generative-ai/docs/video/video-gen-prompt-guide) | Best vocabulary reference tables. 14-component prompt structure. |
| **Grok Imagine Video** | `creative/prompting/grok-prompting.md` | Best when prompts need reference-image placeholders like `<IMAGE_1>` and identity/product carryover. |
| **LTX-2** | [LTX Prompting Guide](https://docs.ltx.video/api-documentation/prompting-guide) | 6-element structure. Audio/voice prompting. Strong "what to avoid" section. |
| **HunyuanVideo 1.5** | [Tencent Prompt Handbook](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/assets/HunyuanVideo_1_5_Prompt_Handbook_EN.md) | Formula: Subject + Motion + Scene + [Shot] + [Camera] + [Lighting] + [Style] + [Atmosphere]. |
| **Runway Gen-4** | [Runway Prompting Guide](https://help.runwayml.com/hc/en-us/articles/39789879462419-Gen-4-Video-Prompting-Guide) | "Focus on motion, not appearance." One scene per clip. Simplicity wins. |
| **Kling 2.6** | [Kling Prompt Guide](https://fal.ai/learn/devs/kling-2-6-pro-prompt-guide) | 4-part structure. Supports `++emphasis++` syntax for key elements. |
| **Wan 2.7 / 2.6 / 2.5 (DashScope API)** | [Aliyun Text-to-Video](https://help.aliyun.com/zh/model-studio/text-to-video-api-reference) · [Image-to-Video](https://help.aliyun.com/zh/model-studio/image-to-video-api-reference/) | Use `wan_video_api` for cloud T2V/I2V. Handles Chinese and English prompts natively; set `operation="image_to_video"` with `img_url` for I2V. Max 10s clips. |
| **Wan 2.1 / CogVideoX (local)** | Use this generic guide | No official prompt guide. Standard cinematographic vocabulary works well. Local GPU path via `wan_video` / `cogvideo_video`. |

## Universal Prompt Formula

All video generation models respond to this structure. Include what's relevant, omit what's not.

```
[Shot type/framing] + [Camera movement] + [Subject description] +
[Action/motion in beats] + [Setting/environment] + [Lighting] +
[Style/aesthetic] + [Audio/atmosphere]
```

**Shorter prompts = more creative freedom. Longer prompts = more control.**

---

## Camera Shot Types

| Shot | When to Use |
|------|-------------|
| **Wide / establishing shot** | Open a scene, show location context |
| **Full / long shot** | Subject head-to-toe with environment |
| **Medium shot** | Waist up, balances detail with context |
| **Medium close-up** | Chest up, conversational intimacy |
| **Close-up** | Face or key object, emphasize emotion |
| **Extreme close-up** | Isolated detail (eye, drop, texture) |
| **Over-the-shoulder** | Conversation framing, connection |
| **Point-of-view (POV)** | Viewer becomes the character |
| **Bird's-eye / top-down** | Map-like overview, omniscient feel |
| **Worm's-eye view** | Looking straight up, emphasize height |
| **Dutch / canted angle** | Tilted horizon, unease or tension |
| **Low-angle** | Subject appears powerful, dominant |
| **High-angle** | Subject appears small, vulnerable |

## Camera Movements

| Movement | What It Does | Best For |
|----------|-------------|----------|
| **Static / fixed** | No movement | Dialogue, contemplation, stability |
| **Pan** (left/right) | Rotates horizontally | Revealing a scene, following action |
| **Tilt** (up/down) | Rotates vertically | Revealing height, slow reveal |
| **Dolly in / out** | Physically moves toward/away | Building tension, emphasis |
| **Truck** (left/right) | Moves sideways | Parallels subject movement |
| **Pedestal** (up/down) | Moves vertically | Smooth elevation changes |
| **Crane shot** | Sweeping vertical arcs | Epic reveals, transitions |
| **Tracking / follow** | Follows subject | Action sequences, walk-and-talk |
| **Arc shot** | Circles around subject | Dramatic emphasis, 360° reveal |
| **Zoom** (in/out) | Lens focal length change | Quick emphasis (cheaper than dolly) |
| **Whip pan** | Extremely fast pan (blurs) | Transitions, energy, surprise |
| **Handheld / shaky cam** | Unstable, human feel | Documentary, urgency, realism |
| **Aerial / drone** | High altitude, smooth | Landscapes, establishing shots |
| **Slow push-in** | Gradual forward movement | Building intimacy or tension |
| **Dolly zoom (vertigo)** | Dolly one way, zoom opposite | Disorientation, revelation |

## Lighting Vocabulary

| Term | Effect |
|------|--------|
| **Natural light** | Soft, realistic (morning sun, overcast, moonlight) |
| **Golden hour** | Warm sunlight, long shadows, romantic |
| **High-key** | Bright, even, cheerful — comedy, lifestyle |
| **Low-key** | Dark, high contrast — thriller, drama |
| **Rembrandt** | Triangle of light on cheek, classic portrait |
| **Film noir** | Deep shadows, stark highlights |
| **Volumetric** | Visible light rays through atmosphere (fog, dust) |
| **Backlighting** | Light behind subject, silhouette effect |
| **Side lighting** | Strong directional, dramatic shadows |
| **Practical lights** | In-frame sources (lamps, candles, neon signs) |
| **Rim / edge light** | Highlights subject outline, separates from background |

**Lighting direction modifiers**: key light, fill light, bounce, rim, spill, negative fill.

**Color temperature**: warm (tungsten, amber), cool (daylight, blue), mixed.

## Lens & Optical Effects

| Effect | Result |
|--------|--------|
| **Shallow depth of field** | Subject sharp, background bokeh |
| **Deep focus** | Everything sharp, foreground to background |
| **Wide-angle lens** (24-35mm) | Broader view, exaggerated perspective |
| **Telephoto** (85mm+) | Compressed perspective, subject isolation |
| **Anamorphic** | Stretched aspect, signature lens flares |
| **Lens flare** | Streaks from bright light hitting lens |
| **Rack focus** | Shift focus between subjects in-shot |
| **Fisheye** | Ultra-wide, barrel distortion |

## Style & Aesthetic References

### Cinematic Styles
- Film noir, period drama, thriller, modern romance
- Documentary, arthouse, experimental film
- Epic space opera, fantasy, horror
- 1970s romantic drama, 90s documentary-style

### Animation Styles
- Studio Ghibli / Japanese anime
- Classic Disney, Pixar-like 3D
- Stop-motion, claymation
- Hand-painted 2D/3D hybrid
- Cel-shaded, low-poly 3D

### Art Movements
- Impressionistic, surrealist, Art Deco, Bauhaus
- Watercolor, charcoal sketch, ink wash
- Graphic novel, blueprint schematic

### Film Stock / Grade
- Kodak warm grade, Fuji cool tones
- 16mm black-and-white, 35mm photochemical contrast
- Vintage grain overlay, halation on speculars
- Teal-and-orange color grade

## Temporal Effects

| Effect | Use |
|--------|-----|
| **Slow motion** | Emphasis, beauty, impact |
| **Time-lapse** | Passage of time, processes |
| **Freeze-frame** | Dramatic pause |
| **Rapid cuts** | Energy, urgency |
| **Continuous / long take** | Immersion, tension |
| **Fade in / fade out** | Scene transitions |
| **Match cut** | Visual continuity between scenes |

## Audio Descriptions

Models that support audio generation (LTX-2, Sora 2, VEO 3) respond to:

**Ambient**: wind, rain, traffic, crowd murmur, forest birds, mechanical hum
**Diegetic sound**: footsteps, door creaking, glass clinking, keyboard typing
**Voice style**: whisper, calm narration, energetic announcer, gravitas
**Music mood**: "soft piano in background", "upbeat electronic"

Put dialogue in quotation marks: `Character says: "Hello world."`

## What to Avoid

| Don't | Why | Do Instead |
|-------|-----|-----------|
| "Beautiful scene" | Too vague, no visual info | "Wet cobblestone street, warm streetlamp glow reflecting in puddles" |
| "Person moves quickly" | No visible action | "Woman sprints three steps and vaults over the railing" |
| "Cinematic look" | Every model already tries this | Specify: "anamorphic lens, shallow DOF, golden hour lighting" |
| "Sad character" | Internal states aren't visible | "Tears on cheek, shoulders slumped, staring at empty chair" |
| Readable text / logos | Models can't render text reliably | Avoid signs with text, or accept imperfect rendering |
| Complex physics | Chaotic motion causes artifacts | Keep physics simple; dancing/walking OK, explosions risky |
| Multiple characters talking | Multi-person dialogue breaks sync | One speaker per clip, or use reaction shots |
| Overloaded prompts | Too many elements = incoherent | Start simple, layer complexity one element at a time |
| Conflicting lighting | "Bright noon" + "dark shadows" | Pick one lighting setup and commit |

## Prompt Iteration Strategy

1. **Start simple** — subject + action + setting. See what the model gives you.
2. **Add one element at a time** — camera, then lighting, then style.
3. **If a shot misfires** — strip back. Freeze camera, simplify action, try again.
4. **For consistency across clips** — repeat the same style/lighting/grade description.
5. **Use seed values** — when you find a good result, save the seed for variations.
6. **For Grok reference-image video** — assign each source image a clear role in the prompt using `<IMAGE_1>`, `<IMAGE_2>`, etc.

## Wan Video via Alibaba DashScope (`wan_video_api`)

Cloud path for the Wan video family that does not require a local GPU.
Use when the project is already on the Alibaba Model Studio stack (paired
with `qwen_image` / `wan_image` / `qwen_tts`) or when GPU is unavailable.

### Models

**Text-to-video:**

| Model | Tier | ~Cost / clip | Max duration |
|-------|------|-------------|--------------|
| `wan2.7-t2v` | pro | ~$0.70 | 10s |
| `wan2.5-t2v-plus` | plus | ~$0.50 | 10s |
| `wan2.2-t2v-plus` | plus | ~$0.40 | 10s |

**Image-to-video** (set `operation="image_to_video"` and pass `img_url`):

| Model | Tier | ~Cost / clip | Max duration |
|-------|------|-------------|--------------|
| `wan2.7-i2v` | pro | ~$0.70 | 10s |
| `wan2.5-i2v-plus` | plus | ~$0.50 | 10s |
| `wan2.6-i2v-flash` | flash | ~$0.25 | 5s |

Supported sizes: `1280*720`, `720*1280`, `960*960`, `1920*1080`, `1080*1920`
(DashScope `W*H` string format, not a pixel list).

### When to use T2V vs I2V

- **T2V (`wan2.7-t2v`):** pure text prompt, the model composes a new scene.
  Good for abstract shots, establishing shots, and content where no hero frame exists.
- **I2V (`wan2.7-i2v`):** starts from an existing still (often a `wan_image` /
  `qwen_image` / `flux_image` hero) and animates it. Strongly preferred for
  keeping visual consistency across an explainer where the image stack is
  already committed to a look.

### Prompt guidance

No official Wan prompt guide exists — use the universal formula in this file.
Observed behavior on Wan 2.7:

- Responds well to English cinematographic vocabulary (shot size, movement,
  lens, lighting, style) and to Chinese prompts natively — do not pre-translate.
- `prompt_extend=true` (default) lets DashScope auto-enrich short prompts; disable
  when you want strict control over phrasing.
- For I2V, describe **motion only**. Do not re-describe what is already in the
  reference image; the model keeps identity and setting from `img_url`.
- Keep a single scene per clip; cut between shots rather than asking for cuts
  inside a prompt.
- Audio is not generated — plan narration and music in the asset stage.

### Example I2V prompt (Wan 2.7)

```
[Camera]: Slow push-in, handheld micro-jitter
[Motion]: The beekeeper tilts the frame toward camera, golden honey
          droplets tremble and release one by one into slow motion
[Lighting]: Warm late-afternoon light unchanged from reference
[Style]: Documentary cinematography, 35mm film grain, 24fps
```

Pair with `img_url` pointing to the hero image generated earlier
(`wan_image` / `qwen_image` / `flux_image` output, uploaded to a URL the
DashScope task can fetch).

## Example: Generic Prompt Template

```
[Shot]: Medium close-up, slight low angle
[Camera]: Slow dolly-in
[Subject]: A weathered fisherman in his 60s, salt-and-pepper beard,
           dark wool sweater, calloused hands gripping a rope
[Action]: He pulls the rope hand-over-hand, muscles straining,
          then pauses and looks out to sea
[Setting]: Wooden dock at dawn, calm grey ocean, distant fog bank,
           seagulls wheeling overhead
[Lighting]: Soft overcast with warm break in clouds on the horizon,
            gentle rim light from the rising sun
[Style]: Documentary cinematography, 35mm film grain,
         muted earth tones with a cold blue-grey palette
[Audio]: Rope creaking, water lapping, distant gull cries, wind
```
