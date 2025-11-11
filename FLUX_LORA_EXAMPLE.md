# FLUX + LoRA Usage Example

## Prerequisites

1. Set your Replicate API token in `.env`:
```bash
REPLICATE_API_TOKEN=your-token-here
```

2. Find the FLUX LoRA model version ID from Replicate:
   - Visit: https://replicate.com/lucataco/flux-dev-lora
   - Click on "Versions" tab
   - Copy the version ID (e.g., `a22c463f959a635f1a4ff7e4d35868)

## Basic Usage

### Simple text generation (no LoRA)
```bash
replicate-runner replicate run-model \
  lucataco/flux-dev-lora \
  <version-id> \
  --param prompt:"a beautiful landscape"
```

### With HuggingFace LoRA
```bash
replicate-runner replicate run-model \
  lucataco/flux-dev-lora \
  <version-id> \
  --param prompt:"a photo of TOK, person, portrait" \
  --param hf_lora:"steveant/steve-lora-v1.1"
```

### Full FLUX LoRA Configuration
```bash
replicate-runner replicate run-model \
  lucataco/flux-dev-lora \
  <version-id> \
  --param prompt:"a photo of TOK, person, professional headshot, studio lighting" \
  --param hf_lora:"steveant/steve-lora-v1.1" \
  --param lora_scale:0.8 \
  --param num_outputs:1 \
  --param num_inference_steps:28 \
  --param guidance_scale:3.5 \
  --param width:1024 \
  --param height:1024 \
  --param seed:42 \
  --param disable_safety_checker:false
```

## Parameter Reference

### Common FLUX Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | *required* | Text description of desired image |
| `hf_lora` | string | - | HuggingFace LoRA model ID (e.g., "user/model-name") |
| `lora_scale` | float | 1.0 | LoRA influence strength (0.0-2.0) |
| `num_outputs` | int | 1 | Number of images to generate |
| `num_inference_steps` | int | 28 | Quality vs speed tradeoff |
| `guidance_scale` | float | 3.5 | How closely to follow prompt |
| `width` | int | 1024 | Output width in pixels |
| `height` | int | 1024 | Output height in pixels |
| `seed` | int | - | Random seed for reproducibility |
| `disable_safety_checker` | bool | false | Disable NSFW filter |

## Type Inference Examples

The CLI automatically infers types:

```bash
# String (default)
--param prompt:"hello world"

# Integer
--param steps:28
--param width:1024

# Float
--param lora_scale:0.8
--param guidance_scale:3.5

# Boolean
--param disable_safety_checker:true
--param enable_feature:false

# Lists (for advanced use)
--param sizes:[512,768,1024]
```

## Tips

1. **LoRA Scale**: Start with `0.8` and adjust:
   - Lower (0.4-0.6) = subtle influence
   - Medium (0.7-0.9) = balanced
   - Higher (1.0-1.5) = strong influence

2. **Quality Settings**:
   - Fast: `num_inference_steps:20`
   - Balanced: `num_inference_steps:28`
   - High quality: `num_inference_steps:50`

3. **Trigger Words**: Use your LoRA's trigger word in the prompt (e.g., "TOK" for person LoRAs)

4. **Reproducibility**: Use `--param seed:42` to get consistent results

## Using Your Published LoRAs

After publishing with `hf publish-hf-lora`:

```bash
# List your models
replicate-runner hf list-models

# Use in FLUX
replicate-runner replicate run-model \
  lucataco/flux-dev-lora \
  <version-id> \
  --param prompt:"a photo of TOK, person, portrait" \
  --param hf_lora:"yourusername/your-lora-name"
```

## Troubleshooting

**Model not found**: Ensure the version ID is correct and up to date

**LoRA not loading**: Verify HuggingFace repo is public or you have access

**Poor results**:
- Adjust `lora_scale` (try 0.6-1.2)
- Increase `num_inference_steps`
- Include trigger word in prompt
- Try different `guidance_scale` values
