# Hybrid Profiles & Prompt Wizard Spec

## 1. Configuration & Profiles

### 1.1 File hierarchy
- Package defaults: `replicate_runner/config/profiles.yaml`
- Workspace overrides: `./config/profiles.yaml`
- User overrides: `${XDG_CONFIG_HOME:-~/.config}/replicate-runner/profiles.yaml`

### 1.2 Loading logic
- Load YAML from package → workspace → user
- Deep merge by profile name with explicit type rules:
  - Scalars (str/int/bool/etc.) override previous values
  - Dicts merge recursively
  - Lists replace the previous list entirely (no automatic append)
- Record which file contributed each profile so `profile show` can display provenance

### 1.3 Schema
```yaml
profiles:
  profile_name:
    description: "Short summary"
    model: "black-forest-labs/flux-dev-lora"
    version: null
    lora: "huggingface.co/steveant/audra-flux-v1"
    trigger: "audra"
    prompt_template: "{trigger}, couture street scene, {persona_action}"
    defaults:
      params:
        lora_scale: 1.2
        num_outputs: 4
        num_inference_steps: 40
        aspect_ratio: "9:16"
        guidance: 4.0
      subject: "a couture street portrait"
      persona_tokens: ["audra"]
      persona_enabled: true
```

## 2. CLI Commands

### 2.1 `explore` (guided discovery)
- `explore models`: list Replicate models with metadata, optional `--save-profile`
- `explore loras`: list HF LoRAs (from `loras.yaml` + HF API), optional `--save-profile`

### 2.2 `profile` management
- `profile list/show/save/delete/run`
- `profile save --scope package|workspace|user` (default user)
  - Without `--scope`, we always write to `${XDG_CONFIG_HOME:-~/.config}/replicate-runner/profiles.yaml`
  - With `--scope`, we write to that file if writable; otherwise error with guidance
- Saving updates only provided fields at the chosen scope; other fields inherit from lower layers
- To clear a value defined at that scope, `profile save` supports `--unset field.path` (dot notation) which removes the key so it falls back to lower layers. Users editing YAML manually can set the field to `null` or delete it entirely for the same effect.

## 3. Running Profiles
- `profile run <name>` resolves merged profile, injects trigger, persona actions, defaults
- CLI overrides still accepted. Supported flags:
  - `--param key:value` (multi-use) for direct model inputs
  - `--prompt "text"` to override the generated prompt string
  - `--model`, `--lora`, `--subject`, `--mood`, `--action`, `--camera`, `--lighting`, `--base-model-only` mirroring the wizard flags so profiles can be run with the same overrides non-interactively
- Templates may include `{persona_action}` to opt into action injection. If the final prompt already contains the literal marker `(persona_action)`, injection is skipped to avoid duplication.
- Persona injection can be disabled with `--no-persona-action` (affects `profile run` and `prompt wizard`) or per profile via `defaults.persona_enabled: false`. Precedence: CLI flag > profile field > default (enabled).
- When persona injection is disabled and the template contains `{persona_action}`, replace the token with an empty string unless an explicit `--action` was provided, in which case use that text.
- `--base-model-only` removes any LoRA configured on the profile. Supplying the flag acts as the acknowledgement that the model will run without additional weights; no extra prompt is shown. Passing both `--lora` and `--base-model-only` is invalid and results in an error; wizard flows prevent this combination. Wizard-generated `--run` commands include the flag whenever the user chose to drop the LoRA.

## 4. Prompt Wizard
- `prompt wizard [--profile name] [--model ... --lora ... --subject ... --mood ... --action ... --camera ... --lighting ... --base-model-only]`
- Priority: CLI flags override interactive answers, which override profile defaults
- If no profile is provided, wizard prompts for model + lora or errors when `--no-interactive` is set
- When no profile supplies `prompt_template`, the wizard falls back to `"{subject_or_trigger}, {mood}, while she is {persona_action} with {camera} lighting"`. `{subject_or_trigger}` resolves first to the profile trigger, then to the subject text. If neither exists we error.
- Required fields:
  - Model (explicit via `--model` or inherited from profile)
  - Prompt subject whenever `{subject}` appears in the resolved template and no `defaults.subject` is provided. For `{subject_or_trigger}`, the subject is only required if no trigger is available.
  - Either a LoRA reference (`--lora` or profile `lora`) or the `--base-model-only` flag confirming no LoRA should be used
- Missing required inputs under `--no-interactive` trigger a clear error.
- Outputs final prompt + ready-to-run command; `--run` executes immediately
- `--no-interactive` or supplying all required flags bypasses prompts

### 4.1 Supported template tokens
- `{trigger}` – profile trigger word
- `{persona_action}` – resolved to either a random persona action or to the explicit `--action` value; the CLI appends `(persona_action)` to mark injections. Random actions come from `replicate_runner/config/persona_actions.yaml` (bundled file) which lists entries like `{ tokens: ["andie"], text: "twirling a transparent umbrella in the rain" }`. Filtering behavior: include actions whose `tokens` array intersects `defaults.persona_tokens` (exact match). If the intersection is empty, fall back to the global list rather than erroring.
- `{subject}` – user-specified subject
- `{mood}` – user-specified mood
- `{action}` – literal action text. If a template includes `{action}` it uses the explicit `--action` (or interactive answer) verbatim. Templates may include both `{persona_action}` and `{action}`; when only `{persona_action}` is present, providing `--action` causes it to use that text instead of a random choice. If persona injection is disabled and no action is provided, `{persona_action}` resolves to an empty string.
- `{subject_or_trigger}` – used only by the fallback template; it resolves to the profile trigger if present, otherwise to the subject text, and errors if both are missing.
- `{camera}`, `{lighting}` – optional technical cues
- Unknown tokens pass through unchanged so custom templates still render

User input is inserted verbatim; when generating `replicate` commands we escape double quotes in prompts.

## 5. UX considerations
- Rich output for listing and prompts
- Echo source scope when saving/running profiles
- Document persona action behavior and disabling flag

## 6. Implementation steps
1. Profile loader with deep merge & save helpers
2. Typer sub-apps for `explore` and `profile`
3. Wizard module supporting partial CLI inputs and interactive steps
4. README + docs update
5. Tests for merge precedence and CLI flows
