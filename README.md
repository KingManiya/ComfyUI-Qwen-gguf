# ComfyUI Qwen GGUF

Run Qwen3-VL, Qwen3.5, and Qwen3.6 GGUF models in ComfyUI with llama.cpp.

This extension adds a simple Qwen node for text and image workflows. It uses
llama.cpp behind the scenes, discovers local GGUF models from
`ComfyUI/models/LLM`, and keeps the ComfyUI workflow focused on the settings
you usually need during generation.

## Features

- Qwen text generation with GGUF models
- Optional image input for supported vision models
- Separate `RESPONSE` and `REASONING` outputs
- System prompt presets from text files
- Recursive model discovery from `ComfyUI/models/LLM`
- Automatic llama.cpp setup on supported Windows systems
- Advanced llama.cpp options for users who need them

## Supported Models

The node is intended for GGUF versions of these Qwen model families:

- Qwen3-VL: `Qwen3-VL-2B`, `Qwen3-VL-4B`, `Qwen3-VL-8B`,
  `Qwen3-VL-30B-A3B`, `Qwen3-VL-32B`, `Qwen3-VL-235B-A22B`
- Qwen3.5: `Qwen3.5-0.8B`, `Qwen3.5-2B`, `Qwen3.5-4B`, `Qwen3.5-9B`,
  `Qwen3.5-27B`, `Qwen3.5-35B-A3B`, `Qwen3.5-122B-A10B`
- Qwen3.6: `Qwen3.6-35B-A3B`

For image workflows, use a matching `mmproj` file from the same model family.

## Installation

### ComfyUI Manager

Open ComfyUI Manager, choose `Install Custom Nodes`, search for
`ComfyUI-Qwen-gguf` or `Qwen GGUF`, install it, then restart ComfyUI.

### Manual Git Clone

Open a terminal in `ComfyUI/custom_nodes` and run:

```bash
git clone https://github.com/KingManiya/ComfyUI-Qwen-gguf.git
```

Restart ComfyUI. The node appears under:

```text
Qwen GGUF -> Qwen GGUF (llama.cpp)
```

No Python package install is required for the basic node. ComfyUI already ships
with the runtime pieces used here. Pillow and NumPy are used when an image input
is connected.

## llama.cpp

The node uses official llama.cpp release binaries. On supported systems, the
required llama.cpp files are prepared automatically the first time you run the
node.

Current automatic setup target:

```text
Windows x64 + CUDA 13
```

Other platforms are not set up automatically yet.

The extension downloads llama.cpp only. It does not download Qwen model weights.

## Model Placement

Put your Qwen GGUF files anywhere under:

```text
ComfyUI/models/LLM
```

Example:

```text
ComfyUI/models/LLM/My-Qwen-Model/model-q4_k_m.gguf
ComfyUI/models/LLM/My-Qwen-Model/mmproj-BF16.gguf
```

The `model` dropdown shows model `.gguf` files. The `mmproj` dropdown shows
vision projector files and `none`.

For image workflows, choose the `mmproj` file that belongs to the selected
model. The node uses only the first image from a ComfyUI image batch.

## System Prompt Presets

Create text files in:

```text
ComfyUI/models/LLM/prompts
```

Example:

```text
ComfyUI/models/LLM/prompts/captioner.txt
```

Each top-level `.txt` file appears in the `system_prompt` dropdown. Choose
`none` to run without a system prompt.

## Recommended Settings

These values are based on official Qwen recommendations and mapped to the
parameters available in this node.

| Model family / use case | `reasoning` | `temperature` | `top_p` | `top_k` | `repeat_penalty` |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-VL Instruct | `off` | 0.7 | 0.8 | 20 | 1.0 |
| Qwen3-VL Thinking | `on` | 0.6 | 0.95 | 20 | 1.0 |
| Qwen3.5 / Qwen3.6 thinking, general tasks | `on` | 1.0 | 0.95 | 20 | 1.0 |
| Qwen3.5 / Qwen3.6 thinking, precise coding | `on` | 0.6 | 0.95 | 20 | 1.0 |
| Qwen3.5 / Qwen3.6 instruct, general tasks | `off` | 0.7 | 0.8 | 20 | 1.0 |
| Qwen3.5 / Qwen3.6 instruct, reasoning tasks | `off` | 1.0 | 1.0 | 40 | 1.0 |

Some Qwen recommendations also mention `min_p` and `presence_penalty`. They are
not regular node inputs yet. Leave `extra_args` empty unless you already know
which additional llama.cpp options your model needs.

## Node Inputs

| Input | Description |
| --- | --- |
| `model` | GGUF model file from `ComfyUI/models/LLM`. |
| `mmproj` | Vision projector GGUF. Required when using image input. |
| `system_prompt` | Prompt preset from `models/LLM/prompts`, or `none`. |
| `prompt` | User prompt sent to the model. |
| `max_tokens` | Maximum generated tokens. |
| `temperature` | Sampling temperature. Lower values are more deterministic. |
| `top_p` | Nucleus sampling threshold. |
| `top_k` | Top-K sampling cutoff. |
| `repeat_penalty` | Penalty for repeated tokens. |
| `ctx_size` | Context window size. Larger values use more memory. |
| `memory_mode` | Advanced memory placement mode: `auto`, `gpu_layers`, `cpu_moe_layers`, or `gpu_and_cpu_moe_layers`. |
| `n_gpu_layers` | Used only in `gpu_layers` and `gpu_and_cpu_moe_layers` modes. |
| `n_cpu_moe_layers` | Used only in `cpu_moe_layers` and `gpu_and_cpu_moe_layers` modes. |
| `seed` | Random seed. Use `-1` for a random seed. |
| `timeout_seconds` | Maximum runtime before generation is stopped. |
| `reasoning` | Reasoning output mode: `auto`, `on`, or `off`. |
| `image` | Optional ComfyUI image input. Uses the first image in a batch. |
| `extra_args` | Optional advanced llama.cpp parameters. Leave empty for normal use. |

Every input includes an in-node tooltip.

## Node Outputs

| Output | Description |
| --- | --- |
| `RESPONSE` | Final model response with reasoning blocks removed. |
| `REASONING` | Extracted reasoning when present in model output. |
| `PERF` | Prompt and generation speed reported by llama.cpp. |

## Troubleshooting

### No models appear

Place at least one `.gguf` model under:

```text
ComfyUI/models/LLM
```

Then refresh or restart ComfyUI.

### Image input fails

Make sure `mmproj` is not set to `none` and that the projector belongs to the
same model family as the selected GGUF model.

### llama.cpp setup fails

Check your internet connection and GitHub access, then run the node again.

### Unsupported platform

Automatic llama.cpp setup currently supports Windows x64 CUDA 13 only.

### Out of memory

Reduce `ctx_size` first. The context window reserves memory for the model's
working context, so a large value can use a lot of memory even when the current
prompt is short.

If memory is still tight, use a smaller GGUF model, a smaller quant, or adjust
the advanced memory placement settings.

### Generation takes too long

Try lowering `max_tokens`, reducing `ctx_size`, using a smaller GGUF model, or
increasing `timeout_seconds`.

Use `extra_args` only if you already know which llama.cpp options your setup
needs.

### Response is empty or cut off

Increase `max_tokens`. This is especially important when `reasoning` is set to
`on` or `auto`, because the model can spend part of the token budget on
reasoning before it reaches the final answer.

## Credits

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Qwen](https://github.com/QwenLM)
