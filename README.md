# ComfyUI LLM Text Processor

Process text and images with GGUF LLMs in ComfyUI using llama.cpp, including
Qwen3-VL, Qwen3.5, Qwen3.6, Gemma 4, and gpt-oss.

This extension adds a local LLM node for prompt writing, prompt rewriting,
translation, captioning, extraction, and other text processing tasks inside
ComfyUI. It discovers local GGUF models from `ComfyUI/models/LLM`, and it can
also accept a single image or a ComfyUI image batch for multimodal models that
use an external `mmproj`.

![Node UI](https://raw.githubusercontent.com/KingManiya/ComfyUI-LLM-text-processor/refs/heads/images/images/node.png)

## Features

- Text generation and text transformation with local GGUF models
- Optional image input for multimodal llama.cpp models that use `mmproj`,
  including multi-image batches
- Separate `RESPONSE` and `REASONING` outputs
- System prompt presets from text files
- Recursive model discovery from `ComfyUI/models/LLM`
- Automatic llama.cpp setup on supported Windows systems and Linux CUDA builds
- Download progress logs during automatic llama.cpp setup
- Advanced llama.cpp options for users who need them
- Optional `enable_processing` toggle for switching between node processing and direct passthrough

## Supported Model Families

Works with local GGUF models for Qwen, Gemma 4, gpt-oss, and other
llama.cpp-compatible families.

Common examples:

- Qwen text and vision families such as `Qwen3-VL`, `Qwen3.5`, and `Qwen3.6`
- Gemma 4 GGUF models such as `gemma-4-E2B`, `gemma-4-E4B`,
  `gemma-4-26b-a4b`, and `gemma-4-31b`
- OpenAI `gpt-oss-20b` and `gpt-oss-120b`

Notes:

- For image workflows, choose a matching `mmproj` file from the same model
  family when the GGUF release provides one.
- Gemma 4 text generation works well. Vision support depends on the specific
  GGUF and `mmproj` release, and can be less reliable on some Windows CUDA
  setups.
- In this node, `gpt-oss` is used as a text model through
  llama.cpp-compatible GGUF releases.

## Installation

### ComfyUI Manager

Open ComfyUI Manager, choose `Install Custom Nodes`, search for
`LLM Text Processor`, install it, then restart ComfyUI.

### Manual Git Clone

Open a terminal in `ComfyUI/custom_nodes` and run:

```bash
git clone https://github.com/tues8557-source/ComfyUI-LLM-text-processor.git
```

Restart ComfyUI. The node appears under:

```text
LLM Text Processor -> LLM Text Processor
```

No extra setup is needed for basic use.

### Example workflow
![Workflow](https://raw.githubusercontent.com/KingManiya/ComfyUI-LLM-text-processor/refs/heads/images/images/flow.png)

## llama.cpp

The node runs `llama-cli`. It does not download model weights.

On Windows, the node uses official llama.cpp release binaries. By default it
uses the latest llama.cpp release so newly added model architectures are not
held back by an old pinned tag. Set `LLAMA_CPP_RELEASE_TAG` if you need to pin a
specific release.

Automatic Windows binary setup is available on:

```text
Windows x64 + CUDA 13
```

On Linux, the node looks for `llama-cli` in this order:

1. `LLAMA_CPP_PATH`
2. Existing local builds under this extension's `vendor/llama.cpp` directory
3. `llama-cli` on `PATH`
4. Automatic CUDA source build with `GGML_CUDA=ON`

During automatic setup, the console shows download progress, total size when
available, and current download speed so slow connections do not look like a
freeze.

### RunPod / Linux NVIDIA Setup

RunPod users can clone this node normally into `ComfyUI/custom_nodes`:

```bash
cd /workspace/ComfyUI/custom_nodes
git clone https://github.com/tues8557-source/ComfyUI-LLM-text-processor.git
```

Then restart ComfyUI and run the node once. On Linux, if no usable `llama-cli`
is found, the extension clones llama.cpp and builds:

```bash
cmake -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ... --target llama-cli
```

The first build can take several minutes. The resulting binary is reused on
later runs.

If you already built llama.cpp yourself, point the node to it before launching
ComfyUI:

```bash
export LLAMA_CPP_PATH=/workspace/llama.cpp/build/bin/llama-cli
```

`LLAMA_CPP_PATH` may point either to the `llama-cli` executable or to a build
directory containing it.

Advanced Linux build overrides:

```bash
export LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp.git
export LLAMA_CPP_REF=master
```

Use `LLAMA_CPP_REF` to pin a known-good branch, tag, or commit. Leaving it on
`master` is recommended when you need very new architectures such as
Muse-Glimmer.

### RTX 5090 Notes

RTX 5090 is a Blackwell GPU. Use a RunPod image with a recent NVIDIA driver and
CUDA toolkit, preferably CUDA 12.8 or newer. Older CUDA images may build
llama.cpp successfully but fail to use the GPU correctly.

For large models such as `Muse-Glimmer-30B-Heretic-Q5_K_M.gguf`, start with:

```text
memory_mode = gpu_layers
n_gpu_layers = 99
ctx_size = 8192 or 16384
```

If you see `unknown model architecture: muse-glimmer`, your `llama-cli` is too
old. Remove the extension's `vendor/llama.cpp/source` and
`vendor/llama.cpp/build/linux-cuda` directories, or set `LLAMA_CPP_PATH` to a
fresh llama.cpp build.

## Model Placement

Put your GGUF files anywhere under:

```text
ComfyUI/models/LLM
```

Example:

```text
ComfyUI/models/LLM/My-Model/model-q4_k_m.gguf
ComfyUI/models/LLM/My-Model/mmproj-bf16.gguf
```

The `model` dropdown shows model `.gguf` files. The `mmproj` dropdown shows
vision projector files and `none`.

For image workflows, choose the `mmproj` file that belongs to the selected
model. You can connect either one image or a ComfyUI image batch; batch items
are sent together in the same llama.cpp request.

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

These presets are a good starting point for common models and tasks.

### Qwen

Qwen starting presets:

| Model family / use case | `reasoning` | `temperature` | `top_p` | `top_k` | `repeat_penalty` |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-VL Instruct | `off` | 0.7 | 0.8 | 20 | 1.0 |
| Qwen3-VL Thinking | `on` | 0.6 | 0.95 | 20 | 1.0 |
| Qwen3.5 / Qwen3.6 thinking, general tasks | `on` | 1.0 | 0.95 | 20 | 1.0 |
| Qwen3.5 / Qwen3.6 thinking, precise coding | `on` | 0.6 | 0.95 | 20 | 1.0 |
| Qwen3.5 / Qwen3.6 instruct, general tasks | `off` | 0.7 | 0.8 | 20 | 1.0 |
| Qwen3.5 / Qwen3.6 instruct, reasoning tasks | `off` | 1.0 | 1.0 | 40 | 1.0 |

Reference: [Qwen3 docs](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/quickstart.md)

### Gemma 4

Gemma 4 starting preset:

| Model family / use case | `reasoning` | `temperature` | `top_p` | `top_k` | `repeat_penalty` |
| --- | --- | ---: | ---: | ---: | ---: |
| Gemma 4 `it` models, general tasks | `off` | 1.0 | 0.95 | 64 | 1.0 |
| Gemma 4 `it` models, reasoning tasks | `on` | 1.0 | 0.95 | 64 | 1.0 |
| Gemma 4 multimodal tasks | `off` | 1.0 | 0.95 | 64 | 1.0 |

Common Gemma 4 variants:

- `gemma-4-E2B`
- `gemma-4-E4B`
- `gemma-4-26b-a4b`
- `gemma-4-31b`

Gemma 4 supports configurable thinking modes across the family. For simple
prompt writing, translation, captioning, and extraction, start with
`reasoning=off`. For harder reasoning or coding tasks, try `reasoning=on`.

Reference:

- [Gemma 4 model overview](https://ai.google.dev/gemma/docs/core)
- [Gemma 4 E4B model card](https://huggingface.co/google/gemma-4-E4B)

### gpt-oss

gpt-oss starting preset:

| Model family / use case | `reasoning` | `temperature` | `top_p` | `top_k` | `repeat_penalty` |
| --- | --- | ---: | ---: | ---: | ---: |
| `gpt-oss-20b`, direct answers and lower-latency local tasks | `off` | 1.0 | 1.0 | 20 | 1.0 |
| `gpt-oss-20b`, reasoning-heavy tasks | `on` | 1.0 | 1.0 | 20 | 1.0 |
| `gpt-oss-120b`, general purpose and stronger reasoning | `on` | 1.0 | 1.0 | 20 | 1.0 |

Use the preset as shown. Leave the other values at default unless you already
know you want different sampling.

Common gpt-oss variants:

- `gpt-oss-20b`
- `gpt-oss-120b`

OpenAI describes `gpt-oss-20b` as the lower-latency option for local or
specialized use cases, and `gpt-oss-120b` as the larger option for production,
general purpose, and higher-reasoning workloads.

OpenAI also documents configurable reasoning effort for `gpt-oss`. This node
does not expose the native low / medium / high control directly, so the presets
above use the simpler `reasoning` toggle available here: start with `off` for
direct answers, and try `on` for harder reasoning tasks.

Reference: [gpt-oss docs](https://github.com/openai/gpt-oss)

## Node Inputs

| Input | Description |
| --- | --- |
| `model` | GGUF model file from `ComfyUI/models/LLM`. |
| `mmproj` | Vision projector GGUF. Required when using image input, whether it is one image or a batch. |
| `system_prompt` | Prompt preset from `models/LLM/prompts`, or `none`. |
| `prompt` | User prompt sent to the selected model. |
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
| `image` | Optional image input. Accepts a single image or a ComfyUI batch, and sends the batch together to llama.cpp. |
| `enable_processing` | When enabled, the node runs normally. When disabled, the node forwards `prompt` directly to `RESPONSE` and skips all model checks and llama.cpp execution. |
| `extra_args` | Optional advanced llama.cpp parameters. Leave empty for normal use. |

## Node Outputs

| Output | Description |
| --- | --- |
| `RESPONSE` | Final model response with reasoning blocks removed, or the input `prompt` when `enable_processing` is disabled. |
| `REASONING` | Extracted reasoning when present in model output. Empty when `enable_processing` is disabled. |
| `PERF` | Prompt and generation speed reported by llama.cpp. Empty when `enable_processing` is disabled. |

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

Lower `ctx_size` first. If that is not enough, use a smaller model, a smaller
quant, or adjust memory placement.

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
- [Qwen](https://github.com/QwenLM/Qwen3)
- [Gemma](https://ai.google.dev/gemma/docs)
- [gpt-oss](https://github.com/openai/gpt-oss)
