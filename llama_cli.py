from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path


THINK_BLOCK_RE = re.compile(r"<think[^>]*>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


def tensor_to_temp_png(image) -> Path:
    import numpy as np
    from PIL import Image

    # ComfyUI IMAGE is normally B,H,W,C. The node uses the first batch item in v1.
    tensor = image
    if hasattr(tensor, "dim") and tensor.dim() == 4:
        tensor = tensor[0]
    array = (tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_image = Image.fromarray(array)
    fd, path = tempfile.mkstemp(prefix="qwen-gguf-", suffix=".png")
    os.close(fd)
    pil_image.save(path, format="PNG")
    return Path(path)


def _write_prompt_file(system_prompt: str, prompt: str, enable_thinking: bool) -> Path:
    # Qwen chat templates understand these control tokens across llama.cpp builds
    # more reliably than version-specific template kwargs.
    think_prefix = "/think" if enable_thinking else "/no_think"
    chunks = []
    if system_prompt:
        chunks.append(system_prompt.strip())
    chunks.append(think_prefix)
    chunks.append(prompt.strip())

    fd, path = tempfile.mkstemp(prefix="qwen-gguf-prompt-", suffix=".txt")
    os.close(fd)
    prompt_path = Path(path)
    prompt_path.write_text("\n\n".join(chunks), encoding="utf-8", newline="\n")
    return prompt_path


def _split_extra_args(extra_args: str) -> list[str]:
    if not extra_args or not extra_args.strip():
        return []
    parts = shlex.split(extra_args, posix=(os.name != "nt"))
    return [part.strip("\"'") for part in parts]


def build_command(
    cli_path: Path,
    model_path: Path,
    mmproj_path: Path | None,
    image_path: Path | None,
    prompt_path: Path,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    ctx_size: int,
    n_gpu_layers: int,
    seed: int,
    extra_args: str = "",
) -> list[str]:
    command = [
        str(cli_path),
        "-m", str(model_path),
        "-f", str(prompt_path),
        "-n", str(max_tokens),
        "--temp", str(temperature),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
        "--repeat-penalty", str(repeat_penalty),
        "-c", str(ctx_size),
        "-ngl", str(n_gpu_layers),
        "--seed", str(seed),
    ]

    if mmproj_path is not None:
        command.extend(["--mmproj", str(mmproj_path)])
    if image_path is not None:
        command.extend(["--image", str(image_path)])

    command.extend(_split_extra_args(extra_args))
    return command


def run_llama_cli(
    cli_path: Path,
    model_path: Path,
    mmproj_path: Path | None,
    image,
    system_prompt: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    ctx_size: int,
    n_gpu_layers: int,
    seed: int,
    enable_thinking: bool,
    timeout_seconds: int,
    extra_args: str = "",
) -> str:
    image_path = None
    prompt_path = None
    try:
        if image is not None:
            if mmproj_path is None:
                raise ValueError("Image input requires a selected mmproj GGUF file.")
            image_path = tensor_to_temp_png(image)

        prompt_path = _write_prompt_file(system_prompt, prompt, enable_thinking)
        command = build_command(
            cli_path=cli_path,
            model_path=model_path,
            mmproj_path=mmproj_path,
            image_path=image_path,
            prompt_path=prompt_path,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            ctx_size=ctx_size,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            extra_args=extra_args,
        )

        print(f"[Qwen GGUF] Running llama.cpp: {model_path.name}")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            shell=False,
        )
    finally:
        # Temp prompt/image files can be large in workflows that run many times,
        # so cleanup happens even when llama.cpp exits with an error.
        for path in (image_path, prompt_path):
            if path is not None and path.exists():
                path.unlink()

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"llama.cpp inference failed with exit code {result.returncode}:\n{stderr[-2000:]}"
        )
    return result.stdout


def extract_thinking(text: str, enable_thinking: bool) -> tuple[str, str]:
    text = str(text or "")
    thinking = ""

    # llama.cpp returns raw stdout; Qwen reasoning may appear as a full block,
    # a closing-only block, or an unfinished block if generation is truncated.
    match = THINK_BLOCK_RE.search(text)
    if match:
        thinking = re.sub(r"</?think[^>]*>", "", match.group(0), flags=re.IGNORECASE).strip()
        text = THINK_BLOCK_RE.sub("", text).strip()
    elif "</think>" in text:
        before, after = text.split("</think>", 1)
        thinking = before.strip()
        text = after.strip()
    elif "<think>" in text:
        before, after = text.split("<think>", 1)
        thinking = after.strip()
        text = before.strip()

    for token in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        text = text.replace(token, "")

    return text.strip(), thinking if enable_thinking else ""
