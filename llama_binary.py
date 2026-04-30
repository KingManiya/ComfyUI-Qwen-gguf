from __future__ import annotations

import fnmatch
import json
import os
import platform
import stat
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


LLAMA_CPP_RELEASE_TAG = "b8840"
RELEASE_API_URL = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{LLAMA_CPP_RELEASE_TAG}"
PACKAGE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = PACKAGE_ROOT / "vendor" / "llama.cpp"
LLAMA_BACKEND_ENV = "LLM_TEXT_PROCESSOR_LLAMA_BACKEND"


@dataclass(frozen=True)
class PlatformSpec:
    key: str
    cli_executable: str
    asset_patterns: tuple[str, ...]
    required_files: tuple[str, ...]


@dataclass(frozen=True)
class LlamaCliPaths:
    cli: Path


WINDOWS_CUDA_13 = PlatformSpec(
    key="win-x64-cuda13",
    cli_executable="llama-cli.exe",
    asset_patterns=(
        "llama-*-bin-win-cuda-13*-x64.zip",
        "cudart-llama-bin-win-cuda-13*-x64.zip",
    ),
    required_files=(
        "llama-cli.exe",
        "ggml-cuda.dll",
        "cudart64_13.dll",
    ),
)

UBUNTU_X64_CPU = PlatformSpec(
    key="ubuntu-x64-cpu",
    cli_executable="llama-cli",
    asset_patterns=("llama-*-bin-ubuntu-x64.tar.gz",),
    required_files=(
        "llama-cli",
        "libllama.so",
        "libllama-common.so",
        "libggml.so",
        "libggml-base.so",
    ),
)

UBUNTU_X64_VULKAN = PlatformSpec(
    key="ubuntu-x64-vulkan",
    cli_executable="llama-cli",
    asset_patterns=("llama-*-bin-ubuntu-vulkan-x64.tar.gz",),
    required_files=UBUNTU_X64_CPU.required_files + ("libggml-vulkan.so",),
)

UBUNTU_X64_ROCM = PlatformSpec(
    key="ubuntu-x64-rocm",
    cli_executable="llama-cli",
    asset_patterns=("llama-*-bin-ubuntu-rocm-*-x64.tar.gz",),
    required_files=UBUNTU_X64_CPU.required_files + ("libggml-hip.so",),
)

UBUNTU_X64_OPENVINO = PlatformSpec(
    key="ubuntu-x64-openvino",
    cli_executable="llama-cli",
    asset_patterns=("llama-*-bin-ubuntu-openvino-*-x64.tar.gz",),
    required_files=UBUNTU_X64_CPU.required_files + ("libggml-openvino.so",),
)

UBUNTU_ARM64_CPU = PlatformSpec(
    key="ubuntu-arm64-cpu",
    cli_executable="llama-cli",
    asset_patterns=("llama-*-bin-ubuntu-arm64.tar.gz",),
    required_files=UBUNTU_X64_CPU.required_files,
)

LINUX_X64_SPECS = {
    "cpu": UBUNTU_X64_CPU,
    "vulkan": UBUNTU_X64_VULKAN,
    "rocm": UBUNTU_X64_ROCM,
    "openvino": UBUNTU_X64_OPENVINO,
}


def _platform_spec() -> PlatformSpec:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows" and machine in {"amd64", "x86_64"}:
        return WINDOWS_CUDA_13
    if system == "linux" and machine in {"amd64", "x86_64"}:
        backend = os.environ.get(LLAMA_BACKEND_ENV, "cpu").strip().lower()
        try:
            return LINUX_X64_SPECS[backend]
        except KeyError:
            supported = ", ".join(sorted(LINUX_X64_SPECS))
            raise RuntimeError(
                f"Unsupported {LLAMA_BACKEND_ENV}={backend!r}. "
                f"Supported Ubuntu x64 backends: {supported}."
            ) from None
    if system == "linux" and machine in {"aarch64", "arm64"}:
        backend = os.environ.get(LLAMA_BACKEND_ENV, "cpu").strip().lower()
        if backend == "cpu":
            return UBUNTU_ARM64_CPU
        raise RuntimeError(
            f"Unsupported {LLAMA_BACKEND_ENV}={backend!r}. "
            "Supported Ubuntu arm64 backends: cpu."
        )
    raise RuntimeError(
        "Automatic llama.cpp binary download currently supports Windows x64 CUDA 13 "
        "and Ubuntu Linux x64/arm64. Other platforms require manual llama.cpp setup."
    )


def _json_get(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-LLM-text-processor"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _format_size(num_bytes: float) -> str:
    units = ("B", "KB", "MB", "GB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-LLM-text-processor"})
    with urllib.request.urlopen(request, timeout=120) as response:
        total_size = response.headers.get("Content-Length")
        total_size = int(total_size) if total_size is not None else None
        downloaded = 0
        chunk_size = 1024 * 256
        started_at = time.monotonic()
        last_reported_at = started_at

        with destination.open("wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)

                now = time.monotonic()
                if now - last_reported_at < 1.0:
                    continue

                elapsed = max(now - started_at, 0.001)
                speed = downloaded / elapsed
                if total_size:
                    percent = (downloaded / total_size) * 100
                    print(
                        "[LLM Text Processor] "
                        f"Downloaded {_format_size(downloaded)} / {_format_size(total_size)} "
                        f"({percent:.1f}%) at {_format_size(speed)}/s"
                    )
                else:
                    print(
                        "[LLM Text Processor] "
                        f"Downloaded {_format_size(downloaded)} at {_format_size(speed)}/s"
                    )
                last_reported_at = now

        elapsed = max(time.monotonic() - started_at, 0.001)
        speed = downloaded / elapsed
        if total_size:
            print(
                "[LLM Text Processor] "
                f"Finished download: {_format_size(downloaded)} / {_format_size(total_size)} "
                f"(100.0%) at {_format_size(speed)}/s"
            )
        else:
            print(
                "[LLM Text Processor] "
                f"Finished download: {_format_size(downloaded)} at {_format_size(speed)}/s"
            )


def _select_assets(release: dict, spec: PlatformSpec) -> list[dict]:
    assets = release.get("assets", [])
    selected = []
    used_names = set()

    # Match explicit release asset names so a future platform can add patterns
    # without changing the download/extract pipeline.
    for pattern in spec.asset_patterns:
        matches = [
            asset for asset in assets
            if fnmatch.fnmatch(asset.get("name", "").lower(), pattern.lower())
        ]
        if not matches:
            raise RuntimeError(f"Could not find llama.cpp release asset matching: {pattern}")
        asset = sorted(matches, key=lambda item: item.get("name", ""))[0]
        if asset["name"] not in used_names:
            selected.append(asset)
            used_names.add(asset["name"])
    return selected


def _find_file(install_dir: Path, name: str) -> Path | None:
    for path in install_dir.rglob(name):
        if path.is_file():
            return path
    return None


def _find_cli_paths(install_dir: Path, spec: PlatformSpec) -> LlamaCliPaths | None:
    cli = _find_file(install_dir, spec.cli_executable)
    if cli is None:
        return None
    _ensure_executable(cli)
    return LlamaCliPaths(cli=cli)


def _ensure_executable(path: Path) -> None:
    if os.name == "nt":
        return
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _has_required_files(install_dir: Path, spec: PlatformSpec) -> bool:
    for name in spec.required_files:
        if not any(path.is_file() for path in install_dir.rglob(name)):
            return False
    return True


def _is_complete_install(install_dir: Path, spec: PlatformSpec) -> bool:
    return _find_cli_paths(install_dir, spec) is not None and _has_required_files(install_dir, spec)


def _existing_install(spec: PlatformSpec) -> LlamaCliPaths | None:
    if not VENDOR_ROOT.exists():
        return None
    for install_dir in VENDOR_ROOT.glob(f"*/{spec.key}"):
        if _is_complete_install(install_dir, spec):
            return _find_cli_paths(install_dir, spec)
    return None


def _safe_extract_tar(archive: tarfile.TarFile, install_dir: Path) -> None:
    destination = install_dir.resolve()
    for member in archive.getmembers():
        member_path = (install_dir / member.name).resolve()
        if member_path != destination and destination not in member_path.parents:
            raise RuntimeError(f"Refusing to extract unsafe archive member: {member.name}")
    archive.extractall(install_dir)


def _extract_archive(archive_path: Path, install_dir: Path) -> None:
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(install_dir)
        return
    if name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as archive:
            _safe_extract_tar(archive, install_dir)
        return
    raise RuntimeError(f"Unsupported llama.cpp archive format: {archive_path.name}")


def _extract_assets(assets: list[dict], install_dir: Path) -> None:
    with TemporaryDirectory(prefix="llm-text-processor-llama-download-") as temp:
        temp_dir = Path(temp)
        for asset in assets:
            archive_path = temp_dir / asset["name"]
            print(f"[LLM Text Processor] Downloading {asset['name']}...")
            _download(asset["browser_download_url"], archive_path)
            _extract_archive(archive_path, install_dir)


def ensure_llama_cli_paths() -> LlamaCliPaths:
    spec = _platform_spec()
    existing = _existing_install(spec)
    if existing is not None:
        return existing

    release = _json_get(RELEASE_API_URL)
    tag = release.get("tag_name") or LLAMA_CPP_RELEASE_TAG
    install_dir = VENDOR_ROOT / tag / spec.key

    if _is_complete_install(install_dir, spec):
        paths = _find_cli_paths(install_dir, spec)
        if paths is None:
            raise RuntimeError(f"Completed install has incomplete CLI executables: {install_dir}")
        return paths

    assets = _select_assets(release, spec)
    install_dir.mkdir(parents=True, exist_ok=True)
    _extract_assets(assets, install_dir)

    paths = _find_cli_paths(install_dir, spec)
    if paths is None:
        raise RuntimeError(
            f"Downloaded llama.cpp assets but could not find CLI executables in {install_dir}"
        )
    if not _has_required_files(install_dir, spec):
        missing = [
            name for name in spec.required_files
            if not any(path.is_file() for path in install_dir.rglob(name))
        ]
        raise RuntimeError(f"Downloaded llama.cpp assets are incomplete; missing: {', '.join(missing)}")

    return paths
