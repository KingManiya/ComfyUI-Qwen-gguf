from __future__ import annotations

import fnmatch
import json
import os
import platform
import shutil
import subprocess
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


LLAMA_CPP_DEFAULT_RELEASE = "latest"
LLAMA_CPP_DEFAULT_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_SOURCE_REF = "master"
PACKAGE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = PACKAGE_ROOT / "vendor" / "llama.cpp"
LINUX_SOURCE_DIR = VENDOR_ROOT / "source"
LINUX_BUILD_DIR = VENDOR_ROOT / "build" / "linux-cuda"


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


def _platform_spec() -> PlatformSpec:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows" and machine in {"amd64", "x86_64"}:
        return WINDOWS_CUDA_13
    raise RuntimeError(
        "Automatic llama.cpp binary download currently supports Windows x64 CUDA 13 only. "
        "Other platforms are intentionally isolated behind the platform mapping for future support."
    )


def _release_api_url() -> str:
    tag = os.environ.get("LLAMA_CPP_RELEASE_TAG", LLAMA_CPP_DEFAULT_RELEASE).strip()
    if not tag or tag.lower() == "latest":
        return "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    return f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}"


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


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _resolve_cli_path(path: Path) -> Path | None:
    path = path.expanduser()
    if path.is_file():
        return path if _is_executable_file(path) else None
    if not path.is_dir():
        return None

    candidates = (
        path / "llama-cli",
        path / "bin" / "llama-cli",
        path / "build" / "bin" / "llama-cli",
        path / "build" / "src" / "llama-cli",
    )
    for candidate in candidates:
        if _is_executable_file(candidate):
            return candidate

    found = _find_file(path, "llama-cli")
    if found is not None and _is_executable_file(found):
        return found
    return None


def _find_cli_paths(install_dir: Path, spec: PlatformSpec) -> LlamaCliPaths | None:
    cli = _find_file(install_dir, spec.cli_executable)
    if cli is None:
        return None
    return LlamaCliPaths(cli=cli)


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


def _env_llama_cli() -> LlamaCliPaths | None:
    configured = os.environ.get("LLAMA_CPP_PATH", "").strip()
    if not configured:
        return None
    cli = _resolve_cli_path(Path(configured))
    if cli is None:
        raise RuntimeError(
            "LLAMA_CPP_PATH is set but no executable llama-cli was found. "
            "Point it to a llama-cli file or to a llama.cpp build directory."
        )
    print(f"[LLM Text Processor] Using llama.cpp from LLAMA_CPP_PATH: {cli}")
    return LlamaCliPaths(cli=cli)


def _existing_linux_cuda_build() -> LlamaCliPaths | None:
    candidates = (
        LINUX_BUILD_DIR,
        LINUX_SOURCE_DIR,
        VENDOR_ROOT,
    )
    for candidate in candidates:
        cli = _resolve_cli_path(candidate)
        if cli is not None:
            return LlamaCliPaths(cli=cli)

    system_cli = shutil.which("llama-cli")
    if system_cli:
        return LlamaCliPaths(cli=Path(system_cli))
    return None


def _extract_assets(assets: list[dict], install_dir: Path) -> None:
    with TemporaryDirectory(prefix="llm-text-processor-llama-download-") as temp:
        temp_dir = Path(temp)
        for asset in assets:
            archive_path = temp_dir / asset["name"]
            print(f"[LLM Text Processor] Downloading {asset['name']}...")
            _download(asset["browser_download_url"], archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(install_dir)


def _run_command(command: list[str], cwd: Path | None = None) -> None:
    print(f"[LLM Text Processor] Running: {' '.join(command)}")
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Required command not found: {command[0]}. Install it and restart ComfyUI."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Command failed with exit code {exc.returncode}: {' '.join(command)}"
        ) from exc


def _clone_or_update_linux_source() -> None:
    repo_url = os.environ.get("LLAMA_CPP_REPO", LLAMA_CPP_DEFAULT_REPO).strip()
    ref = os.environ.get("LLAMA_CPP_REF", LLAMA_CPP_SOURCE_REF).strip()

    if not LINUX_SOURCE_DIR.exists():
        LINUX_SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
        _run_command(["git", "clone", "--depth", "1", repo_url, str(LINUX_SOURCE_DIR)])

    git_dir = LINUX_SOURCE_DIR / ".git"
    if not git_dir.exists():
        raise RuntimeError(
            f"Existing llama.cpp source directory is not a git checkout: {LINUX_SOURCE_DIR}"
        )

    _run_command(["git", "fetch", "--depth", "1", "origin", ref], cwd=LINUX_SOURCE_DIR)
    _run_command(["git", "checkout", "FETCH_HEAD"], cwd=LINUX_SOURCE_DIR)


def _build_linux_cuda() -> LlamaCliPaths:
    print(
        "[LLM Text Processor] No llama-cli found; building latest llama.cpp with CUDA. "
        "This can take several minutes on first run."
    )
    _clone_or_update_linux_source()
    LINUX_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    configure = [
        "cmake",
        "-S", str(LINUX_SOURCE_DIR),
        "-B", str(LINUX_BUILD_DIR),
        "-DGGML_CUDA=ON",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    build = [
        "cmake",
        "--build", str(LINUX_BUILD_DIR),
        "--config", "Release",
        "--target", "llama-cli",
        "-j", str(max((os.cpu_count() or 2) - 1, 1)),
    ]
    _run_command(configure)
    _run_command(build)

    cli = _resolve_cli_path(LINUX_BUILD_DIR)
    if cli is None:
        raise RuntimeError(f"Built llama.cpp but could not find executable llama-cli in {LINUX_BUILD_DIR}")
    return LlamaCliPaths(cli=cli)


def _ensure_windows_llama_cli_paths() -> LlamaCliPaths:
    spec = _platform_spec()
    existing = _existing_install(spec)
    if existing is not None:
        return existing

    release = _json_get(_release_api_url())
    tag = release.get("tag_name") or os.environ.get("LLAMA_CPP_RELEASE_TAG", LLAMA_CPP_DEFAULT_RELEASE)
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


def _ensure_linux_llama_cli_paths() -> LlamaCliPaths:
    existing = _existing_linux_cuda_build()
    if existing is not None:
        return existing

    return _build_linux_cuda()


def ensure_llama_cli_paths() -> LlamaCliPaths:
    configured = _env_llama_cli()
    if configured is not None:
        return configured

    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux" and machine in {"x86_64", "amd64", "aarch64", "arm64"}:
        return _ensure_linux_llama_cli_paths()
    return _ensure_windows_llama_cli_paths()
