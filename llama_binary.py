from __future__ import annotations

import fnmatch
import json
import platform
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


RELEASE_API_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
PACKAGE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = PACKAGE_ROOT / "vendor" / "llama.cpp"


@dataclass(frozen=True)
class PlatformSpec:
    key: str
    executable: str
    asset_patterns: tuple[str, ...]
    required_files: tuple[str, ...]


WINDOWS_CUDA_13 = PlatformSpec(
    key="win-x64-cuda13",
    executable="llama-mtmd-cli.exe",
    asset_patterns=(
        "llama-*-bin-win-cuda-13*-x64.zip",
        "cudart-llama-bin-win-cuda-13*-x64.zip",
    ),
    required_files=(
        "llama-mtmd-cli.exe",
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


def _json_get(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-Qwen-gguf"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-Qwen-gguf"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


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


def _find_executable(install_dir: Path, spec: PlatformSpec) -> Path | None:
    for path in install_dir.rglob(spec.executable):
        if path.is_file():
            return path
    return None


def _has_required_files(install_dir: Path, spec: PlatformSpec) -> bool:
    for name in spec.required_files:
        if not any(path.is_file() for path in install_dir.rglob(name)):
            return False
    return True


def _is_complete_install(install_dir: Path, spec: PlatformSpec) -> bool:
    return _find_executable(install_dir, spec) is not None and _has_required_files(install_dir, spec)


def _existing_install(spec: PlatformSpec) -> Path | None:
    if not VENDOR_ROOT.exists():
        return None
    for install_dir in VENDOR_ROOT.glob(f"*/{spec.key}"):
        if _is_complete_install(install_dir, spec):
            return _find_executable(install_dir, spec)
    return None


def _extract_assets(assets: list[dict], install_dir: Path) -> None:
    with TemporaryDirectory(prefix="qwen-llama-download-") as temp:
        temp_dir = Path(temp)
        for asset in assets:
            archive_path = temp_dir / asset["name"]
            print(f"[Qwen GGUF] Downloading {asset['name']}...")
            _download(asset["browser_download_url"], archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(install_dir)


def ensure_llama_cli() -> Path:
    spec = _platform_spec()
    existing = _existing_install(spec)
    if existing is not None:
        return existing

    release = _json_get(RELEASE_API_URL)
    tag = release.get("tag_name") or "latest"
    install_dir = VENDOR_ROOT / tag / spec.key

    if _is_complete_install(install_dir, spec):
        executable = _find_executable(install_dir, spec)
        if executable is None:
            raise RuntimeError(f"Completed install has no {spec.executable}: {install_dir}")
        return executable

    assets = _select_assets(release, spec)
    install_dir.mkdir(parents=True, exist_ok=True)
    _extract_assets(assets, install_dir)

    executable = _find_executable(install_dir, spec)
    if executable is None:
        raise RuntimeError(
            f"Downloaded llama.cpp assets but could not find {spec.executable} in {install_dir}"
        )
    if not _has_required_files(install_dir, spec):
        missing = [
            name for name in spec.required_files
            if not any(path.is_file() for path in install_dir.rglob(name))
        ]
        raise RuntimeError(f"Downloaded llama.cpp assets are incomplete; missing: {', '.join(missing)}")

    return executable
