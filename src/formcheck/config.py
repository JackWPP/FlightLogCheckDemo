from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT / "assets"
CANONICAL_DIR = ASSETS_DIR / "canonical"
RAW_DIR = ASSETS_DIR / "raw"
OUT_DIR = ROOT / "out"
OUTPUTS_DIR = ROOT / "outputs"
FIELDS_PATH = ROOT / "fields.yaml"


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or (ROOT / ".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str | None
    base_url: str
    model: str


def provider_config(provider: str) -> ProviderConfig:
    load_dotenv()
    normalized = provider.lower()
    if normalized == "siliconflow":
        return ProviderConfig(
            name="siliconflow",
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
            model=os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen3.6-27B"),
        )
    if normalized == "aliyun":
        return ProviderConfig(
            name="aliyun",
            api_key=os.getenv("ALIYUN_API_KEY"),
            base_url=os.getenv(
                "ALIYUN_BASE_URL",
                "https://llm-qcfwflkh57p80u70.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            ),
            model=os.getenv("ALIYUN_MODEL", "qwen3.5-ocr"),
        )
    return ProviderConfig(name="mock", api_key=None, base_url="", model="mock")
