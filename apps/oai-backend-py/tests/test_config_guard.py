"""Guard against env/config drift outside config.py."""

from __future__ import annotations

from pathlib import Path


def test_no_env_access_outside_config() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"
    config_path = src_root / "config.py"

    patterns = ("os.getenv", "os.environ", "BaseSettings")
    violations: list[str] = []

    for path in src_root.rglob("*.py"):
        if path == config_path:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if pattern in text:
                violations.append(f"{path}: found '{pattern}'")

    assert not violations, "Config/env access must be centralized in config.py:\n" + "\n".join(violations)
