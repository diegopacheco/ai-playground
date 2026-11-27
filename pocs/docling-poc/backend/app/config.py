from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    service_name: str
    max_upload_bytes: int

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Settings":
        return cls(
            service_name=env.get("SERVICE_NAME", "docling-poc"),
            max_upload_bytes=int(env.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024,
        )
