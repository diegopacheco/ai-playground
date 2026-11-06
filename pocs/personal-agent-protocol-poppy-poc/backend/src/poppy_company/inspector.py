import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from poppy_company.json_types import JsonObject, JsonValue

SUPPORTED_MAJOR_MINOR = ("0.4",)
LOOPBACK = ("localhost", "127.0.0.1", "::1")


@dataclass(frozen=True)
class Check:
    name: str
    level: str
    detail: str

    def to_json(self) -> JsonObject:
        return {"name": self.name, "level": self.level, "detail": self.detail}


def passed(name: str, detail: str) -> Check:
    return Check(name, "pass", detail)


def failed(name: str, detail: str) -> Check:
    return Check(name, "fail", detail)


def warned(name: str, detail: str) -> Check:
    return Check(name, "warn", detail)


def urls_in(value: JsonValue, path: str) -> list[tuple[str, str]]:
    if isinstance(value, str) and "://" in value:
        return [(path, value)]
    if isinstance(value, dict):
        return [found for key, item in value.items() for found in urls_in(item, f"{path}.{key}")]
    if isinstance(value, list):
        return [found for index, item in enumerate(value) for found in urls_in(item, f"{path}[{index}]")]
    return []


def check_version(document: JsonObject) -> Check:
    version = document.get("protocol_version")
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+", version):
        return failed("protocol_version", "required, as major.minor")
    if version not in SUPPORTED_MAJOR_MINOR:
        return warned("protocol_version", f"{version} is not draft 0.4, which this inspector checks")
    return passed("protocol_version", version)


def check_organization(document: JsonObject, host: str) -> list[Check]:
    organization = document.get("organization")
    if not isinstance(organization, dict):
        return [failed("organization", "required object with name and domain")]
    name, domain = organization.get("name"), organization.get("domain")
    checks = [passed("organization.name", str(name)) if isinstance(name, str) and name else failed("organization.name", "required")]
    if not isinstance(domain, str):
        checks.append(failed("organization.domain", "required"))
    elif domain.lower() != host.lower():
        checks.append(failed("organization.domain", f"{domain} is not the host that served the document ({host})"))
    else:
        checks.append(passed("organization.domain", f"{domain} matches the serving host"))
    return checks


def check_channels(document: JsonObject) -> list[Check]:
    present = [name for name in ("agent", "apis", "web") if name in document]
    checks = [passed("channels", ", ".join(present)) if present else failed("channels", "at least one of agent, apis or web is required")]
    agent = document.get("agent")
    if isinstance(agent, dict):
        protocols = agent.get("protocols")
        valid = isinstance(protocols, list) and protocols and all(isinstance(p, dict) and isinstance(p.get("type"), str) and isinstance(p.get("endpoint"), str) for p in protocols)
        checks.append(passed("agent.protocols", "each entry has type and endpoint") if valid else failed("agent.protocols", "a non empty list of {type, endpoint}"))
    apis = document.get("apis")
    if "apis" in document:
        valid = isinstance(apis, list) and apis and all(isinstance(a, dict) and a.get("type") in ("openapi", "mcp") and isinstance(a.get("url"), str) for a in apis)
        checks.append(passed("apis", "each entry has type openapi or mcp and url") if valid else failed("apis", "a non empty list of {type: openapi|mcp, url}"))
    return checks


def check_optional_endpoints(document: JsonObject) -> list[Check]:
    checks: list[Check] = []
    for name, key in (("feedback", "endpoint"), ("excluded_tasks", "url")):
        if name in document:
            entry = document.get(name)
            ok = isinstance(entry, dict) and isinstance(entry.get(key), str)
            checks.append(passed(f"{name}.{key}", "present") if ok else failed(f"{name}.{key}", f"{name} needs a {key} URL"))
    auth = document.get("auth")
    if isinstance(auth, dict) and auth.get("disabled") is not True and "issuer" in auth:
        checks.append(warned("auth.issuer", "fetch the issuer's OAuth metadata and confirm protected_resources lists the domain"))
    return checks


def check_https(document: JsonObject) -> list[Check]:
    checks: list[Check] = []
    for path, url in urls_in(document, "$"):
        parts = urlsplit(url)
        if parts.scheme == "https":
            continue
        if parts.hostname in LOOPBACK:
            checks.append(warned("https", f"{path} uses {parts.scheme} on loopback, allowed only for local testing"))
        else:
            checks.append(failed("https", f"{path} must be an HTTPS URL"))
    return checks or [passed("https", "every URL is HTTPS")]


def inspect(document: JsonObject, host: str) -> list[Check]:
    return [check_version(document), *check_organization(document, host), *check_channels(document), *check_optional_endpoints(document), *check_https(document)]
