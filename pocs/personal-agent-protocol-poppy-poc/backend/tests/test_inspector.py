from poppy_company.discovery import poppy_document
from poppy_company.inspector import inspect
from poppy_company.json_types import JsonObject


def levels(document: JsonObject, host: str) -> dict[str, str]:
    return {check.name: check.level for check in inspect(document, host)}


def test_https_document_on_its_own_host_passes_every_check() -> None:
    document = poppy_document("https://willow.example")
    assert set(levels(document, "willow.example").values()) == {"pass"}


def test_document_copied_to_another_domain_fails() -> None:
    assert levels(poppy_document("https://willow.example"), "copycat.example")["organization.domain"] == "fail"


def test_document_without_any_channel_fails() -> None:
    assert levels({"protocol_version": "0.4", "organization": {"name": "X", "domain": "x.example"}}, "x.example")["channels"] == "fail"


def test_plain_http_fails_except_on_loopback_where_it_only_warns() -> None:
    assert levels(poppy_document("http://willow.example"), "willow.example")["https"] == "fail"
    assert levels(poppy_document("http://localhost:8741"), "localhost:8741")["https"] == "warn"


def test_bad_version_fails() -> None:
    assert levels({"protocol_version": "v1"}, "x.example")["protocol_version"] == "fail"
