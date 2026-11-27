import pytest

from app.domain import Status
from app.errors import ConversionFailed, Invalid, NotFound, PayloadTooLarge, UnsupportedFormat
from tests.fakes import FakeEngine, make_service


def test_successful_conversion_is_stored_with_measured_duration() -> None:
    service = make_service(FakeEngine())
    conversion = service.convert("a.pdf", b"%PDF", ocr=True)
    assert conversion.status == Status.SUCCESS
    assert conversion.duration_ms == 250
    assert service.get(conversion.id) == conversion


def test_ocr_flag_is_forwarded_to_the_engine() -> None:
    engine = FakeEngine()
    make_service(engine).convert("scan.png", b"x", ocr=True)
    assert engine.calls == [("scan.png", True)]


def test_engine_failure_is_kept_in_history_and_raised() -> None:
    service = make_service(FakeEngine(fail_with="broken pdf"))
    with pytest.raises(ConversionFailed, match="broken pdf"):
        service.convert("bad.pdf", b"x", ocr=False)
    [summary] = service.history()
    assert summary.status == Status.FAILED
    assert summary.error == "broken pdf"


def test_empty_and_oversized_files_never_reach_the_engine() -> None:
    engine = FakeEngine()
    service = make_service(engine, max_upload_bytes=4)
    with pytest.raises(Invalid):
        service.convert("a.pdf", b"", ocr=False)
    with pytest.raises(PayloadTooLarge):
        service.convert("a.pdf", b"12345", ocr=False)
    with pytest.raises(UnsupportedFormat):
        service.convert("a.exe", b"1", ocr=False)
    assert engine.calls == []


def test_history_is_newest_first_with_stats_summary() -> None:
    service = make_service(FakeEngine())
    service.convert("first.pdf", b"x", ocr=False)
    service.convert("second.docx", b"x", ocr=False)
    history = service.history()
    assert [h.filename for h in history] == ["second.docx", "first.pdf"]
    assert history[0].pages == 3
    assert history[0].tables == 1


def test_unknown_id_is_not_found() -> None:
    with pytest.raises(NotFound):
        make_service(FakeEngine()).get("nope")
