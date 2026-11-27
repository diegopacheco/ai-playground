import pytest

from app.domain import DocFormat, detect_format
from app.errors import UnsupportedFormat


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("report.PDF", DocFormat.PDF),
        ("slides.pptx", DocFormat.PPTX),
        ("page.htm", DocFormat.HTML),
        ("scan.jpeg", DocFormat.IMAGE),
    ],
)
def test_detect_format_is_case_insensitive_and_groups_aliases(filename: str, expected: DocFormat) -> None:
    assert detect_format(filename) == expected


@pytest.mark.parametrize("filename", ["malware.exe", "README", "archive.tar.gz"])
def test_unknown_types_are_rejected_before_reaching_docling(filename: str) -> None:
    with pytest.raises(UnsupportedFormat):
        detect_format(filename)
