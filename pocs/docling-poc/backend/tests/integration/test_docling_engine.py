from pathlib import Path

from app.engine import DoclingEngine

HTML = b"""<html><body>
<h1>Quarterly Report</h1>
<p>Revenue grew in every region.</p>
<h2>Regions</h2>
<table><tr><th>Region</th><th>Revenue</th></tr><tr><td>North</td><td>120</td></tr><tr><td>South</td><td>95</td></tr></table>
</body></html>"""

SAMPLE_PDF = Path(__file__).parents[3] / "samples" / "table-page.pdf"


def test_html_tables_and_headings_survive_conversion() -> None:
    doc = DoclingEngine().convert("report.html", HTML, ocr=False)
    assert "# Quarterly Report" in doc.markdown
    [table] = doc.tables
    assert table.headers == ("Region", "Revenue")
    assert table.rows == (("North", "120"), ("South", "95"))
    assert any("Regions" in c.headings for c in doc.chunks)


def test_pdf_layout_model_finds_the_table_on_the_page() -> None:
    doc = DoclingEngine().convert(SAMPLE_PDF.name, SAMPLE_PDF.read_bytes(), ocr=False)
    assert doc.stats.pages == 1
    assert doc.stats.tables >= 1
    assert doc.tables[0].rows
