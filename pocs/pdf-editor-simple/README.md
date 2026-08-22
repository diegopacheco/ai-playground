# PDF Editor Simple

A small command line PDF editor in Python. It reads a PDF, changes its pages, and
writes a new file. Page selection is the whole idea: every command that touches
pages takes the same `1-3,7` syntax, is 1-based like a page number on paper, and
refuses anything it cannot honour instead of silently doing less than you asked.

One dependency, `pypdf`, for reading and writing PDF structure. Everything else is
the standard library, including the sample file writer, which emits raw PDF bytes
so there is nothing to install just to have something to open.

## Commands

| Command   | What it does                                            |
|-----------|---------------------------------------------------------|
| `info`    | page count, page size in points, rotation, metadata     |
| `sample`  | write a small multi page PDF to work on                 |
| `extract` | keep the given pages, **in the order given**            |
| `delete`  | drop the given pages, keep the rest in document order   |
| `rotate`  | turn the given pages clockwise by a multiple of 90      |
| `merge`   | join several PDFs into one                              |
| `split`   | write every page as its own file                        |
| `text`    | print the text of the given pages                       |

`extract` doubles as the reorder tool. `-p 3,1` means page 3 first, then page 1,
because the selection keeps the order you typed rather than sorting it.

## Page selection

```
-p 2          a single page
-p 1-3        an inclusive range, pages 1, 2 and 3
-p 1-3,7      ranges and single pages together
-p 3,1        page 3, then page 1
```

A page outside the document, a backwards range like `4-2`, text instead of a
number, or a selection that would leave an empty document is an error. The command
exits non-zero and writes no file, so a bad selection never leaves a half edited PDF
on disk.

## Requirements

- Python 3.14.6

## Install dependencies

```bash
./install-deps.sh
```

## Run

```bash
./run.sh <command> [options]
```

## Result

```
$ ./run.sh sample -o out/report.pdf -n 4
out/report.pdf

$ ./run.sh info out/report.pdf
out/report.pdf: 4 pages, encrypted=False
page 1: 612x792 pt, rotation 0
page 2: 612x792 pt, rotation 0
page 3: 612x792 pt, rotation 0
page 4: 612x792 pt, rotation 0

$ ./run.sh extract out/report.pdf -p 3,1 -o out/reordered.pdf
out/reordered.pdf

$ ./run.sh text out/reordered.pdf
--- page 1
Page 3
--- page 2
Page 1

$ ./run.sh delete out/report.pdf -p 2 -o out/trimmed.pdf
out/trimmed.pdf

$ ./run.sh rotate out/trimmed.pdf -p 1 -a 90 -o out/rotated.pdf
out/rotated.pdf

$ ./run.sh merge out/reordered.pdf out/rotated.pdf -o out/merged.pdf
out/merged.pdf

$ ./run.sh split out/merged.pdf -o out/pages
out/pages/page-001.pdf
out/pages/page-002.pdf
out/pages/page-003.pdf
out/pages/page-004.pdf
out/pages/page-005.pdf

$ ./run.sh info out/merged.pdf
out/merged.pdf: 5 pages, encrypted=False
producer: pypdf
page 1: 612x792 pt, rotation 0
page 2: 612x792 pt, rotation 0
page 3: 612x792 pt, rotation 90
page 4: 612x792 pt, rotation 0
page 5: 612x792 pt, rotation 0
```

Refused selections:

```
$ ./run.sh extract out/report.pdf -p 9 -o out/x.pdf
error: page 9 is outside 1-4

$ ./run.sh delete out/report.pdf -p 1-4 -o out/x.pdf
error: deleting every page would leave an empty document

$ ./run.sh rotate out/report.pdf -p 1 -a 45 -o out/x.pdf
error: angle must be a multiple of 90
```

## Tests

```bash
./test.sh
```

```
Ran 10 tests in 0.000s

OK
```

The tests cover `parse_pages` and `invert_pages`, where every page selection in the
tool is decided. They pin the behaviour the commands depend on: selections keep the
order you typed so `extract` can reorder, a page named twice is written once, ranges
are inclusive, and out of range, backwards, non numeric and empty selections raise
instead of quietly selecting fewer pages.

## Layout

```
pdf-editor-simple
├── src
│   ├── main.py     argument parsing and command dispatch
│   ├── editor.py   the page operations, on top of pypdf
│   ├── pages.py    page selection parsing
│   └── sample.py   raw PDF bytes writer for the sample file
├── test_pages.py
├── install-deps.sh
├── run.sh
└── test.sh
```

## Stack

- Python 3.14.6
- [pypdf](https://pypi.org/project/pypdf/)
