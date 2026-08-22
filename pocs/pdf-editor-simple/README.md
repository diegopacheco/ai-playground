# PDF Editor Simple

A visual PDF editor that runs in the browser on your own machine. Open any PDF and
**click a line of text to retype it, drag a line to move it, highlight it, or write
on the page**. Whole pages can be rotated, deleted, dragged into a new order or
merged in from another file. The page operations are also a command line tool, so
anything you can do by clicking to a page you can do in a script.

Nothing leaves the machine. The server binds to `127.0.0.1` and the file you open is
held in memory until you replace it.

![A paper with a line highlighted and a note written on the page](screenshot-text.png)

![The editor with three pages selected and rotated](screenshot.png)

![The empty editor waiting for a file](screenshot-empty.png)

## What it does

| Action        | How                                                          |
|---------------|--------------------------------------------------------------|
| Open a PDF    | `Open`, or drop the file anywhere on the page                 |
| Select pages  | click a page, shift-click for a range, `Cmd/Ctrl+A` for all   |
| Rotate        | `Rotate left` / `Rotate right`, a quarter turn each press     |
| Delete        | `Delete`, or the `Backspace` key                              |
| Keep only     | drops every page you did not select                           |
| Reorder       | drag a page and drop it before or after another               |
| Merge         | `Add PDF`, or drop a second file in, appends its pages        |
| Undo          | `Undo`, or `Cmd/Ctrl+Z`, back through the last 30 edits       |
| Edit text     | select one page, `Edit text`, then click any line and retype  |
| Move a line   | drag it to a new place on the page                            |
| Highlight     | the `Highlight` tool, then drag a marker over anything         |
| Write on it   | the `Write` tool, then click and type                          |
| Save          | `Save` downloads the edited PDF                               |

Page edits are held as a list of page slots, each one pointing at a page in an
opened file plus a rotation. Nothing is rewritten while you work, so undo is cheap
and the original bytes stay untouched. The PDF is built once, when you save.

## Editing the text

`Edit text` renders the page and lays a clickable box over every line it finds.
Click one, retype it, press Enter. Escape cancels, `Undo` reverts, and the page
thumbnail updates. Dragging a line moves it, and it keeps its own typeface as it
goes.

Two tools sit next to `Select`. `Highlight` draws a marker over anything you drag
across, in a translucent yellow that multiplies with the page so the words stay
readable underneath. `Write` puts new text wherever you click. Both can be dragged
around afterwards, a written note can be clicked and retyped, and both are only
written into the file when you save.

A PDF stores positioned glyphs, not sentences, so how a line can be changed depends
on how the file was produced. The editor picks the best of three routes per line and
tells you which one it used when you hover it:

| Route      | When it is used                                              | What you get                           |
|------------|--------------------------------------------------------------|----------------------------------------|
| in place   | the line is a single text operator whose font can encode it   | **the file is barely touched**         |
| replaced   | the line is several operators, or it moved                    | **the original font and size are kept**|
| covered    | the text is drawn somewhere this editor cannot reach into     | old text covered, redrawn in Helvetica |

The first two keep the typeface. A line is redrawn with the page's own font
resource, so an embedded font stays exactly as it was, as long as the characters you
type exist in it. For an embedded subset that is decided by reading the font's own
`/ToUnicode` map, and for the standard fonts every Latin character is available. Only
a character that is genuinely absent falls back to Helvetica.

Size follows the same care. A font asked for at 26.67pt inside a page scaled by 0.75
is 20pt on paper, so the text matrix and the current transform are multiplied out and
the line is redrawn at the size a reader actually sees.

Which route a line takes is decided by position. The content stream is walked while
tracking the text matrix, so every text operator has a place on the page, and the
line you clicked is matched to the operators that actually drew it. That is why a
line can be edited even in a file whose text is broken into kerned fragments, as
LaTeX does.

New text is drawn with the graphics state reset first. Files written by word
processors often start the page with a transform that is never restored, such as
`1 0 0 -1 0 792 cm` to flip the y axis, and anything appended after it inherits that
flip and lands mirrored somewhere off the page. Wrapping the page's own drawing puts
the coordinate system back to the page default before the new line is written.

Every route keeps the position, the size and the colour of the line it replaces. The
covering rectangle is filled with the page's own dominant background colour, not
plain white, so it disappears on tinted pages. Turned pages are handled: a page with
a `/Rotate` of 90, 180 or 270 has its boxes mapped into the turned view, so the line
you click is the line you see.

### What it will not do

- **No reflow.** A longer line does not push the rest of the page down, because a
  PDF has no paragraphs to reflow. A much longer line can run into what sits next
  to it.
- **The covered route leaves the old text in the file.** It is hidden from view and
  from this editor, but a text search or a copy and paste still finds the old words.
  The other two routes remove it properly, which is the usual case.
- **A subset font may not have the letter you typed.** Embedded fonts usually carry
  only the characters the document already used. If one is missing, that line falls
  back to Helvetica rather than drawing a wrong glyph.
- **Latin characters only when falling back.** A character Helvetica cannot draw is
  refused with a message rather than quietly turned into a question mark.
- **Written notes are Helvetica**, and a highlight is a rectangle, not a shaped
  outline of the glyphs.

## Requirements

- Python 3.14.6

## Install dependencies

```bash
./install-deps.sh
```

## Run the editor

```bash
./serve.sh
```

Then open http://127.0.0.1:8099. `PORT=9000 ./serve.sh` picks another port.

## Command line

The page operations without the browser. Text editing is browser only. Page selection uses one syntax everywhere,
1-based like a page number on paper.

```bash
./run.sh <command> [options]
```

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

```
-p 2          a single page
-p 1-3        an inclusive range, pages 1, 2 and 3
-p 1-3,7      ranges and single pages together
-p 3,1        page 3, then page 1
```

A page outside the document, a backwards range like `4-2`, text instead of a number,
or a selection that would leave an empty document is an error. The command exits
non-zero and writes no file, so a bad selection never leaves a half edited PDF on
disk. The editor refuses the same things, and says so instead of silently doing less
than you asked.

## Result

```
$ ./run.sh sample -o out/deck.pdf -n 8
out/deck.pdf
```

Opened in the editor, pages 2, 3 and 4 selected and rotated right, then page 8
dragged to the front and saved:

```
$ ./run.sh info out/edited.pdf
out/edited.pdf: 8 pages, encrypted=False
producer: pypdf
page 1: 612x792 pt, rotation 0
page 2: 612x792 pt, rotation 0
page 3: 612x792 pt, rotation 90
page 4: 612x792 pt, rotation 90
page 5: 612x792 pt, rotation 90
page 6: 612x792 pt, rotation 0
page 7: 612x792 pt, rotation 0
page 8: 612x792 pt, rotation 0

$ ./run.sh text out/edited.pdf | head -4
--- page 1
Page 8
--- page 2
Page 1
```

The same file from the command line only:

```
$ ./run.sh extract out/deck.pdf -p 3,1 -o out/reordered.pdf
out/reordered.pdf

$ ./run.sh text out/reordered.pdf
--- page 1
Page 3
--- page 2
Page 1

$ ./run.sh split out/reordered.pdf -o out/pages
out/pages/page-001.pdf
out/pages/page-002.pdf
```

Refused selections:

```
$ ./run.sh extract out/deck.pdf -p 99 -o out/x.pdf
error: page 99 is outside 1-8

$ ./run.sh delete out/deck.pdf -p 1-8 -o out/x.pdf
error: deleting every page would leave an empty document

$ ./run.sh rotate out/deck.pdf -p 1 -a 45 -o out/x.pdf
error: angle must be a multiple of 90
```

## Tests

```bash
./test.sh
```

```
Ran 66 tests in 0.074s

OK
```

The tests cover the three places every edit is decided. `test_pages.py` pins page
selection: order is kept so `extract` can reorder, a page named twice is written
once, ranges are inclusive, and out of range, backwards, non numeric and empty
selections raise instead of quietly selecting fewer pages. `test_document.py` pins
the editor model against the file it produces: the saved PDF follows the order shown
on screen, rotation reaches the file and wraps at 360, undo restores what was
deleted, a freshly opened file has nothing to undo, and a reorder that would lose a
page is refused. `test_textedit.py` pins the text layer: escaped and octal strings
survive the tokenizer, a kerned array counts as one operation, a single operator in
a standard font is edited in place and leaves no covering rectangle, a line split
across two operators has its old text removed rather than hidden, a redrawn line
keeps the position of the line it replaced, and a character Helvetica cannot draw is
refused instead of mangled. Four of its cases are there because they were bugs: an
operator's position must follow the text matrix and the current transform, a clicked
box must stay on the page whatever the `/Rotate`, editing the same line twice must
leave no trace of the first edit, and new text must land where the old text was even
when the page begins with a transform it never restores.

## How the pieces fit

The browser holds no PDF logic. It draws the page list the server sends, and posts
back what you clicked. Every edit returns the whole new state, so the screen cannot
drift from the document.

```
web/app.js ──POST /open, /add, /op, /text──▶ server.py ──▶ document.py ──▶ pypdf
     ▲                                           │
     └────── page list as JSON ──────────────────┘
  POST /move, /note ────────────────────────▶ document.py
       GET /runs?uid=<page>  ──▶ textedit.py ──▶ textmap.py ──▶ content.py
       GET /thumb|/view/<source>/<page>.png ──▶ render.py ──▶ pypdfium2 ──▶ png.py
       GET /save ─────────────────────────────▶ document.save()
```

Thumbnails are rendered once per source page and cached, and rotation is applied in
the browser with a CSS transform, so turning a page is instant and never re-renders.

## Layout

```
pdf-editor-simple
├── src
│   ├── server.py    HTTP endpoints, one document in memory
│   ├── document.py  the editor model: page slots, undo, save
│   ├── textedit.py  applying a text change, a move or an annotation to a page
│   ├── fontmap.py   reading a font's own character map, to reuse its glyphs
│   ├── textmap.py   finding the editable lines and how each can be changed
│   ├── content.py   content stream tokenizer
│   ├── render.py    page thumbnails and page views, cached
│   ├── png.py       PNG encoder, zlib and struct only
│   ├── main.py      command line argument parsing
│   ├── editor.py    command line page operations
│   ├── pages.py     page selection parsing
│   └── sample.py    raw PDF bytes writer for the sample file
├── web
│   ├── index.html
│   ├── app.css
│   └── app.js
├── test_pages.py
├── test_document.py
├── test_textedit.py
├── install-deps.sh
├── serve.sh
├── run.sh
└── test.sh
```

## Stack

- Python 3.14.6, standard library HTTP server
- [pypdf](https://pypi.org/project/pypdf/) to read and write PDF structure
- [pypdfium2](https://pypi.org/project/pypdfium2/) to rasterize pages and to locate
  the text on them
- No frontend framework, no JavaScript dependencies, no PNG library
