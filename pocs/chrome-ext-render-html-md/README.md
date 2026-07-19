# GitHub Render

GitHub Render turns HTML and Markdown files into calm, readable documents as you browse GitHub. It works on GitHub file pages and raw file URLs, and a toolbar switch lets you enable or disable the behavior at any time.

## What it does

- Renders `.html` and `.htm` files in an isolated frame
- Resolves relative images and preserves page CSS and JavaScript
- Renders `.md`, `.markdown`, `.mdown`, and `.mkd` files as formatted documents
- Supports headings, links, images, lists, tasks, tables, quotes, code fences, and inline formatting
- Follows GitHub navigation without requiring a page refresh
- Keeps the enabled state in Chrome sync storage
- Provides one-click access to the original source and raw file
- Uses no third-party runtime libraries and sends no analytics

## Install

Run:

```bash
./install.sh
```

Chrome will open its extensions page and the built extension folder. Enable **Developer mode**, select **Load unpacked**, and choose the `dist` folder shown by the script.

Chrome requires this final confirmation for locally built extensions. After loading it once, rebuilding the extension only requires selecting the refresh button on its Chrome extensions card.

## Use

Open an HTML or Markdown file in a GitHub repository. GitHub Render replaces the source view with the rendered document.

Use **View source** in the document toolbar to return to GitHub for the current file. Use the extension button in the Chrome toolbar to turn automatic rendering on or off across all GitHub tabs.

## Build

Run:

```bash
./build.sh
```

The build creates:

- `dist/` for Chrome's **Load unpacked** flow
- `github-render.zip` as a portable release archive

The project uses plain HTML, CSS, and JavaScript. No dependency installation is required.

## Test

Run:

```bash
npm test
```

The test suite checks Markdown formatting, unsafe markup handling, the Manifest V3 configuration, and required package files.

## Uninstall

Run:

```bash
./uninstall.sh
```

The script removes local build artifacts and opens Chrome's extensions page. Select **GitHub Render**, then choose **Remove** to clear Chrome's locally loaded entry.

## Security

Markdown input is escaped before formatting. HTML runs in a Chrome sandbox with scripts enabled but without access to the extension, GitHub, a trusted origin, forms, popups, or top navigation. Embedded frames and browser redirects are removed before display.

Source content is requested only from GitHub and `raw.githubusercontent.com`.

## Files

```text
manifest.json
src/
  background.js
  content.css
  content.js
  html-viewer.html
  html-viewer.js
  popup.css
  popup.html
  popup.js
  renderer.js
scripts/
  generate-icons.js
test/
  package.test.js
  renderer.test.js
build.sh
install.sh
uninstall.sh
```
