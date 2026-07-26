# LumaOS 2003

LumaOS is a self-contained browser desktop inspired by the tactile, colorful personal computers of the early 2000s. It uses plain HTML, CSS and JavaScript with no runtime libraries.

## Run

```bash
./start.sh
```

The script searches from port `4173`, prints the selected port and displays the full address.

Set a different starting port when needed:

Use another port when needed:

```bash
WEBOS_PORT=8080 ./start.sh
```

Stop the exact tracked local server and its selected port:

```bash
./stop.sh
```

Run the local checks:

```bash
./test.sh
```

Run the Chromium interaction checks after installing development dependencies:

```bash
npm install
WEBOS_PORT=43174 ./start.sh
npx playwright test tests/webos.spec.js --browser chromium --workers 1
WEBOS_PORT=43174 ./stop.sh
```

## Included apps

- My Computer with persistent folders and text files
- Notepad with local saving and text download
- Luma Paint with pencil, brush, eraser, undo and PNG saving
- Safe Terminal with a small allowed command set
- Luma Explorer with restricted web framing
- Picture Viewer with five bundled scenes
- Video Player with two local canvas animations
- Clock, Desktop Properties, Help and Recycle Bin

## Desktop controls

- Double-click an icon to open an app
- Drag title bars to move windows
- Use title bar controls to minimize, maximize or close
- Right-click the wallpaper for desktop actions
- Use Desktop Properties for five bundled wallpapers or a direct image link
- Open apps from the Start menu or taskbar

## Storage and safety

Notes, folders and personalization settings use browser local storage. Clearing site data resets them.

Safe Terminal never runs system shell commands. It accepts only `help`, `pwd`, `ls`, `cd`, `mkdir`, `touch`, `cat`, `echo`, `date`, `whoami`, `hostname`, `history`, `clear` and `open`. Destructive command names and shell operators are rejected.

Luma Explorer accepts only HTTP and HTTPS addresses and loads pages inside a restricted frame. Some sites prevent framed display by policy.

The scripts bind the server to `127.0.0.1`, track one explicit process ID, wait in one-second checks and never scan or stop unrelated processes.
