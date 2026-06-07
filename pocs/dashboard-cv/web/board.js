const board = document.getElementById("board");
const ctx = board.getContext("2d");
const cam = document.getElementById("cam");
const grab = document.getElementById("grab");
const gctx = grab.getContext("2d");
const pip = document.getElementById("pip");
const pctx = pip.getContext("2d");

const elStatus = document.getElementById("status");
const elTool = document.getElementById("tool");
const stage = document.querySelector(".stage");

const W = board.width;
const H = board.height;
const TOOLBAR_H = 66;
const NUM_PENS = 2;

const COLORS = ["#111827", "#ef4444", "#f59e0b", "#22c55e", "#3b82f6", "#a855f7"];
const COLOR_NAMES = ["black", "red", "amber", "green", "blue", "violet"];
const SIZES = [4, 9, 18];
const SIZE_NAMES = ["thin", "medium", "thick"];
const PEN_TINT = ["#0ea5e9", "#ec4899"];

const items = [];
COLORS.forEach((c, i) => items.push({ kind: "color", i, color: c }));
items.push({ kind: "eraser" });
SIZES.forEach((s, i) => items.push({ kind: "size", i, size: s }));
items.push({ kind: "undo" });
items.push({ kind: "clear" });
const CELL = W / items.length;

const layer = document.createElement("canvas");
layer.width = W;
layer.height = H;
const lctx = layer.getContext("2d");

let strokes = [];
let colorIdx = 0;
let sizeIdx = 1;
let eraser = false;

function makePen() {
  return { present: false, mode: "idle", prevMode: "idle", cursor: { x: W / 2, y: H / 2 }, target: { x: W / 2, y: H / 2 }, rx: 0.5, ry: 0.5, current: null, hasTarget: false };
}
const pens = [makePen(), makePen()];

let camReady = false;
let wsReady = false;

const mouseHolder = { current: null };
let mouseDrawing = false;
let mouseCursor = { x: W / 2, y: H / 2 };
let lastInput = "hand";

let ws = null;
let awaiting = false;
let lastSend = 0;

function clamp01(v) { return Math.min(1, Math.max(0, v)); }
function eraserSize() { return SIZES[sizeIdx] * 2.6; }

function paintSeg(s, a, b) {
  lctx.save();
  lctx.lineCap = "round";
  lctx.lineJoin = "round";
  if (s.erase) {
    lctx.globalCompositeOperation = "destination-out";
    lctx.strokeStyle = "#000";
    lctx.fillStyle = "#000";
  } else {
    lctx.strokeStyle = s.color;
    lctx.fillStyle = s.color;
  }
  lctx.lineWidth = s.size;
  if (!b) {
    lctx.beginPath();
    lctx.arc(a.x, a.y, s.size / 2, 0, Math.PI * 2);
    lctx.fill();
  } else {
    lctx.beginPath();
    lctx.moveTo(a.x, a.y);
    lctx.lineTo(b.x, b.y);
    lctx.stroke();
  }
  lctx.restore();
}

function rebuild() {
  lctx.clearRect(0, 0, W, H);
  for (const s of strokes) {
    if (s.points.length === 1) { paintSeg(s, s.points[0]); continue; }
    for (let i = 1; i < s.points.length; i++) paintSeg(s, s.points[i - 1], s.points[i]);
  }
}

function startStroke(holder, x, y, erase) {
  holder.current = { color: COLORS[colorIdx], size: erase ? eraserSize() : SIZES[sizeIdx], erase, points: [{ x, y }] };
  paintSeg(holder.current, holder.current.points[0]);
}

function extendStroke(holder, x, y) {
  if (!holder.current) return;
  const p = holder.current.points;
  const last = p[p.length - 1];
  if (Math.hypot(x - last.x, y - last.y) < 1.2) return;
  p.push({ x, y });
  paintSeg(holder.current, p[p.length - 2], p[p.length - 1]);
}

function commitStroke(holder) {
  if (holder.current) { strokes.push(holder.current); holder.current = null; }
}

function undo() { for (const p of pens) commitStroke(p); strokes.pop(); rebuild(); }
function clearAll() { for (const p of pens) commitStroke(p); strokes = []; rebuild(); }
function cycleColor() { colorIdx = (colorIdx + 1) % COLORS.length; eraser = false; updateTool(); }

function activateItem(idx) {
  const it = items[idx];
  if (!it) return;
  if (it.kind === "color") { eraser = false; colorIdx = it.i; }
  else if (it.kind === "eraser") { eraser = true; }
  else if (it.kind === "size") { sizeIdx = it.i; }
  else if (it.kind === "undo") { undo(); }
  else if (it.kind === "clear") { clearAll(); }
  updateTool();
}

function updateTool() {
  elTool.textContent = eraser ? "eraser · " + SIZE_NAMES[sizeIdx] : COLOR_NAMES[colorIdx] + " · " + SIZE_NAMES[sizeIdx];
}

function itemIndexAt(x, y) {
  if (y > TOOLBAR_H) return -1;
  const idx = Math.floor(x / CELL);
  if (idx < 0 || idx >= items.length) return -1;
  return idx;
}

function applyPen(pen) {
  if (!pen.present) { commitStroke(pen); pen.prevMode = "idle"; return; }
  const x = pen.cursor.x, y = pen.cursor.y;
  if (pen.mode === "draw") {
    if (y > TOOLBAR_H) { if (!pen.current) startStroke(pen, x, y, false); else extendStroke(pen, x, y); }
    else commitStroke(pen);
  } else if (pen.mode === "erase") {
    if (y > TOOLBAR_H) { if (!pen.current) startStroke(pen, x, y, true); else extendStroke(pen, x, y); }
    else commitStroke(pen);
  } else if (pen.mode === "color") {
    if (pen.prevMode !== "color") cycleColor();
    commitStroke(pen);
  } else {
    commitStroke(pen);
  }
  pen.prevMode = pen.mode;
}

function assignHands(hands) {
  const dets = hands.slice(0, NUM_PENS).map((h) => ({
    tx: clamp01(0.5 + (h.x - 0.5) * 1.35) * W,
    ty: clamp01(0.5 + (h.y - 0.5) * 1.35) * H,
    rx: h.x, ry: h.y, mode: h.mode,
  }));
  const pairs = [];
  for (let d = 0; d < dets.length; d++)
    for (let p = 0; p < NUM_PENS; p++)
      pairs.push({ d, p, dist: Math.hypot(dets[d].tx - pens[p].cursor.x, dets[d].ty - pens[p].cursor.y) });
  pairs.sort((a, b) => a.dist - b.dist);
  const detTaken = new Array(dets.length).fill(false);
  const penTaken = new Array(NUM_PENS).fill(false);
  const seen = new Array(NUM_PENS).fill(false);
  for (const pr of pairs) {
    if (detTaken[pr.d] || penTaken[pr.p]) continue;
    detTaken[pr.d] = true; penTaken[pr.p] = true;
    const pen = pens[pr.p], d = dets[pr.d];
    pen.present = true;
    pen.mode = d.mode;
    pen.target.x = d.tx; pen.target.y = d.ty;
    pen.rx = d.rx; pen.ry = d.ry;
    if (!pen.hasTarget) { pen.cursor.x = d.tx; pen.cursor.y = d.ty; pen.hasTarget = true; }
    seen[pr.p] = true;
  }
  for (let p = 0; p < NUM_PENS; p++) if (!seen[p]) pens[p].present = false;
}

function connectWS() {
  try {
    ws = new WebSocket("ws://" + location.hostname + ":8765");
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { wsReady = true; };
    ws.onmessage = (e) => {
      awaiting = false;
      try {
        const m = JSON.parse(e.data);
        assignHands(Array.isArray(m.hands) ? m.hands : []);
        if (!mouseDrawing) lastInput = "hand";
      } catch (_) {}
    };
    ws.onclose = () => { wsReady = false; awaiting = false; setTimeout(connectWS, 1500); };
    ws.onerror = () => {};
  } catch (_) {
    setTimeout(connectWS, 1500);
  }
}

async function initCam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 }, audio: false });
    cam.srcObject = stream;
    await cam.play();
    camReady = true;
    return true;
  } catch (_) {
    camReady = false;
    return false;
  }
}

function maybeSend(ts) {
  if (!ws || ws.readyState !== 1 || awaiting || !camReady) return;
  if (ts - lastSend < 60) return;
  lastSend = ts;
  awaiting = true;
  gctx.drawImage(cam, 0, 0, grab.width, grab.height);
  grab.toBlob((blob) => {
    if (!blob || !ws || ws.readyState !== 1) { awaiting = false; return; }
    blob.arrayBuffer().then((b) => { if (ws.readyState === 1) ws.send(b); else awaiting = false; });
  }, "image/jpeg", 0.5);
}

function drawGrid() {
  ctx.fillStyle = "#f9f9f5";
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "#e6e6dc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 40; x < W; x += 40) { ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, H); }
  for (let y = 40; y < H; y += 40) { ctx.moveTo(0, y + 0.5); ctx.lineTo(W, y + 0.5); }
  ctx.stroke();
}

function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}

function drawToolbar() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, W, TOOLBAR_H);
  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, TOOLBAR_H + 0.5);
  ctx.lineTo(W, TOOLBAR_H + 0.5);
  ctx.stroke();

  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    const cx = i * CELL + CELL / 2;
    const cy = TOOLBAR_H / 2;
    let active = false;
    if (it.kind === "color") active = !eraser && colorIdx === it.i;
    else if (it.kind === "eraser") active = eraser;
    else if (it.kind === "size") active = sizeIdx === it.i;

    if (it.kind === "color") {
      ctx.fillStyle = it.color;
      ctx.beginPath();
      ctx.arc(cx, cy, 15, 0, Math.PI * 2);
      ctx.fill();
    } else if (it.kind === "eraser") {
      ctx.fillStyle = "#ffd6e7";
      roundRect(ctx, cx - 15, cy - 11, 30, 22, 5);
      ctx.fill();
      ctx.fillStyle = "#9d2c54";
      ctx.font = "bold 11px Trebuchet MS, sans-serif";
      ctx.fillText("ERASE", cx, cy + 1);
    } else if (it.kind === "size") {
      ctx.fillStyle = "#374151";
      ctx.beginPath();
      ctx.arc(cx, cy, it.size / 2 + 1, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.fillStyle = "#374151";
      ctx.font = "bold 13px Trebuchet MS, sans-serif";
      ctx.fillText(it.kind === "undo" ? "UNDO" : "CLEAR", cx, cy + 1);
    }

    if (active) {
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 3;
      roundRect(ctx, i * CELL + 4, 5, CELL - 8, TOOLBAR_H - 10, 10);
      ctx.stroke();
    }
  }
}

function drawCursor(x, y, mode, drawing, tint, label) {
  const col = mode === "erase" ? "#9aa3c0" : COLORS[colorIdx];
  const r = Math.max(8, (mode === "erase" ? eraserSize() : SIZES[sizeIdx]) / 2 + 4);
  ctx.lineWidth = 2;
  ctx.strokeStyle = col;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.stroke();
  if (drawing || mode === "draw") {
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
  let tag = "";
  if (mode === "color") tag = "color";
  else if (mode === "erase") tag = "erase";
  if (tag || label) {
    ctx.fillStyle = tint || "#3b82f6";
    ctx.font = "bold 12px Trebuchet MS, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText((label ? label + " " : "") + tag, x, y - r - 6);
  }
  if (tint) {
    ctx.strokeStyle = tint;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, r + 4, 0, Math.PI * 2);
    ctx.stroke();
  }
}

function render() {
  drawGrid();
  ctx.drawImage(layer, 0, 0);
  drawToolbar();

  if (lastInput === "mouse") {
    drawCursor(mouseCursor.x, mouseCursor.y, eraser ? "erase" : "draw", mouseDrawing, null, null);
  } else {
    const many = pens.filter((p) => p.present).length > 1;
    pens.forEach((pen, i) => {
      if (pen.present) drawCursor(pen.cursor.x, pen.cursor.y, pen.mode, false, many ? PEN_TINT[i] : null, many ? "#" + (i + 1) : null);
    });
  }

  const count = pens.filter((p) => p.present).length;
  if (!wsReady) { elStatus.textContent = "link…"; elStatus.style.color = "#9aa3c0"; }
  else if (!camReady) { elStatus.textContent = "no cam"; elStatus.style.color = "#d97706"; }
  else if (count > 0) { elStatus.textContent = count === 1 ? "1 hand" : count + " hands"; elStatus.style.color = "#059669"; }
  else { elStatus.textContent = "show hand"; elStatus.style.color = "#d97706"; }
}

function drawPip() {
  pctx.save();
  pctx.translate(pip.width, 0);
  pctx.scale(-1, 1);
  if (camReady) {
    pctx.drawImage(cam, 0, 0, pip.width, pip.height);
    pctx.restore();
    const many = pens.filter((p) => p.present).length > 1;
    pens.forEach((pen, i) => {
      if (!pen.present) return;
      const mx = pen.rx * pip.width;
      const my = pen.ry * pip.height;
      const col = pen.mode === "draw" ? "#10b981" : pen.mode === "erase" ? "#9aa3c0" : pen.mode === "color" ? "#a855f7" : "#cbd5e1";
      pctx.fillStyle = many ? PEN_TINT[i] : col;
      pctx.beginPath();
      pctx.arc(mx, my, 9, 0, Math.PI * 2);
      pctx.fill();
      pctx.strokeStyle = "rgba(255,255,255,.9)";
      pctx.lineWidth = 2;
      pctx.stroke();
    });
  } else {
    pctx.restore();
    pctx.fillStyle = "#0b1020";
    pctx.fillRect(0, 0, pip.width, pip.height);
    pctx.fillStyle = "#9aa3c0";
    pctx.font = "14px Trebuchet MS, sans-serif";
    pctx.textAlign = "center";
    pctx.fillText("waiting for camera", pip.width / 2, pip.height / 2);
  }
}

function frame(ts) {
  maybeSend(ts);
  if (lastInput !== "mouse") {
    for (const pen of pens) {
      pen.cursor.x += (pen.target.x - pen.cursor.x) * 0.55;
      pen.cursor.y += (pen.target.y - pen.cursor.y) * 0.55;
      applyPen(pen);
    }
  } else {
    for (const pen of pens) commitStroke(pen);
  }
  render();
  drawPip();
  requestAnimationFrame(frame);
}

function canvasPos(e) {
  const r = board.getBoundingClientRect();
  return { x: (e.clientX - r.left) * (W / r.width), y: (e.clientY - r.top) * (H / r.height) };
}

board.addEventListener("mousedown", (e) => {
  const p = canvasPos(e);
  lastInput = "mouse";
  mouseCursor.x = p.x; mouseCursor.y = p.y;
  if (p.y <= TOOLBAR_H) { const idx = itemIndexAt(p.x, p.y); if (idx >= 0) activateItem(idx); return; }
  mouseDrawing = true;
  startStroke(mouseHolder, p.x, p.y, eraser);
});

window.addEventListener("mousemove", (e) => {
  if (!mouseDrawing) return;
  const p = canvasPos(e);
  mouseCursor.x = p.x; mouseCursor.y = p.y;
  extendStroke(mouseHolder, p.x, p.y);
});

window.addEventListener("mouseup", () => {
  if (mouseDrawing) { mouseDrawing = false; commitStroke(mouseHolder); }
});

document.getElementById("enable").addEventListener("click", initCam);
document.getElementById("undo").addEventListener("click", undo);
document.getElementById("clear").addEventListener("click", clearAll);
document.getElementById("save").addEventListener("click", () => {
  const out = document.createElement("canvas");
  out.width = W; out.height = H;
  const octx = out.getContext("2d");
  octx.fillStyle = "#f9f9f5";
  octx.fillRect(0, 0, W, H);
  octx.drawImage(layer, 0, 0);
  const a = document.createElement("a");
  a.href = out.toDataURL("image/png");
  a.download = "air-board.png";
  a.click();
});

function toggleFullscreen() {
  const on = document.fullscreenElement || document.webkitFullscreenElement;
  if (on) (document.exitFullscreen || document.webkitExitFullscreen).call(document);
  else (stage.requestFullscreen || stage.webkitRequestFullscreen).call(stage);
}
document.getElementById("fs").addEventListener("click", toggleFullscreen);

function syncFullscreen() {
  const on = !!(document.fullscreenElement || document.webkitFullscreenElement);
  stage.classList.toggle("fs", on);
}
document.addEventListener("fullscreenchange", syncFullscreen);
document.addEventListener("webkitfullscreenchange", syncFullscreen);

window.addEventListener("keydown", (e) => {
  if (e.key === "z" || e.key === "Z") undo();
  else if (e.key === "c" || e.key === "C") clearAll();
  else if (e.key === "e" || e.key === "E") { eraser = !eraser; updateTool(); }
  else if (e.key === "x" || e.key === "X") cycleColor();
  else if (e.key === "f" || e.key === "F") toggleFullscreen();
});

updateTool();
connectWS();
initCam();
requestAnimationFrame(frame);
