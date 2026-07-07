const cv = document.getElementById("game");
const ctx = cv.getContext("2d");
const W = cv.width;
const H = cv.height;
const cam = document.getElementById("cam");
const grab = document.getElementById("grab");
const gctx = grab.getContext("2d");
const pip = document.getElementById("pip");
const pctx = pip.getContext("2d");
const statusEl = document.getElementById("status");

const goal = { x: 190, y: 64, w: 580, h: 142 };
const spot = { x: 480, y: 502 };
const kickPoint = { x: 526, y: 528 };
const gloves = [];
const fx = [];

let mode = "menu";
let frame = 0;
let score = 0;
let saves = 0;
let goals = 0;
let streak = 0;
let best = Number(localStorage.getItem("gestureGoalkeeperBest") || 0);
let delay = 0;
let shake = 0;
let flash = 0;
let ball = null;
let ws = null;
let wsReady = false;
let camReady = false;
let awaiting = false;
let mouse = { x: W / 2, y: H / 2, seen: false };
let ac = null;

function audio() {
  if (!ac) ac = new (window.AudioContext || window.webkitAudioContext)();
  if (ac.state === "suspended") ac.resume();
  return ac;
}

function tone(type, a, b, len, vol) {
  const au = audio();
  const t = au.currentTime;
  const o = au.createOscillator();
  const g = au.createGain();
  o.type = type;
  o.frequency.setValueAtTime(a, t);
  o.frequency.exponentialRampToValueAtTime(b, t + len);
  g.gain.setValueAtTime(vol, t);
  g.gain.exponentialRampToValueAtTime(0.01, t + len);
  o.connect(g);
  g.connect(au.destination);
  o.start(t);
  o.stop(t + len);
}

function saveSound() { tone("square", 260, 760, .18, .25); }
function goalSound() { tone("sawtooth", 180, 58, .4, .4); }
function kickSound() { tone("triangle", 110, 310, .12, .26); }

function resetMatch() {
  score = 0;
  saves = 0;
  goals = 0;
  streak = 0;
  delay = 36;
  fx.length = 0;
  ball = null;
  mode = "play";
}

function spawnBall() {
  const power = Math.min(1.9, 1 + saves * .045);
  const targetX = goal.x + 70 + Math.random() * (goal.w - 140);
  const targetY = goal.y + 36 + Math.random() * 76;
  const frames = Math.max(32, 62 - saves * 1.35);
  const bend = (Math.random() * 2 - 1) * (1.2 + saves * .025);
  ball = {
    x: kickPoint.x,
    y: kickPoint.y,
    vx: (targetX - kickPoint.x) / frames,
    vy: (targetY - kickPoint.y) / frames,
    r: 14,
    t: 0,
    bend,
    spin: Math.random() * Math.PI * 2,
    power,
    done: false
  };
  kickSound();
  addBurst(kickPoint.x, kickPoint.y, "#d8f1c1", 10, 1.2);
}

function addText(x, y, text, color) {
  fx.push({ kind: "text", x, y, text, color, t: 0, ttl: 58 });
}

function addBurst(x, y, color, n, speed) {
  for (let i = 0; i < n; i++) {
    const a = Math.random() * Math.PI * 2;
    const s = speed + Math.random() * speed;
    fx.push({ kind: "dot", x, y, vx: Math.cos(a) * s, vy: Math.sin(a) * s, color, t: 0, ttl: 24 + Math.random() * 24, r: 3 + Math.random() * 5 });
  }
}

function saveBall(g) {
  if (!ball || ball.done) return;
  ball.done = true;
  score += 100 + streak * 20;
  saves++;
  streak++;
  best = Math.max(best, score);
  localStorage.setItem("gestureGoalkeeperBest", String(best));
  flash = 12;
  shake = Math.max(shake, 6);
  saveSound();
  addText(ball.x, ball.y - 18, "SAVE +" + (100 + (streak - 1) * 20), "#bdf58a");
  addBurst(ball.x, ball.y, "#bdf58a", 18, 2.6);
  ball.vx = (ball.x - g.x) * .05;
  ball.vy = 7;
  delay = 46;
}

function scoreGoal() {
  if (!ball || ball.done) return;
  ball.done = true;
  goals++;
  streak = 0;
  shake = Math.max(shake, 12);
  goalSound();
  addText(ball.x, goal.y + 124, "GOAL", "#ff6b6b");
  addBurst(ball.x, ball.y, "#ff6b6b", 16, 2.2);
  delay = 58;
  if (goals >= 5) mode = "over";
}

function updateGloves(hands) {
  const src = hands.length ? hands : (mouse.seen ? [{ x: mouse.x / W, y: mouse.y / H, size: .16 }] : []);
  while (gloves.length < src.length) gloves.push({ x: W / 2, y: H / 2, r: 72, seen: 0 });
  gloves.length = src.length;
  for (let i = 0; i < src.length; i++) {
    const s = src[i];
    const g = gloves[i];
    const tx = s.x * W;
    const ty = s.y * H;
    g.x += (tx - g.x) * .42;
    g.y += (ty - g.y) * .42;
    g.r = 48 + Math.min(.32, Math.max(.08, s.size || .16)) * 170;
    g.seen = 12;
  }
}

function update() {
  frame++;
  if (shake > .1) shake *= .82;
  else shake = 0;
  if (flash > 0) flash--;
  if (mode === "play") {
    if (!ball && delay > 0) delay--;
    if (!ball && delay <= 0) spawnBall();
    if (ball) {
      ball.t++;
      ball.spin += .24 * ball.power;
      ball.x += ball.vx + Math.sin(ball.t * .1) * ball.bend;
      ball.y += ball.vy;
      ball.r = Math.max(8, 15 - ball.t * .04);
      if (!ball.done) {
        for (const g of gloves) {
          if (Math.hypot(ball.x - g.x, ball.y - g.y) < g.r + ball.r * .7) saveBall(g);
        }
        if (ball.y <= goal.y + goal.h - 12) scoreGoal();
      } else if (ball.y > H + 80 || ball.y < -80 || ball.x < -80 || ball.x > W + 80) {
        ball = null;
      }
    }
    if (ball && ball.done && delay > 0) delay--;
    if (ball && ball.done && delay <= 0) ball = null;
  }
  for (let i = fx.length - 1; i >= 0; i--) {
    const e = fx[i];
    e.t++;
    if (e.kind === "dot") {
      e.x += e.vx;
      e.y += e.vy;
      e.vy += .04;
    }
    if (e.kind === "text") e.y -= .7;
    if (e.t >= e.ttl) fx.splice(i, 1);
  }
}

function rect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawPitch() {
  const grd = ctx.createLinearGradient(0, 0, 0, H);
  grd.addColorStop(0, "#2f8f52");
  grd.addColorStop(.52, "#186c3d");
  grd.addColorStop(1, "#0d442b");
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, W, H);
  ctx.globalAlpha = .18;
  ctx.fillStyle = "#d9f7c7";
  for (let i = -2; i < 12; i++) ctx.fillRect(i * 96 + (frame % 96), 0, 48, H);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "rgba(245,249,222,.72)";
  ctx.lineWidth = 5;
  ctx.strokeRect(84, 26, W - 168, H - 52);
  ctx.beginPath();
  ctx.arc(W / 2, 428, 86, Math.PI, 0);
  ctx.stroke();
}

function drawGoal() {
  ctx.fillStyle = "rgba(8,25,20,.58)";
  rect(goal.x - 22, goal.y - 18, goal.w + 44, goal.h + 44, 10);
  ctx.fill();
  ctx.strokeStyle = "rgba(244,246,230,.42)";
  ctx.lineWidth = 1;
  for (let x = goal.x; x <= goal.x + goal.w; x += 32) {
    ctx.beginPath();
    ctx.moveTo(x, goal.y);
    ctx.lineTo(x, goal.y + goal.h);
    ctx.stroke();
  }
  for (let y = goal.y; y <= goal.y + goal.h; y += 24) {
    ctx.beginPath();
    ctx.moveTo(goal.x, y);
    ctx.lineTo(goal.x + goal.w, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "#fff6d6";
  ctx.lineWidth = 12;
  ctx.strokeRect(goal.x, goal.y, goal.w, goal.h);
  ctx.strokeStyle = "#a87939";
  ctx.lineWidth = 4;
  ctx.strokeRect(goal.x - 8, goal.y - 8, goal.w + 16, goal.h + 16);
}

function drawShooter() {
  ctx.save();
  ctx.translate(spot.x, spot.y + 6);
  ctx.lineCap = "round";
  ctx.fillStyle = "#1c2530";
  rect(-18, -72, 36, 58, 15);
  ctx.fill();
  ctx.fillStyle = "#ffd9a0";
  ctx.beginPath();
  ctx.arc(0, -88, 17, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#0e161b";
  ctx.lineWidth = 8;
  ctx.beginPath();
  ctx.moveTo(-10, -16);
  ctx.lineTo(-42, 34);
  ctx.moveTo(10, -16);
  ctx.lineTo(38, 20);
  ctx.stroke();
  ctx.strokeStyle = "#f7f2df";
  ctx.lineWidth = 11;
  ctx.beginPath();
  ctx.moveTo(28, 24);
  ctx.lineTo(52, 20);
  ctx.moveTo(-32, 38);
  ctx.lineTo(-56, 34);
  ctx.stroke();
  ctx.restore();
}

function drawBall() {
  if (!ball) return;
  ctx.save();
  ctx.translate(ball.x, ball.y);
  ctx.rotate(ball.spin);
  const shade = ctx.createRadialGradient(-5, -7, 2, 0, 0, ball.r * 1.2);
  shade.addColorStop(0, "#ffffff");
  shade.addColorStop(.78, "#e9e7dd");
  shade.addColorStop(1, "#a5a094");
  ctx.fillStyle = shade;
  ctx.beginPath();
  ctx.arc(0, 0, ball.r, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#111";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(-ball.r * .7, 0);
  ctx.lineTo(ball.r * .7, 0);
  ctx.moveTo(0, -ball.r * .7);
  ctx.lineTo(0, ball.r * .7);
  ctx.stroke();
  ctx.restore();
}

function drawGloves() {
  for (const g of gloves) {
    const grd = ctx.createRadialGradient(g.x - 16, g.y - 20, 8, g.x, g.y, g.r);
    grd.addColorStop(0, "rgba(255,245,189,.98)");
    grd.addColorStop(.42, "rgba(255,214,96,.86)");
    grd.addColorStop(1, "rgba(255,84,84,.72)");
    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(g.x, g.y, g.r, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,.78)";
    ctx.lineWidth = 5;
    ctx.stroke();
    ctx.fillStyle = "rgba(8,20,16,.26)";
    rect(g.x - g.r * .52, g.y - g.r * .2, g.r * 1.04, g.r * .4, 8);
    ctx.fill();
  }
}

function drawHud() {
  ctx.fillStyle = "rgba(2,12,9,.68)";
  rect(20, 18, 316, 48, 8);
  ctx.fill();
  ctx.fillStyle = "#fff6d6";
  ctx.font = "800 22px Avenir, Verdana, sans-serif";
  ctx.fillText("SCORE " + score, 38, 49);
  ctx.fillStyle = "#ffd660";
  ctx.fillText("BEST " + best, 196, 49);
  ctx.fillStyle = "rgba(2,12,9,.68)";
  rect(W - 252, 18, 232, 48, 8);
  ctx.fill();
  ctx.fillStyle = "#bdf58a";
  ctx.fillText("SAVES " + saves, W - 236, 49);
  ctx.fillStyle = "#ff8d7f";
  ctx.fillText("GOALS " + goals + "/5", W - 112, 49);
}

function drawFx() {
  for (const e of fx) {
    const a = 1 - e.t / e.ttl;
    ctx.globalAlpha = a;
    if (e.kind === "dot") {
      ctx.fillStyle = e.color;
      ctx.beginPath();
      ctx.arc(e.x, e.y, e.r * a, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.fillStyle = e.color;
      ctx.font = "900 30px Avenir, Verdana, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(e.text, e.x, e.y);
      ctx.textAlign = "left";
    }
    ctx.globalAlpha = 1;
  }
}

function drawOverlay() {
  if (mode === "menu" || mode === "over") {
    ctx.fillStyle = "rgba(3,12,9,.72)";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#ffd660";
    ctx.font = "900 72px Avenir, Verdana, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("GESTURE GOALKEEPER", W / 2, 202);
    ctx.fillStyle = "#f7f2df";
    ctx.font = "800 24px Avenir, Verdana, sans-serif";
    const line = mode === "menu" ? "Move your hand to block penalty shots" : "Final score " + score + " with " + saves + " saves";
    ctx.fillText(line, W / 2, 250);
    ctx.fillStyle = "#bdf58a";
    ctx.font = "800 18px Avenir, Verdana, sans-serif";
    ctx.fillText("Click or press space to start", W / 2, 296);
    ctx.textAlign = "left";
  }
}

function draw() {
  ctx.save();
  if (shake) ctx.translate((Math.random() * 2 - 1) * shake, (Math.random() * 2 - 1) * shake);
  drawPitch();
  drawGoal();
  drawShooter();
  drawBall();
  drawGloves();
  drawFx();
  drawHud();
  if (flash > 0) {
    ctx.fillStyle = "rgba(189,245,138," + flash / 38 + ")";
    ctx.fillRect(0, 0, W, H);
  }
  drawOverlay();
  ctx.restore();
}

function loop() {
  update();
  draw();
  requestAnimationFrame(loop);
}

function start() {
  if (mode === "menu" || mode === "over") resetMatch();
}

function connectWs() {
  ws = new WebSocket("ws://" + location.hostname + ":8765");
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { wsReady = true; setStatus(); };
  ws.onclose = () => { wsReady = false; setStatus(); setTimeout(connectWs, 1000); };
  ws.onerror = () => { wsReady = false; setStatus(); };
  ws.onmessage = e => {
    awaiting = false;
    try {
      const msg = JSON.parse(e.data);
      updateGloves(Array.isArray(msg.hands) ? msg.hands : []);
    } catch (err) {
      updateGloves([]);
    }
  };
}

function setStatus() {
  if (camReady && wsReady) statusEl.textContent = "camera tracking ready";
  else if (wsReady) statusEl.textContent = "allow camera or use mouse";
  else statusEl.textContent = "connecting...";
}

function capture() {
  if (camReady && wsReady && ws.readyState === 1 && !awaiting) {
    gctx.drawImage(cam, 0, 0, grab.width, grab.height);
    pctx.drawImage(cam, 0, 0, pip.width, pip.height);
    grab.toBlob(blob => {
      if (!blob || !wsReady || ws.readyState !== 1) return;
      awaiting = true;
      ws.send(blob);
    }, "image/jpeg", .58);
  } else {
    pctx.fillStyle = "#050807";
    pctx.fillRect(0, 0, pip.width, pip.height);
    pctx.fillStyle = "#ffd660";
    pctx.font = "800 13px Avenir, Verdana, sans-serif";
    pctx.fillText("mouse fallback", 24, 70);
  }
  setTimeout(capture, 80);
}

navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
  .then(stream => {
    cam.srcObject = stream;
    camReady = true;
    setStatus();
  })
  .catch(() => {
    camReady = false;
    setStatus();
  });

cv.addEventListener("mousemove", e => {
  const r = cv.getBoundingClientRect();
  mouse.x = (e.clientX - r.left) / r.width * W;
  mouse.y = (e.clientY - r.top) / r.height * H;
  mouse.seen = true;
  if (!gloves.length) updateGloves([]);
});
cv.addEventListener("click", start);
window.addEventListener("keydown", e => {
  if (e.code === "Space") start();
});

connectWs();
capture();
loop();
