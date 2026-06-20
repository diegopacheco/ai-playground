const http = require("http");
const fs = require("fs");
const path = require("path");
const { execFile } = require("child_process");

const PORT = process.env.PORT || 4321;
const PUBLIC_DIR = path.join(__dirname, "public");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".ico": "image/x-icon",
};

function serveStatic(req, res) {
  let urlPath = req.url.split("?")[0];
  if (urlPath === "/") urlPath = "/index.html";
  const filePath = path.join(PUBLIC_DIR, path.normalize(urlPath));
  if (!filePath.startsWith(PUBLIC_DIR)) {
    res.writeHead(403);
    res.end("forbidden");
    return;
  }
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end("not found");
      return;
    }
    res.writeHead(200, { "Content-Type": MIME[path.extname(filePath)] || "application/octet-stream" });
    res.end(data);
  });
}

function buildPrompt(s) {
  return [
    "You are the RED army general in a real-time strategy game called Times 2, in the style of Age of Empires II.",
    "Rules: villagers gather gold and cost 50 gold; soldiers fight and cost 80 gold.",
    "You win by destroying the enemy (BLUE) Town Center; you lose if yours falls.",
    "More villagers grow your economy faster; more soldiers grow your military.",
    "Only launch an attack when your army can win or to break a stall.",
    `Battlefield state: gold=${s.gold}, myVillagers=${s.villagers}, mySoldiers=${s.soldiers}, ` +
      `enemyVillagers=${s.enemyVillagers}, enemySoldiers=${s.enemySoldiers}, ` +
      `myBaseHp=${s.myTcHp}, enemyBaseHp=${s.enemyTcHp}, secondsElapsed=${s.gameTime}.`,
    'Decide your next unit to train and whether to send your soldiers to attack now.',
    'Reply with ONLY compact JSON, no prose: {"build":"villager"|"soldier","attack":true|false,"reason":"<max 8 words>"}.',
  ].join(" ");
}

function fallback(s) {
  const wantsArmy = s.soldiers <= s.enemySoldiers || s.villagers >= 5;
  return {
    build: s.villagers < 4 || !wantsArmy ? "villager" : "soldier",
    attack: s.soldiers >= 6 && s.soldiers >= s.enemySoldiers,
    reason: "heuristic reserve plan",
  };
}

function parseDecision(text, s) {
  const match = String(text).match(/\{[\s\S]*\}/);
  if (!match) return null;
  try {
    const d = JSON.parse(match[0]);
    const build = d.build === "soldier" ? "soldier" : "villager";
    return { build, attack: d.attack === true, reason: String(d.reason || "").slice(0, 60) };
  } catch (e) {
    return null;
  }
}

function askClaude(state) {
  return new Promise((resolve) => {
    execFile(
      "claude",
      ["-p", buildPrompt(state), "--model", "sonnet"],
      { timeout: 30000, maxBuffer: 1024 * 1024 },
      (err, stdout) => {
        if (err) {
          resolve({ ...fallback(state), source: "fallback" });
          return;
        }
        const decision = parseDecision(stdout, state);
        if (!decision) {
          resolve({ ...fallback(state), source: "fallback" });
          return;
        }
        resolve({ ...decision, source: "sonnet" });
      }
    );
  });
}

function handleStrategy(req, res) {
  let body = "";
  req.on("data", (chunk) => {
    body += chunk;
    if (body.length > 1e5) req.destroy();
  });
  req.on("end", async () => {
    let state = {};
    try {
      state = JSON.parse(body || "{}");
    } catch (e) {}
    const numeric = (v) => (Number.isFinite(+v) ? +v : 0);
    const safe = {
      gold: numeric(state.gold),
      villagers: numeric(state.villagers),
      soldiers: numeric(state.soldiers),
      enemyVillagers: numeric(state.enemyVillagers),
      enemySoldiers: numeric(state.enemySoldiers),
      myTcHp: numeric(state.myTcHp),
      enemyTcHp: numeric(state.enemyTcHp),
      gameTime: numeric(state.gameTime),
    };
    const result = await askClaude(safe);
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(result));
  });
}

const server = http.createServer((req, res) => {
  if (req.method === "POST" && req.url === "/ai-strategy") {
    handleStrategy(req, res);
    return;
  }
  serveStatic(req, res);
});

server.listen(PORT, () => {
  console.log(`Times 2 running on http://localhost:${PORT}`);
});
