import * as Scene from "./scene.js";

(() => {
  const Sounds = window.Sounds;
  const W = 960, H = 540;
  const COST = { villager: 50, soldier: 80 };
  const TRAIN_TIME = { villager: 4, soldier: 5 };
  const TC_MAX = 520;
  const VILLAGER = { hp: 40, speed: 58, carry: 10, gatherTime: 1.8 };
  const SOLDIER = { hp: 60, speed: 74, dmg: 10, range: 26, interval: 0.8, engage: 150 };
  const DEFEND_LEASH = 230;
  const STRATEGY_INTERVAL = 6;
  const MAX_QUEUE = 6;

  const canvas = document.getElementById("board");

  let game, last, rafId, strategyTimer;
  const strategyInFlight = { me: false, ai: false };

  function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
  function enemyOf(owner) { return owner === "me" ? "ai" : "me"; }
  function ownTc(owner) { return game.tc[owner]; }
  function enemyTc(owner) { return game.tc[enemyOf(owner)]; }

  function spawnUnit(owner, type, x, y) {
    game.units.push({
      id: game.uid++, owner, type, x, y,
      hp: type === "villager" ? VILLAGER.hp : SOLDIER.hp,
      maxHp: type === "villager" ? VILLAGER.hp : SOLDIER.hp,
      state: "idle", carry: 0, gatherT: 0, attackCd: 0, mine: null,
    });
  }

  function newGame(mode) {
    game = {
      units: [], uid: 0,
      tc: {
        me: { owner: "me", x: 120, y: 430, hp: TC_MAX, maxHp: TC_MAX },
        ai: { owner: "ai", x: 840, y: 110, hp: TC_MAX, maxHp: TC_MAX },
      },
      mines: [
        { x: 220, y: 470, amount: 1600 },
        { x: 90, y: 320, amount: 1600 },
        { x: 740, y: 70, amount: 1600 },
        { x: 870, y: 230, amount: 1600 },
        { x: 480, y: 270, amount: 2400 },
      ],
      gold: { me: 120, ai: 120 },
      queue: { me: [], ai: [] },
      order: { me: { mode: "defend", rx: 0, ry: 0 }, ai: { mode: "defend", rx: 0, ry: 0 } },
      aiBuild: { me: "villager", ai: "villager" },
      controller: { me: mode === "cvc" ? "ai" : "human", ai: "ai" },
      mode, time: 0, over: false, winner: null,
    };
    for (let i = 0; i < 3; i++) spawnUnit("me", "villager", 160 + i * 14, 440 + i * 8);
    for (let i = 0; i < 3; i++) spawnUnit("ai", "villager", 800 - i * 14, 120 + i * 8);
  }

  function count(owner, type) {
    let n = 0;
    for (const u of game.units) if (u.owner === owner && u.type === type) n++;
    return n;
  }

  function nearestMine(u) {
    let best = null, bd = Infinity;
    for (const m of game.mines) {
      if (m.amount <= 0) continue;
      const d = dist(u, m);
      if (d < bd) { bd = d; best = m; }
    }
    return best;
  }

  function nearestEnemyUnit(u, range, anchor, leash) {
    let best = null, bd = range;
    for (const e of game.units) {
      if (e.owner === u.owner) continue;
      if (anchor && dist(e, anchor) > leash) continue;
      const d = dist(u, e);
      if (d < bd) { bd = d; best = e; }
    }
    return best;
  }

  function moveToward(u, tx, ty, speed, dt) {
    const dx = tx - u.x, dy = ty - u.y;
    const d = Math.hypot(dx, dy);
    if (d < 1) return 0;
    const step = Math.min(d, speed * dt);
    u.x += (dx / d) * step;
    u.y += (dy / d) * step;
    return d - step;
  }

  function updateVillager(u, dt) {
    if (u.state === "returning") {
      const tc = ownTc(u.owner);
      const left = moveToward(u, tc.x, tc.y, VILLAGER.speed, dt);
      if (left < 26) {
        game.gold[u.owner] += u.carry;
        u.carry = 0;
        u.state = "idle";
        if (u.owner === "me" && game.controller.me === "human") Sounds.play("coin");
      }
      return;
    }
    if (!u.mine || u.mine.amount <= 0) { u.mine = nearestMine(u); u.state = "idle"; }
    if (!u.mine) return;
    const left = moveToward(u, u.mine.x, u.mine.y, VILLAGER.speed, dt);
    if (left < 22) {
      u.state = "gathering";
      u.gatherT += dt;
      if (u.gatherT >= VILLAGER.gatherTime) {
        u.gatherT = 0;
        const taken = Math.min(VILLAGER.carry, u.mine.amount);
        u.mine.amount -= taken;
        u.carry = taken;
        u.state = "returning";
      }
    } else {
      u.state = "moving";
    }
  }

  function updateSoldier(u, dt) {
    u.attackCd = Math.max(0, u.attackCd - dt);
    const order = game.order[u.owner];
    const own = ownTc(u.owner);
    let target = null;

    if (order.mode === "defend") {
      target = nearestEnemyUnit(u, SOLDIER.engage, own, DEFEND_LEASH);
      if (!target) {
        const eTc = enemyTc(u.owner);
        const ax = own.x + (eTc.x - own.x) * 0.18 + (u.id % 5) * 14 - 28;
        const ay = own.y + (eTc.y - own.y) * 0.18 + ((u.id % 3) - 1) * 16;
        moveToward(u, ax, ay, SOLDIER.speed, dt);
        return;
      }
    } else {
      target = nearestEnemyUnit(u, SOLDIER.engage);
    }

    if (target) {
      const d = moveToward(u, target.x, target.y, SOLDIER.speed, dt);
      if (d <= SOLDIER.range && u.attackCd <= 0) {
        target.hp -= SOLDIER.dmg;
        u.attackCd = SOLDIER.interval;
        if (u.owner === "me" && game.controller.me === "human") Sounds.play("hit");
      }
      return;
    }

    if (order.mode === "rally") {
      moveToward(u, order.rx, order.ry, SOLDIER.speed, dt);
      return;
    }

    const eTc = enemyTc(u.owner);
    const d = Math.hypot(eTc.x - u.x, eTc.y - u.y);
    if (d <= SOLDIER.range + 26) {
      if (u.attackCd <= 0) {
        eTc.hp -= SOLDIER.dmg;
        u.attackCd = SOLDIER.interval;
      }
    } else {
      moveToward(u, eTc.x, eTc.y, SOLDIER.speed, dt);
    }
  }

  function updateProduction(owner, dt) {
    const q = game.queue[owner];
    if (!q.length) return;
    q[0].timeLeft -= dt;
    if (q[0].timeLeft <= 0) {
      const item = q.shift();
      const tc = ownTc(owner);
      const a = Math.random() * Math.PI * 2;
      spawnUnit(owner, item.type, tc.x + Math.cos(a) * 46, tc.y + Math.sin(a) * 46);
      if (owner === "me" && game.controller.me === "human") Sounds.play("train");
    }
  }

  function enqueue(owner, type) {
    if (game.over) return false;
    if (game.queue[owner].length >= MAX_QUEUE) return false;
    if (game.gold[owner] < COST[type]) return false;
    game.gold[owner] -= COST[type];
    game.queue[owner].push({ type, timeLeft: TRAIN_TIME[type] });
    return true;
  }

  function aiAutoProduce(owner) {
    if (game.queue[owner].length > 0) return;
    const want = game.aiBuild[owner] === "soldier" ? "soldier" : "villager";
    if (game.gold[owner] >= COST[want]) enqueue(owner, want);
    else if (want === "soldier" && game.gold[owner] >= COST.villager && count(owner, "villager") < 3) {
      enqueue(owner, "villager");
    }
  }

  function checkOver() {
    if (game.over) return;
    if (game.tc.ai.hp <= 0) { game.over = true; game.winner = "me"; endGame(); }
    else if (game.tc.me.hp <= 0) { game.over = true; game.winner = "ai"; endGame(); }
  }

  function step(dt) {
    if (game.over) return;
    game.time += dt;
    for (const u of game.units) {
      if (u.type === "villager") updateVillager(u, dt);
      else updateSoldier(u, dt);
    }
    game.units = game.units.filter((u) => u.hp > 0);
    updateProduction("me", dt);
    updateProduction("ai", dt);
    if (game.controller.me === "ai") aiAutoProduce("me");
    aiAutoProduce("ai");
    checkOver();
  }

  function syncHud() {
    document.getElementById("gold").textContent = Math.floor(game.gold.me);
    document.getElementById("villagers").textContent = count("me", "villager");
    document.getElementById("soldiers").textContent = count("me", "soldier");
    document.getElementById("e-villagers").textContent = count("ai", "villager");
    document.getElementById("e-soldiers").textContent = count("ai", "soldier");
    const t = Math.floor(game.time);
    document.getElementById("clock").textContent =
      Math.floor(t / 60) + ":" + String(t % 60).padStart(2, "0");
    const human = game.controller.me === "human";
    document.getElementById("train-villager").disabled = !human || game.over || game.gold.me < COST.villager;
    document.getElementById("train-soldier").disabled = !human || game.over || game.gold.me < COST.soldier;
    document.getElementById("attack").disabled = !human || game.over;
    document.getElementById("recall").disabled = !human || game.over;
  }

  function loop(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    step(dt);
    Scene.update(game);
    syncHud();
    if (!game.over) rafId = requestAnimationFrame(loop);
  }

  function logCouncil(owner, d) {
    const ul = document.getElementById("council-log");
    const li = document.createElement("li");
    if (d.source === "fallback") li.className = "fallback";
    const t = Math.floor(game.time);
    const clock = Math.floor(t / 60) + ":" + String(t % 60).padStart(2, "0");
    const side = owner === "me" ? "BLUE" : "RED";
    const verb = d.attack ? "ATTACK" : "build " + d.build;
    li.innerHTML =
      `<span class="t">${clock}</span><span class="badge">${side}: ${verb}</span> &mdash; ${d.reason || ""} ` +
      `<span class="t">[${d.source}]</span>`;
    ul.prepend(li);
    while (ul.children.length > 16) ul.removeChild(ul.lastChild);
  }

  async function fetchStrategy(owner) {
    if (strategyInFlight[owner] || game.over) return;
    strategyInFlight[owner] = true;
    const foe = enemyOf(owner);
    const payload = {
      gold: Math.floor(game.gold[owner]),
      villagers: count(owner, "villager"),
      soldiers: count(owner, "soldier"),
      enemyVillagers: count(foe, "villager"),
      enemySoldiers: count(foe, "soldier"),
      myTcHp: Math.floor(game.tc[owner].hp),
      enemyTcHp: Math.floor(game.tc[foe].hp),
      gameTime: Math.floor(game.time),
    };
    try {
      const res = await fetch("/ai-strategy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const d = await res.json();
      if (game.over) return;
      game.aiBuild[owner] = d.build;
      game.order[owner].mode = d.attack ? "attack" : "defend";
      logCouncil(owner, d);
    } catch (e) {
      game.aiBuild[owner] = count(owner, "villager") < 4 ? "villager" : "soldier";
    } finally {
      strategyInFlight[owner] = false;
    }
  }

  function tickStrategy() {
    if (game.controller.me === "ai") fetchStrategy("me");
    fetchStrategy("ai");
  }

  function endGame() {
    cancelAnimationFrame(rafId);
    clearInterval(strategyTimer);
    Scene.update(game);
    syncHud();
    const win = game.winner === "me";
    let title, text;
    if (game.mode === "cvc") {
      title = win ? "Blue Empire wins" : "Red Empire wins";
      text = "Sonnet " + (win ? "BLUE" : "RED") + " razed the rival Town Center.";
    } else {
      title = win ? "Victory!" : "Defeat";
      text = win
        ? "Sonnet's Town Center has fallen. The realm is yours."
        : "Your Town Center has fallen to Sonnet's army.";
    }
    Sounds.play(game.mode === "cvc" || win ? "victory" : "defeat");
    showOverlay(title, text, [{ label: "New battle", fn: showIntro }]);
  }

  function showOverlay(title, text, buttons) {
    document.getElementById("overlay-title").innerHTML = title;
    document.getElementById("overlay-text").textContent = text;
    const box = document.getElementById("overlay-buttons");
    box.innerHTML = "";
    buttons.forEach((b) => {
      const el = document.createElement("button");
      el.textContent = b.label;
      if (b.alt) el.className = "alt";
      el.addEventListener("click", b.fn);
      box.appendChild(el);
    });
    document.getElementById("overlay").classList.remove("hidden");
  }

  function showIntro() {
    cancelAnimationFrame(rafId);
    clearInterval(strategyTimer);
    showOverlay("Times&nbsp;2", "Choose your battle", [
      { label: "Human vs Sonnet", fn: () => start("pvc") },
      { label: "Sonnet vs Sonnet", fn: () => start("cvc"), alt: true },
    ]);
  }

  function start(mode) {
    newGame(mode);
    Scene.newMatch(game);
    document.getElementById("me-label").textContent = mode === "cvc" ? "Sonnet Blue" : "You";
    document.getElementById("ai-label").textContent = mode === "cvc" ? "Sonnet Red" : "Sonnet";
    document.getElementById("overlay").classList.add("hidden");
    document.getElementById("council-log").innerHTML = "";
    last = performance.now();
    clearInterval(strategyTimer);
    cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(loop);
    setTimeout(tickStrategy, 700);
    strategyTimer = setInterval(tickStrategy, STRATEGY_INTERVAL * 1000);
  }

  document.getElementById("train-villager").addEventListener("click", () => enqueue("me", "villager"));
  document.getElementById("train-soldier").addEventListener("click", () => enqueue("me", "soldier"));
  document.getElementById("attack").addEventListener("click", () => {
    if (game.controller.me !== "human") return;
    game.order.me.mode = "attack";
    Sounds.play("attack");
  });
  document.getElementById("recall").addEventListener("click", () => {
    if (game.controller.me !== "human") return;
    game.order.me.mode = "defend";
  });
  document.getElementById("sound").addEventListener("change", (e) => Sounds.setEnabled(e.target.checked));

  canvas.addEventListener("click", (e) => {
    if (game.over || game.controller.me !== "human") return;
    const p = Scene.screenToWorld(e.clientX, e.clientY);
    if (!p) return;
    game.order.me.mode = "rally";
    game.order.me.rx = Math.max(0, Math.min(W, p.x));
    game.order.me.ry = Math.max(0, Math.min(H, p.y));
    Sounds.play("attack");
  });

  Scene.init(canvas);
  newGame("pvc");
  Scene.newMatch(game);
  Scene.update(game);
  showIntro();
})();
