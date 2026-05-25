const themes = {
  algorithms: {
    title: "Algorithms",
    flavor: "A clockwork maze where every step has a cost.",
    boss: "Complexity Hydra",
    questions: [
      ["Which complexity usually describes binary search on a sorted array?", ["O(n)", "O(log n)", "O(n log n)", "O(1)"], 1, "Binary search cuts the remaining space each step.", "Hint Token"],
      ["A recursive function needs what to stop safely?", ["A base case", "A larger input", "A global variable", "A sorted array"], 0, "Recursion needs a stopping condition.", "Shield"],
      ["Which sorting algorithm has average O(n log n) behavior?", ["Bubble sort", "Insertion sort", "Merge sort", "Linear scan"], 2, "Divide and merge is the core pattern.", "Double XP"],
      ["What does a greedy algorithm do at each step?", ["Tries every subset", "Chooses a local best move", "Stores all previous answers", "Ignores constraints"], 1, "Greedy methods commit to a strong local choice.", "Retry Charm"],
      ["Dynamic programming is strongest when a problem has what?", ["Random inputs only", "No repeated work", "Overlapping subproblems", "Only one valid path"], 2, "DP reuses answers for repeated subproblems.", "Boss Key"]
    ],
    bossQuestions: [
      ["A graph search must avoid revisiting nodes to prevent what?", ["Cleaner names", "Cycles causing repeated work", "Smaller variables", "Sorted edges"], 1, "Visited sets prevent repeated traversal."],
      ["For shortest path with nonnegative edge weights, which algorithm fits best?", ["Dijkstra", "Quicksort", "Binary search", "Reservoir sampling"], 0, "Nonnegative weighted paths fit Dijkstra."],
      ["What is the main tradeoff of memoization?", ["Less memory for more CPU", "More memory for less repeated work", "No storage and no speed", "Only works on arrays"], 1, "Saved answers cost memory."],
      ["If an O(n) pass is enough, what should you avoid?", ["A single loop", "Extra O(n log n) sorting", "Reading input", "Comparing values"], 1, "Sorting can dominate the runtime."]
    ]
  },
  datastructures: {
    title: "Data Structures",
    flavor: "A vault of shapes, pointers, buckets, and queues.",
    boss: "Hash Collision Beast",
    questions: [
      ["Which structure is LIFO?", ["Queue", "Stack", "Hash map", "Tree"], 1, "The last item added leaves first.", "Hint Token"],
      ["Which structure is FIFO?", ["Stack", "Heap", "Queue", "Graph"], 2, "The first item added leaves first.", "Shield"],
      ["What does a hash map optimize for on average?", ["Lookup by key", "Sorted traversal", "Recursive calls", "Matrix rotation"], 0, "Keys map directly to buckets.", "Double XP"],
      ["Which structure is natural for hierarchical data?", ["Array", "Tree", "Queue", "Stack"], 1, "Parents and children form hierarchy.", "Retry Charm"],
      ["A heap is commonly used to implement what?", ["Priority queue", "Linked list", "Hash collision chain", "Adjacency matrix"], 0, "The top item has strongest priority.", "Boss Key"]
    ],
    bossQuestions: [
      ["A linked list makes which operation cheap when the node is known?", ["Insert near that node", "Binary search", "Random access", "Sorting by default"], 0, "Pointers can reconnect nearby nodes."],
      ["A graph adjacency list is usually better when edges are what?", ["Sparse", "All possible", "Sorted only", "Missing weights"], 0, "Sparse graphs avoid huge empty tables."],
      ["Which issue can hurt hash map performance?", ["Too many collisions", "Too many base cases", "Balanced tree height", "FIFO order"], 0, "Collisions put many keys together."],
      ["Which traversal visits tree levels from top to bottom?", ["DFS", "BFS", "Quicksort", "Hashing"], 1, "A queue drives level order."]
    ]
  },
  "generative-ai": {
    title: "Generative AI",
    flavor: "A live lab of prompts, tools, retrieval, and eval gates.",
    boss: "Hallucination Phantom",
    questions: [
      ["What improves retrieval-augmented answers most?", ["Relevant context", "Longer random text", "Hidden prompts only", "More colors"], 0, "The model can only ground on useful context.", "Hint Token"],
      ["A tool-using agent should do what before a risky action?", ["Explain intent", "Hide the command", "Skip validation", "Ignore output"], 0, "The user should understand the action.", "Shield"],
      ["What is hallucination?", ["A true citation", "Confident unsupported output", "A token counter", "A file format"], 1, "It sounds certain but lacks grounding.", "Double XP"],
      ["What should an eval measure?", ["Only style", "Task success against criteria", "Screen brightness", "Prompt length alone"], 1, "Evaluation needs a target behavior.", "Retry Charm"],
      ["What limits how much context a model can use at once?", ["Context window", "File name", "Terminal width", "Branch name"], 0, "The model has a bounded input budget.", "Boss Key"]
    ],
    bossQuestions: [
      ["A strong system instruction should be what?", ["Clear and scoped", "Contradictory", "Hidden in output", "Unrelated to task"], 0, "Instruction hierarchy works best when clear."],
      ["When retrieval returns weak matches, what should the agent do?", ["State uncertainty", "Invent citations", "Ignore the user", "Delete files"], 0, "Weak evidence should lower confidence."],
      ["Which pattern helps multi-step agent work?", ["Plan, act, observe", "Guess, hide, finish", "Repeat the same command", "Avoid feedback"], 0, "Observation improves control."],
      ["Why keep prompts concise?", ["To reduce useful signal", "To improve signal density", "To hide requirements", "To break tools"], 1, "Concise context reduces noise."]
    ]
  },
  "machine-learning": {
    title: "Machine Learning",
    flavor: "A training arena where metrics decide the winner.",
    boss: "Overfit Dragon",
    questions: [
      ["What is overfitting?", ["Training too little", "Memorizing training data too closely", "Using validation data", "Cleaning features"], 1, "The model performs poorly outside training data.", "Hint Token"],
      ["Which split estimates generalization during development?", ["Training", "Validation", "Source", "Binary"], 1, "Validation checks behavior away from training.", "Shield"],
      ["Precision focuses on what?", ["Correct positive predictions", "All real positives found", "File size", "Learning rate only"], 0, "Precision asks how many predicted positives were right.", "Double XP"],
      ["Recall focuses on what?", ["All real positives found", "Only true negatives", "Code formatting", "Batch names"], 0, "Recall asks how many actual positives were found.", "Retry Charm"],
      ["Which metric fits imbalanced classification better than accuracy alone?", ["F1", "Line count", "CPU brand", "Disk path"], 0, "F1 balances precision and recall.", "Boss Key"]
    ],
    bossQuestions: [
      ["A feature should usually be derived from what?", ["Information available at prediction time", "Future labels", "Random secrets", "Test answers"], 0, "Future data leaks into training."],
      ["Why use a test set once at the end?", ["To tune forever", "To estimate final generalization", "To store logs", "To rename columns"], 1, "The final set should stay untouched."],
      ["What does regularization usually reduce?", ["Overfitting", "Input rows", "Labels", "All metrics"], 0, "It constrains the model."],
      ["A confusion matrix shows what?", ["Prediction counts by class", "Only training loss", "Disk use", "Prompt tokens"], 0, "It breaks outcomes into classes."]
    ]
  },
  "sre-devops": {
    title: "SRE/DevOps",
    flavor: "An incident bridge with alarms, rollouts, and recovery paths.",
    boss: "Pager Storm",
    questions: [
      ["What is an SLO?", ["A reliability target", "A shell alias", "A container image", "A log color"], 0, "It sets expected service behavior.", "Hint Token"],
      ["What should observability help answer?", ["What changed and why", "Only file names", "Only team titles", "Nothing actionable"], 0, "Signals should guide diagnosis.", "Shield"],
      ["What does CrashLoopBackOff usually mean?", ["A pod repeatedly starts and fails", "A sorted queue", "A healthy rollout", "A completed batch"], 0, "The process exits after starting.", "Double XP"],
      ["Why keep deployments small?", ["To reduce blast radius", "To hide changes", "To skip tests", "To increase outage scope"], 0, "Small changes are easier to reason about.", "Retry Charm"],
      ["What is a runbook for?", ["Repeatable operational response", "Random brainstorming", "Replacing monitoring", "Deleting history"], 0, "It guides response under pressure.", "Boss Key"]
    ],
    bossQuestions: [
      ["During an incident, what comes first?", ["Stabilize user impact", "Rewrite everything", "Assign blame", "Ignore alerts"], 0, "Protect the service first."],
      ["Which signal tracks request failure rate?", ["Errors", "Saturation only", "Disk names", "Branch count"], 0, "Failures are error signals."],
      ["What should a rollback do?", ["Return to a known good version", "Add untested scope", "Remove monitoring", "Hide symptoms"], 0, "Known good beats uncertain new."],
      ["Why use health checks?", ["Detect service readiness and liveness", "Style shell output", "Sort logs", "Name pods"], 0, "Health checks feed automation."]
    ]
  },
  management: {
    title: "Management",
    flavor: "A planning war room where clarity defeats chaos.",
    boss: "Scope Creep Titan",
    questions: [
      ["Good prioritization starts with what?", ["Clear goals and tradeoffs", "More meetings only", "Hidden criteria", "Random order"], 0, "Priority needs a decision frame.", "Hint Token"],
      ["Effective feedback should be what?", ["Specific and timely", "Vague and delayed", "Only public", "Always avoided"], 0, "Useful feedback names behavior and timing.", "Shield"],
      ["Delegation should include what?", ["Desired outcome and constraints", "Only task title", "No ownership", "No context"], 0, "People need the target and boundaries.", "Double XP"],
      ["Stakeholder alignment reduces what?", ["Surprise and rework", "Learning", "Team trust", "Clear scope"], 0, "Alignment catches mismatch early.", "Retry Charm"],
      ["A healthy plan should include what?", ["Risks and dependencies", "Only dates", "No owners", "Hidden assumptions"], 0, "Plans need the parts that can break.", "Boss Key"]
    ],
    bossQuestions: [
      ["When priorities conflict, what helps most?", ["Explicit tradeoff decision", "More hidden work", "Ignoring one group", "Changing nothing"], 0, "Tradeoffs need visibility."],
      ["In conflict resolution, what should a manager seek first?", ["Shared facts and interests", "A quick winner", "Silence", "Longer status reports"], 0, "Facts and interests reduce heat."],
      ["What is a useful one-on-one pattern?", ["Listen, clarify, follow through", "Talk only about status", "Cancel often", "Avoid hard topics"], 0, "Trust depends on follow-through."],
      ["What should hiring evaluate?", ["Role needs and evidence", "Vibes only", "Speed only", "One opinion"], 0, "Evidence anchors the decision."]
    ]
  }
};

const aliases = {
  "1": "algorithms",
  algorithm: "algorithms",
  algorithms: "algorithms",
  algo: "algorithms",
  "2": "datastructures",
  datastructures: "datastructures",
  "data-structures": "datastructures",
  "data structures": "datastructures",
  ds: "datastructures",
  "3": "generative-ai",
  "generative-ai": "generative-ai",
  "generative ai": "generative-ai",
  genai: "generative-ai",
  ai: "generative-ai",
  "4": "machine-learning",
  "machine-learning": "machine-learning",
  "machine learning": "machine-learning",
  ml: "machine-learning",
  "5": "sre-devops",
  "sre-devops": "sre-devops",
  "sre/devops": "sre-devops",
  sre: "sre-devops",
  devops: "sre-devops",
  "6": "management",
  management: "management",
  manager: "management",
  leadership: "management"
};

let state = null;

const el = id => document.getElementById(id);

function historyKey(theme) {
  return `terminal-rpg-agent:${theme}:history`;
}

function getHistory(theme) {
  try {
    return JSON.parse(localStorage.getItem(historyKey(theme)) || "[]");
  } catch {
    return [];
  }
}

function saveHistory() {
  const history = getHistory(state.theme);
  history.unshift({
    score: state.score,
    level: state.level,
    xp: state.xp,
    bossDefeated: state.bossDefeated,
    finishedAt: new Date().toISOString()
  });
  localStorage.setItem(historyKey(state.theme), JSON.stringify(history.slice(0, 8)));
}

function startGame(theme) {
  const picked = themes[theme];
  state = {
    theme,
    data: picked,
    hp: 100,
    maxHp: 100,
    xp: 0,
    level: 1,
    score: 0,
    streak: 0,
    questIndex: 0,
    bossIndex: 0,
    bossHp: 100,
    bossUnlocked: false,
    bossDefeated: false,
    finished: false,
    inventory: {
      "Hint Token": 1,
      Shield: 1,
      "Double XP": 0,
      "Retry Charm": 1,
      "Boss Key": 0
    }
  };
  el("hero").classList.add("hidden");
  el("game").classList.remove("hidden");
  el("realmName").textContent = `${picked.title} Realm`;
  el("playerTitle").textContent = "Level 1 Adventurer";
  el("realmFlavor").textContent = picked.flavor;
  el("bossName").textContent = picked.boss;
  el("bossFlavor").textContent = "Clear five quests, then break the boss health bar.";
  log(`New ${picked.title} run started. Previous completed runs: ${getHistory(theme).length}.`);
  render();
}

function currentQuest() {
  if (state.bossUnlocked) {
    return state.data.bossQuestions[state.bossIndex];
  }
  return state.data.questions[state.questIndex];
}

function addItem(name) {
  state.inventory[name] = (state.inventory[name] || 0) + 1;
}

function gainXp(amount) {
  let gained = amount;
  if (state.inventory["Double XP"] > 0) {
    state.inventory["Double XP"] -= 1;
    gained *= 2;
  }
  state.xp += gained;
  state.level = Math.floor(state.xp / 100) + 1;
  return gained;
}

function answer(index) {
  if (!state || state.finished) {
    return;
  }
  const quest = currentQuest();
  const correct = quest[2] === index;
  if (correct) {
    state.streak += 1;
    if (state.bossUnlocked) {
      state.bossHp = Math.max(0, state.bossHp - 25);
      const gained = gainXp(40);
      state.score += 150 + state.streak * 25;
      pulse();
      log(`Direct hit. ${state.data.boss} loses 25 HP. XP gained: ${gained}.`);
      state.bossIndex += 1;
      if (state.bossHp === 0) {
        state.bossDefeated = true;
        state.finished = true;
        state.score += 500;
        gainXp(100);
        saveHistory();
        log(`Victory. ${state.data.boss} defeated. Final score: ${state.score}.`);
      }
    } else {
      const reward = quest[4];
      addItem(reward);
      const gained = gainXp(25);
      state.score += 100 + state.streak * 25;
      state.questIndex += 1;
      pulse();
      log(`Quest cleared. Reward gained: ${reward}. XP gained: ${gained}.`);
      if (state.questIndex >= state.data.questions.length) {
        state.bossUnlocked = true;
        log(`Boss gate opened. ${state.data.boss} enters the arena.`);
      }
    }
  } else {
    fail(quest[2]);
  }
  render();
}

function fail(correctIndex) {
  if (state.inventory["Retry Charm"] > 0) {
    state.inventory["Retry Charm"] -= 1;
    log("Retry Charm blocked the miss. Choose again.");
    render();
    return;
  }
  if (state.inventory.Shield > 0) {
    state.inventory.Shield -= 1;
    state.streak = 0;
    shake();
    log("Shield absorbed the hit. Streak reset.");
    return;
  }
  state.streak = 0;
  state.hp = Math.max(0, state.hp - (state.bossUnlocked ? 20 : 10));
  shake();
  log(`Hit taken. Correct answer was ${correctIndex + 1}. HP is now ${state.hp}.`);
  if (state.hp === 0) {
    state.finished = true;
    saveHistory();
    log(`Defeat. ${state.data.boss} holds the gate. Final score: ${state.score}.`);
  }
}

function useHint() {
  if (!state || state.finished) {
    return;
  }
  if (state.inventory["Hint Token"] < 1) {
    log("No Hint Tokens left.");
    return;
  }
  state.inventory["Hint Token"] -= 1;
  log(`Hint: ${currentQuest()[3]}`);
  render();
}

function renderThemes() {
  el("themeGrid").innerHTML = Object.entries(themes).map(([key, theme], index) => `
    <button class="theme-card" type="button" data-theme="${key}">
      <p class="kicker">Realm ${index + 1}</p>
      <h2>${theme.title}</h2>
      <p>${theme.flavor}</p>
    </button>
  `).join("");
  document.querySelectorAll("[data-theme]").forEach(button => {
    button.addEventListener("click", () => startGame(button.dataset.theme));
  });
}

function render() {
  if (!state) {
    return;
  }
  const quest = currentQuest();
  el("playerTitle").textContent = `Level ${state.level} Adventurer`;
  el("hpValue").textContent = `${state.hp}/${state.maxHp}`;
  el("xpValue").textContent = `${state.xp}`;
  el("levelValue").textContent = state.level;
  el("scoreValue").textContent = state.score;
  el("streakValue").textContent = state.streak;
  el("historyValue").textContent = getHistory(state.theme).length;
  el("bossValue").textContent = state.bossUnlocked ? `${state.bossHp}/100` : "Locked";
  el("hpBar").style.width = `${state.hp}%`;
  el("xpBar").style.width = `${state.xp % 100}%`;
  el("bossBar").style.width = `${state.bossUnlocked ? state.bossHp : 0}%`;
  el("questType").textContent = state.bossUnlocked ? "Boss Fight" : "Quest";
  el("questCounter").textContent = state.bossUnlocked ? `${state.bossIndex + 1}/4` : `${state.questIndex + 1}/5`;
  el("questPrompt").textContent = state.finished ? "Run complete. Start a new run when ready." : quest[0];
  el("answers").innerHTML = state.finished ? "" : quest[1].map((choice, index) => `
    <button class="answer" type="button" data-answer="${index}">
      <strong>${index + 1}</strong>
      ${choice}
    </button>
  `).join("");
  document.querySelectorAll("[data-answer]").forEach(button => {
    button.addEventListener("click", () => answer(Number(button.dataset.answer)));
  });
  renderInventory();
  renderMap();
  renderHistory();
}

function renderInventory() {
  el("inventory").innerHTML = Object.entries(state.inventory).map(([name, count]) => `
    <div class="inventory-item">
      <span>${name}</span>
      <strong>${count}</strong>
    </div>
  `).join("");
}

function renderMap() {
  const nodes = ["Q1", "Q2", "Q3", "Q4", "Q5", "Boss"];
  el("mapStrip").innerHTML = nodes.map((node, index) => {
    let className = "node";
    if (!state.bossUnlocked && index === state.questIndex) {
      className += " active";
    }
    if (index < state.questIndex || state.bossUnlocked && index < 5) {
      className += " done";
    }
    if (state.bossUnlocked && index === 5) {
      className += " active";
    }
    if (state.bossDefeated && index === 5) {
      className += " done";
    }
    return `<div class="${className}">${node}</div>`;
  }).join("");
}

function renderHistory() {
  const history = getHistory(state.theme);
  if (history.length === 0) {
    el("historyList").innerHTML = "<p>No completed runs yet.</p>";
    return;
  }
  el("historyList").innerHTML = history.slice(0, 5).map(run => `
    <div class="history-entry">
      <span>LV ${run.level}</span>
      <span>${run.score}</span>
      <span>${run.bossDefeated ? "Victory" : "Defeat"}</span>
    </div>
  `).join("");
}

function log(message) {
  el("logCard").textContent = message;
}

function shake() {
  el("questCard").classList.remove("hit");
  void el("questCard").offsetWidth;
  el("questCard").classList.add("hit");
}

function pulse() {
  el("questCard").classList.remove("pulse");
  void el("questCard").offsetWidth;
  el("questCard").classList.add("pulse");
}

function routeTheme() {
  const params = new URLSearchParams(window.location.search);
  const raw = params.get("theme");
  if (!raw) {
    return;
  }
  const theme = aliases[raw.toLowerCase()];
  if (theme) {
    startGame(theme);
  }
}

el("hintButton").addEventListener("click", useHint);
el("restartButton").addEventListener("click", () => {
  if (state) {
    startGame(state.theme);
  }
});

renderThemes();
routeTheme();
