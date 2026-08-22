const grid = document.getElementById("grid");
const picker = document.getElementById("picker");
const message = document.getElementById("message");
const buttons = {
  open: document.getElementById("open"),
  append: document.getElementById("append"),
  left: document.getElementById("left"),
  right: document.getElementById("right"),
  delete: document.getElementById("delete"),
  keep: document.getElementById("keep"),
  edit: document.getElementById("edit"),
  undo: document.getElementById("undo"),
  save: document.getElementById("save"),
};

let state = { name: "", pages: [], canUndo: false };
let selected = new Set();
let target = "open";
let dragged = null;
let editing = null;

async function call(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (payload.error) {
    say(payload.error, true);
    return false;
  }
  state = payload;
  selected = new Set([...selected].filter((uid) => state.pages.some((page) => page.uid === uid)));
  draw();
  if (editing !== null && state.pages.some((page) => page.uid === editing)) await drawSheet();
  return true;
}

const send = (op, extra) => call("/op", {
  method: "POST",
  body: JSON.stringify({ op, pages: [...selected], ...extra }),
});

const upload = (file) => call(target === "open" ? "/open" : "/add", {
  method: "POST",
  body: file,
  headers: { "X-Filename": file.name },
});

function draw() {
  document.getElementById("name").textContent = state.name;
  document.getElementById("count").textContent = state.pages.length
    ? `${state.pages.length} pages${selected.size ? `, ${selected.size} selected` : ""}`
    : "";
  document.getElementById("empty").classList.toggle("on", !state.pages.length);

  const some = selected.size > 0;
  buttons.append.disabled = !state.pages.length;
  buttons.left.disabled = buttons.right.disabled = buttons.delete.disabled = buttons.keep.disabled = !some;
  buttons.edit.disabled = selected.size !== 1;
  buttons.undo.disabled = !state.canUndo;
  buttons.save.disabled = !state.pages.length;

  grid.replaceChildren(...state.pages.map((page, index) => card(page, index)));
}

function card(page, index) {
  const node = document.createElement("div");
  node.className = "page" + (selected.has(page.uid) ? " selected" : "");
  node.draggable = true;
  node.dataset.uid = page.uid;
  node.innerHTML = `
    <div class="thumb r${page.rotation}"><img src="${page.thumb}" alt="page ${index + 1}"></div>
    <div class="label"><span>${index + 1}</span><span>${page.rotation ? page.rotation + "°" : ""}</span></div>`;

  node.onclick = (event) => {
    if (!event.shiftKey) {
      selected.has(page.uid) ? selected.delete(page.uid) : selected.add(page.uid);
    } else {
      const last = [...selected].pop();
      const from = state.pages.findIndex((other) => other.uid === last);
      const range = [Math.min(from, index), Math.max(from, index)];
      state.pages.slice(range[0], range[1] + 1).forEach((other) => selected.add(other.uid));
    }
    draw();
  };

  node.ondragstart = () => {
    dragged = page.uid;
    node.classList.add("dragging");
  };
  node.ondragend = () => {
    dragged = null;
    draw();
  };
  node.ondragover = (event) => {
    if (dragged === null || dragged === page.uid) return;
    event.preventDefault();
    const after = event.offsetX > node.offsetWidth / 2;
    node.classList.toggle("before", !after);
    node.classList.toggle("after", after);
  };
  node.ondragleave = () => node.classList.remove("before", "after");
  node.ondrop = (event) => {
    event.preventDefault();
    const after = event.offsetX > node.offsetWidth / 2;
    const order = state.pages.map((other) => other.uid).filter((uid) => uid !== dragged);
    order.splice(order.indexOf(page.uid) + (after ? 1 : 0), 0, dragged);
    dragged = null;
    send("reorder", { pages: order });
  };
  return node;
}

function say(text, bad) {
  message.textContent = text;
  message.classList.toggle("bad", Boolean(bad));
  message.classList.add("on");
  clearTimeout(say.timer);
  say.timer = setTimeout(() => message.classList.remove("on"), bad ? 9000 : 2600);
}

buttons.open.onclick = () => {
  target = "open";
  picker.click();
};
buttons.append.onclick = () => {
  target = "add";
  picker.click();
};
buttons.left.onclick = () => send("rotate", { angle: -90 });
buttons.right.onclick = () => send("rotate", { angle: 90 });
buttons.delete.onclick = () => send("delete");
buttons.keep.onclick = () => send("keep");
buttons.undo.onclick = () => send("undo");
buttons.save.onclick = () => {
  location.href = "/save";
  say("saved");
};

picker.onchange = () => {
  if (picker.files[0]) upload(picker.files[0]);
  picker.value = "";
};

document.ondragover = (event) => {
  event.preventDefault();
  if (event.dataTransfer.types.includes("Files")) document.body.classList.add("dropping");
};
document.ondragleave = (event) => {
  if (!event.relatedTarget) document.body.classList.remove("dropping");
};
document.ondrop = (event) => {
  event.preventDefault();
  document.body.classList.remove("dropping");
  const file = event.dataTransfer.files[0];
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    say(`${file.name} is not a PDF`, true);
    return;
  }
  target = state.pages.length ? "add" : "open";
  upload(file);
};

document.onkeydown = (event) => {
  if (editing !== null) {
    if (event.key === "Escape") closeEditor();
    return;
  }
  if (event.key === "Backspace" && selected.size) send("delete");
  if (event.key === "z" && (event.metaKey || event.ctrlKey)) send("undo");
  if (event.key === "a" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    state.pages.forEach((page) => selected.add(page.uid));
    draw();
  }
  if (event.key === "Escape") {
    selected.clear();
    draw();
  }
};

let tool = "select";
let sheetScale = 1;
let pageHeight = 0;

async function openEditor(uid) {
  const page = state.pages.find((item) => item.uid === uid);
  if (!page) return;
  editing = uid;
  document.getElementById("grid").hidden = true;
  document.getElementById("empty").classList.remove("on");
  document.getElementById("editor").hidden = false;
  window.scrollTo(0, 0);
  document.getElementById("editing").textContent = "Page " + (state.pages.indexOf(page) + 1);
  await drawSheet();
}

async function drawSheet() {
  const page = state.pages.find((item) => item.uid === editing);
  const sheet = document.getElementById("sheet");
  const layer = document.getElementById("runs");
  layer.replaceChildren();

  const data = await fetch("/runs?uid=" + editing).then((reply) => reply.json());
  if (data.error) return say(data.error, true);

  if (sheet.getAttribute("src") !== page.view) {
    await new Promise((done) => {
      sheet.onload = done;
      sheet.onerror = done;
      sheet.src = page.view;
    });
  }

  sheetScale = sheet.clientWidth / data.width;
  pageHeight = data.height;
  layer.replaceChildren(
    ...data.runs.map((run) => lineNode(run)),
    ...(page.notes || []).map((note) => noteNode(note)),
  );
  const kept = data.runs.filter((run) => run.mode === "inplace").length;
  say(data.runs.length + " lines, " + kept + " keep their font when edited");
}

const toPage = (pixels) => pixels / sheetScale;

function place(node, box) {
  const [left, bottom, right, top] = box;
  node.style.left = left * sheetScale + "px";
  node.style.top = (pageHeight - top) * sheetScale + "px";
  node.style.width = (right - left) * sheetScale + "px";
  node.style.height = (top - bottom) * sheetScale + "px";
}

function draggable(node, onDrop) {
  node.onpointerdown = (event) => {
    if (event.target.tagName === "INPUT" || tool !== "select") return;
    const from = { x: event.clientX, y: event.clientY };
    let moved = false;
    node.setPointerCapture(event.pointerId);
    const move = (step) => {
      const dx = step.clientX - from.x;
      const dy = step.clientY - from.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
      if (moved) {
        node.style.transform = `translate(${dx}px, ${dy}px)`;
        node.classList.add("moving");
      }
    };
    const up = async (step) => {
      node.onpointermove = null;
      node.onpointerup = null;
      node.classList.remove("moving");
      node.style.transform = "";
      if (!moved) return;
      node.dataset.moved = "1";
      await onDrop(toPage(step.clientX - from.x), -toPage(step.clientY - from.y));
    };
    node.onpointermove = move;
    node.onpointerup = up;
  };
}

function lineNode(run) {
  const node = document.createElement("div");
  node.className = "run " + run.mode;
  place(node, run.box);
  node.title = {
    inplace: "Edited in place, the original font is kept",
    replaced: "Redrawn in the same font",
    redraw: "Covered and redrawn",
  }[run.mode] + ". Drag to move it.";

  draggable(node, (dx, dy) => call("/move", {
    method: "POST",
    body: JSON.stringify({ page: editing, run: run.id, dx, dy }),
  }));

  node.onclick = () => {
    if (node.dataset.moved || tool !== "select" || node.querySelector("input")) {
      delete node.dataset.moved;
      return;
    }
    const field = document.createElement("input");
    field.value = run.text;
    field.style.fontSize = Math.max(11, run.size * sheetScale * 0.9) + "px";
    node.appendChild(field);
    field.focus();
    field.select();
    field.onkeydown = async (event) => {
      event.stopPropagation();
      if (event.key === "Escape") { field.remove(); return; }
      if (event.key !== "Enter") return;
      const value = field.value;
      if (value === run.text) { field.remove(); return; }
      field.disabled = true;
      const applied = await call("/text", {
        method: "POST",
        body: JSON.stringify({ page: editing, edits: { [run.id]: value } }),
      });
      if (!applied) {
        if (field.isConnected) { field.disabled = false; field.focus(); }
        return;
      }
      say({
        inplace: "Line updated, the original font is kept",
        replaced: "Line updated in the same font",
        redrawn: "Line updated, redrawn in Helvetica",
        redraw: "Line updated, covered and redrawn",
      }[(state.applied || [{}])[0].mode] || "Line updated");
    };
  };
  return node;
}

function noteNode(note) {
  const node = document.createElement("div");
  node.className = "note " + note.kind;
  place(node, [note.x, note.y, note.x + note.width, note.y + note.height]);
  const colour = note.color.map((channel) => Math.round(channel * 255)).join(",");

  if (note.kind === "highlight") {
    node.style.background = `rgba(${colour}, 0.42)`;
    node.title = "Highlight. Drag to move it, Backspace to remove it.";
  } else {
    node.style.color = `rgb(${colour})`;
    node.style.fontSize = note.size * sheetScale + "px";
    node.textContent = note.text;
    node.title = "Note. Click to retype, drag to move, Backspace to remove.";
    node.onclick = () => {
      if (node.dataset.moved) { delete node.dataset.moved; return; }
      writeInto(node, note);
    };
  }

  node.onpointerdown = (event) => event.stopPropagation();
  draggable(node, (dx, dy) => call("/note", {
    method: "POST",
    body: JSON.stringify({
      page: editing, action: "update", id: note.id,
      note: { x: note.x + dx, y: note.y + dy },
    }),
  }));
  return node;
}

function writeInto(node, note) {
  const field = document.createElement("input");
  field.value = note.text;
  field.style.fontSize = Math.max(12, note.size * sheetScale) + "px";
  node.textContent = "";
  node.appendChild(field);
  field.focus();
  field.onkeydown = async (event) => {
    event.stopPropagation();
    if (event.key === "Escape") { await drawSheet(); return; }
    if (event.key !== "Enter") return;
    await call("/note", {
      method: "POST",
      body: JSON.stringify({
        page: editing, action: note.id ? "update" : "add", id: note.id,
        note: { ...note, text: field.value },
      }),
    });
  };
}

function startSheetTool(event) {
  if (tool === "select" || event.target.closest(".note, .run input")) return;
  const layer = document.getElementById("runs");
  const bounds = layer.getBoundingClientRect();
  const from = { x: event.clientX - bounds.left, y: event.clientY - bounds.top };

  if (tool === "write") {
    const ghost = document.createElement("div");
    ghost.className = "note text";
    const size = 14;
    const note = {
      kind: "text", size, color: [0.85, 0.1, 0.1], text: "",
      x: toPage(from.x), y: pageHeight - toPage(from.y) - size, width: 200, height: size * 1.2,
    };
    place(ghost, [note.x, note.y, note.x + note.width, note.y + note.height]);
    ghost.style.color = "rgb(217,26,26)";
    layer.appendChild(ghost);
    writeInto(ghost, note);
    return;
  }

  const ghost = document.createElement("div");
  ghost.className = "note highlight";
  ghost.style.background = "rgba(255,235,59,0.42)";
  layer.appendChild(ghost);
  const move = (step) => {
    const x = Math.min(from.x, step.clientX - bounds.left);
    const y = Math.min(from.y, step.clientY - bounds.top);
    ghost.style.left = x + "px";
    ghost.style.top = y + "px";
    ghost.style.width = Math.abs(step.clientX - bounds.left - from.x) + "px";
    ghost.style.height = Math.abs(step.clientY - bounds.top - from.y) + "px";
  };
  const up = async (step) => {
    window.onpointermove = null;
    window.onpointerup = null;
    const width = toPage(Math.abs(step.clientX - bounds.left - from.x));
    const height = toPage(Math.abs(step.clientY - bounds.top - from.y));
    const left = toPage(Math.min(from.x, step.clientX - bounds.left));
    const bottom = pageHeight - toPage(Math.max(from.y, step.clientY - bounds.top));
    ghost.remove();
    if (width < 3 || height < 3) return;
    await call("/note", {
      method: "POST",
      body: JSON.stringify({
        page: editing, action: "add",
        note: { kind: "highlight", x: left, y: bottom, width, height,
                text: "", size: 12, color: [1, 0.92, 0.23] },
      }),
    });
  };
  window.onpointermove = move;
  window.onpointerup = up;
}

function pickTool(name) {
  tool = name;
  for (const button of document.querySelectorAll(".tool")) {
    button.classList.toggle("on", button.dataset.tool === name);
  }
  document.getElementById("runs").classList.toggle("drawing", name !== "select");
}

function closeEditor() {
  editing = null;
  document.getElementById("editor").hidden = true;
  document.getElementById("grid").hidden = false;
  draw();
}

buttons.edit.onclick = () => openEditor([...selected][0]);
document.getElementById("back").onclick = closeEditor;
document.getElementById("runs").onpointerdown = startSheetTool;
for (const button of document.querySelectorAll(".tool")) {
  button.onclick = () => pickTool(button.dataset.tool);
}

call("/state");
