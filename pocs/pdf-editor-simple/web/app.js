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

async function openEditor(uid) {
  const page = state.pages.find((item) => item.uid === uid);
  if (!page) return;
  editing = uid;
  document.getElementById("grid").hidden = true;
  document.getElementById("empty").classList.remove("on");
  document.getElementById("editor").hidden = false;
  window.scrollTo(0, 0);
  document.getElementById("editing").textContent =
    "Page " + (state.pages.indexOf(page) + 1);
  await drawSheet();
}

async function drawSheet() {
  const page = state.pages.find((item) => item.uid === editing);
  const sheet = document.getElementById("sheet");
  const layer = document.getElementById("runs");
  layer.replaceChildren();

  const response = await fetch("/runs?uid=" + editing);
  const data = await response.json();
  if (data.error) return say(data.error, true);

  if (sheet.getAttribute("src") !== page.view) {
    await new Promise((done) => {
      sheet.onload = done;
      sheet.onerror = done;
      sheet.src = page.view;
    });
  }

  const scale = sheet.clientWidth / data.width;
  layer.replaceChildren(...data.runs.map((run) => box(run, scale, data.height)));
  const kept = data.runs.filter((run) => run.mode === "inplace").length;
  say(data.runs.length + " lines, " + kept + " keep their font when edited");
}

function box(run, scale, pageHeight) {
  const [left, bottom, right, top] = run.box;
  const node = document.createElement("div");
  node.className = "run " + run.mode;
  node.style.left = left * scale + "px";
  node.style.top = (pageHeight - top) * scale + "px";
  node.style.width = (right - left) * scale + "px";
  node.style.height = (top - bottom) * scale + "px";
  node.title = {
    inplace: "Edited in place, the original font is kept",
    replaced: "The original text is removed and redrawn in Helvetica",
    redraw: "The original text is covered and redrawn in Helvetica",
  }[run.mode];

  node.onclick = () => {
    if (node.querySelector("input")) return;
    const field = document.createElement("input");
    field.value = run.text;
    field.style.fontSize = Math.max(11, run.size * scale * 0.9) + "px";
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
        if (field.isConnected) {
          field.disabled = false;
          field.focus();
        }
        return;
      }
      say({
        inplace: "Line updated, the original font is kept",
        replaced: "Line updated, redrawn in Helvetica",
        redraw: "Line updated, covered and redrawn in Helvetica",
      }[(state.applied || [{}])[0].mode] || "Line updated");
    };
  };
  return node;
}

function closeEditor() {
  editing = null;
  document.getElementById("editor").hidden = true;
  document.getElementById("grid").hidden = false;
  draw();
}

buttons.edit.onclick = () => openEditor([...selected][0]);
document.getElementById("back").onclick = closeEditor;

call("/state");
