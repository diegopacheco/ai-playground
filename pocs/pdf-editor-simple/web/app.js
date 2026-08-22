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
  undo: document.getElementById("undo"),
  save: document.getElementById("save"),
};

let state = { name: "", pages: [], canUndo: false };
let selected = new Set();
let target = "open";
let dragged = null;

async function call(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (payload.error) {
    say(payload.error, true);
    return;
  }
  state = payload;
  selected = new Set([...selected].filter((uid) => state.pages.some((page) => page.uid === uid)));
  draw();
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
  say.timer = setTimeout(() => message.classList.remove("on"), 2600);
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

call("/state");
