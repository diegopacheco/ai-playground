const drop = document.getElementById("drop");
const fileInput = document.getElementById("file");
const samplesEl = document.getElementById("samples");
const emptyEl = document.getElementById("empty");
const resultEl = document.getElementById("result");
const errorEl = document.getElementById("error");

function fmtSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 / 1024).toFixed(1) + " MB";
}

function esc(s) {
  return String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

async function loadSamples() {
  const res = await fetch("/api/samples");
  const items = await res.json();
  samplesEl.innerHTML = "";
  for (const it of items) {
    const li = document.createElement("li");
    li.className = "sample";
    li.innerHTML = `<span class="s-left"><span class="s-name">${esc(it.name)}</span>` +
      `<span class="s-label">${esc(it.label)}</span></span>` +
      `<span class="s-size">${fmtSize(it.size)}</span>`;
    li.onclick = () => explainSample(it.name);
    samplesEl.appendChild(li);
  }
}

async function explainSample(name) {
  const res = await fetch("/api/explain?sample=" + encodeURIComponent(name), { method: "POST" });
  render(await res.json(), res.ok);
}

async function explainFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("/api/explain", { method: "POST", body: fd });
  render(await res.json(), res.ok);
}

function render(data, ok) {
  emptyEl.classList.add("hidden");
  if (!ok || data.error) {
    resultEl.classList.add("hidden");
    errorEl.classList.remove("hidden");
    errorEl.textContent = "Could not explain " + (data.filename || "file") + ": " + (data.error || "unknown error");
    return;
  }
  errorEl.classList.add("hidden");
  resultEl.classList.remove("hidden");

  const head = document.getElementById("format-head");
  head.innerHTML =
    `<div class="fmt-badge">${esc(data.format_key)}</div>` +
    `<div><h2>${esc(data.format)}</h2>` +
    `<p class="tagline">${esc(data.tagline)}</p>` +
    `<p class="desc">${esc(data.description)}</p>` +
    `<p class="filename">&#9679; ${esc(data.filename)}</p></div>`;

  const info = document.getElementById("file-info");
  info.innerHTML = data.fileInfo.map(p =>
    `<div class="info-item"><div class="k">${esc(p.label)}</div><div class="v">${esc(p.value)}</div></div>`
  ).join("");

  document.getElementById("notes").innerHTML = data.notes.map(n => `<li>${esc(n)}</li>`).join("");

  document.getElementById("field-count").textContent = data.fields.length + " fields";
  document.getElementById("fields").innerHTML = data.fields.map(f => {
    const nullable = f.nullable
      ? `<span class="tag-null">nullable</span>`
      : `<span class="tag-req">required</span>`;
    return `<tr><td class="f-name">${esc(f.name)}</td>` +
      `<td class="f-type">${esc(f.type)}</td>` +
      `<td class="f-meaning">${esc(f.logicalType)}</td>` +
      `<td>${nullable}</td>` +
      `<td class="f-notes">${esc(f.notes || "")}</td></tr>`;
  }).join("");

  document.getElementById("raw").textContent = data.rawSchema;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

drop.onclick = () => fileInput.click();
fileInput.onchange = () => { if (fileInput.files[0]) explainFile(fileInput.files[0]); };

drop.ondragover = e => { e.preventDefault(); drop.classList.add("drag"); };
drop.ondragleave = () => drop.classList.remove("drag");
drop.ondrop = e => {
  e.preventDefault();
  drop.classList.remove("drag");
  if (e.dataTransfer.files[0]) explainFile(e.dataTransfer.files[0]);
};

loadSamples();
