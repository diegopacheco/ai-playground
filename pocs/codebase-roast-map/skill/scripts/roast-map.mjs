#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { execFileSync, spawnSync } from "node:child_process";

const mode = process.argv[2] || "report";
const repo = path.resolve(process.argv[3] || process.cwd());
const outDir = path.join(repo, ".roast-map");
const maxReadBytes = 900000;
const ignoreDirs = new Set([".git", "node_modules", "vendor", "target", "dist", "build", ".next", ".nuxt", ".cache", ".roast-map", "coverage"]);
const codeExts = new Set([".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".java", ".go", ".rs", ".py", ".rb", ".php", ".cs", ".c", ".cc", ".cpp", ".h", ".hpp", ".kt", ".kts", ".swift", ".scala", ".sh", ".bash", ".zsh", ".sql", ".lua", ".zig", ".ex", ".exs", ".erl", ".fs", ".fsx", ".clj", ".cljs", ".dart", ".vue", ".svelte"]);
const bugWords = /\b(fix|bug|issue|defect|crash|hotfix|broken|regression|fail|failure|flaky)\b/i;
const markerWords = /\b(TODO|FIXME|HACK)\b/g;

function git(args, fallback = "") {
  try {
    return execFileSync("git", args, { cwd: repo, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], maxBuffer: 80 * 1024 * 1024 }).trim();
  } catch {
    return fallback;
  }
}

function rel(file) {
  return path.relative(repo, file).split(path.sep).join("/");
}

function walk(dir, acc = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith(".") && entry.name !== ".github") {
      if (entry.name !== ".github") continue;
    }
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (!ignoreDirs.has(entry.name)) walk(full, acc);
    } else if (entry.isFile()) {
      acc.push(rel(full));
    }
  }
  return acc;
}

function files() {
  const listed = git(["ls-files"], "");
  if (listed) return listed.split(/\r?\n/).filter(Boolean);
  return walk(repo);
}

function safeRead(file) {
  const full = path.join(repo, file);
  try {
    const stat = fs.statSync(full);
    if (stat.size > maxReadBytes) return "";
    const raw = fs.readFileSync(full);
    if (raw.includes(0)) return "";
    return raw.toString("utf8");
  } catch {
    return "";
  }
}

function isCode(file) {
  return codeExts.has(path.extname(file).toLowerCase());
}

function isTest(file) {
  const lower = file.toLowerCase();
  if (/(^|\/)(__tests__|tests?|specs?)(\/|$)/.test(lower)) return true;
  if (/\.(test|spec)\.[a-z0-9]+$/i.test(file)) return true;
  if (/_test\.[a-z0-9]+$/i.test(file)) return true;
  if (lower.endsWith("test.java") || lower.endsWith("tests.java")) return true;
  return false;
}

function baseName(file) {
  return path.basename(file, path.extname(file)).replace(/\.(test|spec)$/i, "").replace(/_test$/i, "").toLowerCase();
}

function hasNearbyTest(file, allTests) {
  const name = baseName(file);
  const dir = path.dirname(file);
  return allTests.some((test) => {
    const testName = baseName(test);
    const testDir = path.dirname(test);
    return testName === name || testName.includes(name) || (testDir.includes(dir) && test.toLowerCase().includes(name));
  });
}

function nestingScore(lines) {
  let curly = 0;
  let maxCurly = 0;
  let maxIndent = 0;
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const indent = Math.floor((line.match(/^\s*/)?.[0].replace(/\t/g, "  ").length || 0) / 2);
    maxIndent = Math.max(maxIndent, indent);
    const closes = (line.match(/[}\])]/g) || []).length;
    const opens = (line.match(/[{[(]/g) || []).length;
    curly = Math.max(0, curly - closes);
    maxCurly = Math.max(maxCurly, curly + opens);
    curly += opens;
  }
  return Math.max(maxCurly, maxIndent);
}

function functionCount(text) {
  const patterns = [
    /\bfunction\s+[A-Za-z0-9_$]+\s*\(/g,
    /\b[A-Za-z0-9_$]+\s*=\s*(async\s*)?\([^)]*\)\s*=>/g,
    /\b(def|fn|func|fun|pub fn|void|int|long|double|float|boolean|char|String)\s+[A-Za-z0-9_$]+\s*\(/g,
    /\b(class|interface|struct|enum)\s+[A-Za-z0-9_$]+/g
  ];
  return patterns.reduce((sum, rx) => sum + (text.match(rx) || []).length, 0);
}

function historyMap(wantedFiles) {
  const wanted = new Set(wantedFiles);
  const map = new Map();
  for (const file of wantedFiles) map.set(file, { commits: 0, authors: new Set(), bugs: 0, lastDate: "" });
  const raw = git(["log", "--max-count=6000", "--date=short", "--format=__ROAST_COMMIT__%x09%H%x09%an%x09%ad%x09%s", "--name-only"], "");
  if (!raw) return new Map(wantedFiles.map((file) => [file, { commits: 0, authors: 0, bugs: 0, lastDate: "" }]));
  let current = null;
  for (const line of raw.split(/\r?\n/)) {
    if (!line) continue;
    if (line.startsWith("__ROAST_COMMIT__\t")) {
      const parts = line.split("\t");
      current = {
        author: parts[2] || "",
        date: parts[3] || "",
        bug: bugWords.test(parts.slice(4).join(" "))
      };
      continue;
    }
    if (!current || !wanted.has(line)) continue;
    const item = map.get(line);
    item.commits += 1;
    if (current.author) item.authors.add(current.author);
    if (!item.lastDate && current.date) item.lastDate = current.date;
    if (current.bug) item.bugs += 1;
  }
  for (const [file, item] of map.entries()) {
    map.set(file, { commits: item.commits, authors: item.authors.size, bugs: item.bugs, lastDate: item.lastDate });
  }
  return map;
}

function daysSince(dateText) {
  if (!dateText) return 0;
  const time = new Date(`${dateText}T00:00:00Z`).getTime();
  if (!Number.isFinite(time)) return 0;
  return Math.max(0, Math.floor((Date.now() - time) / 86400000));
}

function clamp(n, max) {
  return Math.max(0, Math.min(max, n));
}

function round(n) {
  return Math.round(n * 10) / 10;
}

function roastFor(file) {
  const top = file.scores;
  const worst = Object.entries(top).filter(([key]) => key !== "overall").sort((a, b) => b[1] - a[1])[0]?.[0] || "overall";
  if (worst === "tests") return "No nearby test coverage found. This block is walking around without a seatbelt.";
  if (worst === "churn") return "Changed a lot. This file has subscription-based drama.";
  if (worst === "complexity") return "Complexity is high. The control flow brought its own maze.";
  if (worst === "stale") return "Old and quiet. Could be stable, could be haunted legacy real estate.";
  if (worst === "ownership") return "Many hands touched it. Ownership looks like a group project at midnight.";
  if (worst === "bugs") return "Bug-fix history is loud. This path has been on fire before.";
  if (worst === "size") return "Large file. It may need a zoning permit.";
  return "Looks calm today. Keep an eye on the neighborhood.";
}

function analyze() {
  const all = files().filter((file) => isCode(file));
  const textByFile = new Map();
  const tests = [];
  for (const file of all) {
    const text = safeRead(file);
    textByFile.set(file, text);
    if (isTest(file)) tests.push(file);
  }
  const histories = historyMap(all);
  const analyzed = all.map((file) => {
    const text = textByFile.get(file) || "";
    const lines = text ? text.split(/\r?\n/) : [];
    const loc = lines.filter((line) => line.trim()).length;
    const nested = nestingScore(lines);
    const funcs = functionCount(text);
    const markers = (text.match(markerWords) || []).length;
    const test = isTest(file);
    const nearTest = test || hasNearbyTest(file, tests);
    const hist = histories.get(file) || { commits: 0, authors: 0, bugs: 0, lastDate: "" };
    const age = daysSince(hist.lastDate);
    const complexity = clamp(loc / 30 + nested * 2 + funcs * 0.8 + markers * 3, 25);
    const churn = clamp(hist.commits * 0.45, 25);
    const size = clamp(loc / 45, 15);
    const stale = loc > 40 ? clamp(age / 60, 15) : 0;
    const ownership = clamp(Math.max(0, hist.authors - 1) * 1.6, 15);
    const testsScore = !test && !nearTest && loc > 20 ? 15 : 0;
    const bugs = clamp(hist.bugs * 4, 20);
    const overall = clamp(complexity + churn + size + stale + ownership + testsScore + bugs, 100);
    return {
      path: file,
      dir: path.dirname(file) === "." ? "" : path.dirname(file),
      ext: path.extname(file).replace(".", "") || "file",
      loc,
      nesting: nested,
      functions: funcs,
      markers,
      isTest: test,
      hasTest: nearTest,
      commits: hist.commits,
      authors: hist.authors,
      bugs: hist.bugs,
      lastDate: hist.lastDate || "unknown",
      ageDays: age,
      scores: {
        overall: round(overall),
        complexity: round(complexity),
        churn: round(churn),
        size: round(size),
        stale: round(stale),
        ownership: round(ownership),
        tests: round(testsScore),
        bugs: round(bugs)
      }
    };
  });
  analyzed.sort((a, b) => b.scores.overall - a.scores.overall);
  for (const file of analyzed) file.roast = roastFor(file);
  const folders = folderScores(analyzed);
  return {
    repo,
    generatedAt: new Date().toISOString(),
    files: analyzed,
    folders,
    totals: {
      files: analyzed.length,
      folders: folders.length,
      tests: analyzed.filter((file) => file.isTest).length,
      risky: analyzed.filter((file) => file.scores.overall >= 60).length
    }
  };
}

function folderScores(files) {
  const map = new Map();
  for (const file of files) {
    const parts = file.path.split("/");
    for (let i = 1; i < parts.length; i += 1) {
      const folder = parts.slice(0, i).join("/");
      const item = map.get(folder) || { path: folder, files: 0, score: 0, loc: 0 };
      item.files += 1;
      item.score += file.scores.overall;
      item.loc += file.loc;
      map.set(folder, item);
    }
  }
  return [...map.values()].map((item) => ({ ...item, score: round(item.files ? item.score / item.files : 0) })).sort((a, b) => b.score - a.score);
}

function terminalReport(data) {
  const lines = [];
  lines.push(`Codebase Roast Map`);
  lines.push(`Repo: ${data.repo}`);
  lines.push(`Files scanned: ${data.totals.files}`);
  lines.push(`Risky files: ${data.totals.risky}`);
  lines.push("");
  lines.push("Hottest blocks:");
  for (const file of data.files.slice(0, 12)) {
    lines.push(`${String(file.scores.overall).padStart(5)}  ${file.path}`);
    lines.push(`       ${file.roast}`);
    lines.push(`       loc=${file.loc} commits=${file.commits} authors=${file.authors} bugs=${file.bugs} tests=${file.hasTest ? "nearby" : "missing"}`);
  }
  lines.push("");
  lines.push("District watchlist:");
  for (const folder of data.folders.slice(0, 8)) {
    lines.push(`${String(folder.score).padStart(5)}  ${folder.path}  files=${folder.files} loc=${folder.loc}`);
  }
  return lines.join(os.EOL);
}

function markdownSummary(data) {
  const lines = [];
  lines.push("# Codebase Roast Map");
  lines.push("");
  lines.push(`Repo: ${data.repo}`);
  lines.push(`Generated: ${data.generatedAt}`);
  lines.push(`Files scanned: ${data.totals.files}`);
  lines.push(`Risky files: ${data.totals.risky}`);
  lines.push("");
  lines.push("## Hottest Blocks");
  lines.push("");
  for (const file of data.files.slice(0, 20)) {
    lines.push(`- ${file.scores.overall} ${file.path}`);
    lines.push(`  ${file.roast}`);
  }
  lines.push("");
  lines.push("## District Watchlist");
  lines.push("");
  for (const folder of data.folders.slice(0, 12)) {
    lines.push(`- ${folder.score} ${folder.path} files=${folder.files} loc=${folder.loc}`);
  }
  return lines.join("\n");
}

function html(data) {
  const payload = JSON.stringify(data).replace(/</g, "\\u003c");
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Codebase Roast Map</title>
<style>
:root{--ink:#171512;--paper:#f4efe3;--road:#2f2b25;--water:#7fb7b6;--park:#8fa76f;--hot:#e24b2d;--warm:#f0a33d;--mild:#d8c25f;--safe:#7ea866;--panel:#fffaf0;--line:#2b261f}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:Georgia,"Times New Roman",serif;overflow:hidden}
button,input{font:inherit}
.app{display:grid;grid-template-columns:320px 1fr 340px;height:100vh}
.left,.right{background:var(--panel);border-color:var(--line);border-style:solid;overflow:auto}
.left{border-width:0 2px 0 0;padding:18px}
.right{border-width:0 0 0 2px;padding:18px}
.brand{font-size:30px;line-height:1;font-weight:900;text-transform:uppercase}
.sub{font-size:13px;margin:8px 0 18px;color:#5b5145}
.search{width:100%;border:2px solid var(--line);background:#fffdf7;padding:10px;border-radius:4px}
.layers{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:14px 0}
.layers button,.zoom button{border:2px solid var(--line);background:#fff6d6;padding:9px;border-radius:4px;cursor:pointer}
.layers button.active{background:var(--road);color:var(--paper)}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:16px 0}
.stat{border:2px solid var(--line);padding:10px;background:#fffdf7}
.stat strong{display:block;font-size:24px}
.list{display:flex;flex-direction:column;gap:8px}
.item{border:2px solid var(--line);background:#fffdf7;padding:10px;border-radius:4px;cursor:pointer}
.item:hover{background:#ffe6bb}
.item b{display:block;font-size:13px;word-break:break-word}
.item span{font-size:12px;color:#645b4f}
.mapWrap{position:relative;overflow:hidden;background:linear-gradient(90deg,rgba(47,43,37,.08) 1px,transparent 1px),linear-gradient(rgba(47,43,37,.08) 1px,transparent 1px),#ede2c8;background-size:48px 48px}
.toolbar{position:absolute;top:16px;left:16px;display:flex;gap:8px;z-index:3}
.zoom button{width:42px;height:42px;background:#fffaf0}
.map{position:absolute;inset:0;transform-origin:0 0}
.district{position:absolute;border:3px solid rgba(47,43,37,.55);background:rgba(143,167,111,.22);border-radius:8px}
.districtLabel{position:absolute;font-size:12px;font-weight:900;text-transform:uppercase;letter-spacing:0;background:#fffaf0;border:2px solid var(--line);padding:3px 6px;border-radius:4px;white-space:nowrap}
.road{position:absolute;height:8px;background:var(--road);border-radius:12px;opacity:.75;transform-origin:left center}
.file{position:absolute;border:2px solid var(--line);border-radius:4px;display:flex;align-items:center;justify-content:center;text-align:center;padding:4px;font-size:11px;font-weight:900;cursor:pointer;box-shadow:3px 3px 0 rgba(0,0,0,.25);overflow:hidden}
.file:hover{outline:4px solid #2d6cdf;z-index:4}
.file.hidden{display:none}
.badge{display:inline-block;border:2px solid var(--line);background:#fff6d6;padding:5px 8px;margin:3px;border-radius:20px;font-size:12px}
.detail h2{font-size:20px;word-break:break-word;margin:0 0 10px}
.score{font-size:48px;font-weight:900;line-height:1}
.meter{height:14px;border:2px solid var(--line);background:#fffdf7;margin:8px 0 12px}
.meter span{display:block;height:100%;background:var(--hot)}
.kv{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:14px 0}
.kv div{border:2px solid var(--line);padding:8px;background:#fffdf7}
.kv b{display:block;font-size:18px}
.empty{color:#6c6255}
@media (max-width:900px){.app{grid-template-columns:1fr}.left,.right{position:absolute;z-index:5;width:min(92vw,360px);height:42vh}.left{left:0;top:0}.right{right:0;bottom:0}.mapWrap{height:100vh}}
</style>
</head>
<body>
<div class="app">
<aside class="left">
<div class="brand">Roast Map</div>
<div class="sub">A local city atlas of code pain, traffic, abandoned lots, and suspiciously tall buildings.</div>
<input class="search" id="search" placeholder="Search paths">
<div class="layers" id="layers"></div>
<div class="stats">
<div class="stat"><strong id="fileCount">0</strong>files</div>
<div class="stat"><strong id="riskCount">0</strong>risky</div>
<div class="stat"><strong id="testCount">0</strong>tests</div>
<div class="stat"><strong id="folderCount">0</strong>districts</div>
</div>
<div class="list" id="hotList"></div>
</aside>
<main class="mapWrap" id="wrap">
<div class="toolbar zoom"><button id="zin">+</button><button id="zout">-</button><button id="reset">=</button></div>
<div class="map" id="map"></div>
</main>
<aside class="right detail" id="detail">
<h2>Pick a block</h2>
<p class="empty">Click a file to see why the map is judging it.</p>
</aside>
</div>
<script>
window.ROAST_DATA=${payload};
const data=window.ROAST_DATA;
const fileByPath=new Map(data.files.map(file=>[file.path,file]));
let layer="overall";
let scale=1;
let tx=40;
let ty=40;
let selected=null;
const map=document.getElementById("map");
const detail=document.getElementById("detail");
const search=document.getElementById("search");
const layerNames=["overall","complexity","churn","stale","ownership","tests","bugs","size"];
document.getElementById("fileCount").textContent=data.totals.files;
document.getElementById("riskCount").textContent=data.totals.risky;
document.getElementById("testCount").textContent=data.totals.tests;
document.getElementById("folderCount").textContent=data.totals.folders;
const layers=document.getElementById("layers");
for(const name of layerNames){
  const b=document.createElement("button");
  b.textContent=name;
  b.onclick=()=>{layer=name;renderLayers();};
  layers.appendChild(b);
}
function color(score){
  if(score>=75)return "#e24b2d";
  if(score>=55)return "#f0a33d";
  if(score>=35)return "#d8c25f";
  return "#7ea866";
}
function fileName(path){
  return path.split("/").pop();
}
function layout(){
  const groups=new Map();
  for(const file of data.files){
    const district=file.dir.split("/")[0]||"root";
    if(!groups.has(district))groups.set(district,[]);
    groups.get(district).push(file);
  }
  let x=40;
  let y=40;
  let colH=0;
  const placed=[];
  for(const [district,files] of groups){
    const cols=Math.ceil(Math.sqrt(files.length));
    const cell=92;
    const w=Math.max(260,cols*cell+40);
    const h=Math.max(170,Math.ceil(files.length/cols)*cell+64);
    if(y+h>1800){y=40;x+=430;colH=0;}
    placed.push({type:"district",district,x,y,w,h});
    files.forEach((file,i)=>{
      const fx=x+20+(i%cols)*cell;
      const fy=y+46+Math.floor(i/cols)*cell;
      const s=Math.max(42,Math.min(86,34+file.scores.overall*.45+Math.sqrt(file.loc)));
      placed.push({type:"file",file,x:fx,y:fy,w:s,h:s});
    });
    y+=h+54;
    colH=Math.max(colH,h);
  }
  return placed;
}
const placed=layout();
function transform(){
  map.style.transform="translate("+tx+"px,"+ty+"px) scale("+scale+")";
}
function renderBase(){
  map.innerHTML="";
  const width=Math.max(2400,...placed.map(p=>p.x+p.w+200));
  const height=Math.max(1600,...placed.map(p=>p.y+p.h+200));
  map.style.width=width+"px";
  map.style.height=height+"px";
  for(let i=0;i<36;i++){
    const road=document.createElement("div");
    road.className="road";
    road.style.left=(80+i*150)+"px";
    road.style.top=(120+(i%9)*190)+"px";
    road.style.width=(260+(i%5)*90)+"px";
    road.style.transform="rotate("+(i%2?4:-6)+"deg)";
    map.appendChild(road);
  }
  for(const item of placed.filter(p=>p.type==="district")){
    const d=document.createElement("div");
    d.className="district";
    d.style.left=item.x+"px";
    d.style.top=item.y+"px";
    d.style.width=item.w+"px";
    d.style.height=item.h+"px";
    map.appendChild(d);
    const label=document.createElement("div");
    label.className="districtLabel";
    label.textContent=item.district;
    label.style.left=(item.x+12)+"px";
    label.style.top=(item.y+10)+"px";
    map.appendChild(label);
  }
  for(const item of placed.filter(p=>p.type==="file")){
    const f=document.createElement("div");
    f.className="file";
    f.dataset.path=item.file.path;
    f.style.left=item.x+"px";
    f.style.top=item.y+"px";
    f.style.width=item.w+"px";
    f.style.height=item.h+"px";
    f.textContent=fileName(item.file.path);
    f.onclick=()=>selectFile(item.file);
    map.appendChild(f);
  }
  transform();
}
function renderLayers(){
  [...layers.children].forEach(b=>b.classList.toggle("active",b.textContent===layer));
  document.querySelectorAll(".file").forEach(el=>{
    const file=fileByPath.get(el.dataset.path);
    const score=file.scores[layer]||0;
    el.style.background=color(layer==="overall"?score:score*4);
    el.title=file.path+" "+score;
  });
}
function renderList(){
  const q=search.value.toLowerCase();
  const hot=document.getElementById("hotList");
  hot.innerHTML="";
  const matches=data.files.filter(f=>f.path.toLowerCase().includes(q));
  for(const file of matches.slice(0,18)){
    const item=document.createElement("div");
    item.className="item";
    const title=document.createElement("b");
    title.textContent=file.path;
    const meta=document.createElement("span");
    meta.textContent=file.scores.overall+" "+file.roast;
    item.appendChild(title);
    item.appendChild(meta);
    item.onclick=()=>selectFile(file);
    hot.appendChild(item);
  }
  document.querySelectorAll(".file").forEach(el=>el.classList.toggle("hidden",!el.dataset.path.toLowerCase().includes(q)));
}
function selectFile(file){
  selected=file;
  detail.textContent="";
  const title=document.createElement("h2");
  title.textContent=file.path;
  const score=document.createElement("div");
  score.className="score";
  score.textContent=file.scores.overall;
  const meter=document.createElement("div");
  meter.className="meter";
  const fill=document.createElement("span");
  fill.style.width=file.scores.overall+"%";
  meter.appendChild(fill);
  const roast=document.createElement("p");
  roast.textContent=file.roast;
  const badges=document.createElement("div");
  for(const [k,v] of Object.entries(file.scores)){
    const badge=document.createElement("span");
    badge.className="badge";
    badge.textContent=k+": "+v;
    badges.appendChild(badge);
  }
  const kv=document.createElement("div");
  kv.className="kv";
  const rows=[["loc",file.loc],["commits",file.commits],["authors",file.authors],["bug hits",file.bugs],["nesting",file.nesting],["blocks",file.functions],["nearby tests",file.hasTest?"yes":"no"],["last touched",file.lastDate]];
  for(const [label,value] of rows){
    const box=document.createElement("div");
    const b=document.createElement("b");
    b.textContent=value;
    box.appendChild(b);
    box.appendChild(document.createTextNode(label));
    kv.appendChild(box);
  }
  detail.append(title,score,meter,roast,badges,kv);
}
let dragging=false;
let sx=0;
let sy=0;
document.getElementById("wrap").addEventListener("mousedown",e=>{dragging=true;sx=e.clientX-tx;sy=e.clientY-ty;});
window.addEventListener("mouseup",()=>dragging=false);
window.addEventListener("mousemove",e=>{if(dragging){tx=e.clientX-sx;ty=e.clientY-sy;transform();}});
document.getElementById("zin").onclick=()=>{scale=Math.min(2.5,scale+.15);transform();};
document.getElementById("zout").onclick=()=>{scale=Math.max(.35,scale-.15);transform();};
document.getElementById("reset").onclick=()=>{scale=1;tx=40;ty=40;transform();};
search.addEventListener("input",renderList);
renderBase();
renderLayers();
renderList();
</script>
</body>
</html>`;
}

function writeMap(data) {
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "data.json"), `${JSON.stringify(data, null, 2)}\n`);
  fs.writeFileSync(path.join(outDir, "summary.md"), `${markdownSummary(data)}\n`);
  fs.writeFileSync(path.join(outDir, "index.html"), html(data));
}

function openMap() {
  const target = path.join(outDir, "index.html");
  const platform = process.platform;
  const command = platform === "darwin" ? "open" : platform === "win32" ? "cmd" : "xdg-open";
  const args = platform === "win32" ? ["/c", "start", "", target] : [target];
  try {
    spawnSync(command, args, { stdio: "ignore", detached: true });
  } catch {
  }
}

const data = analyze();

if (mode === "map" || mode === "ui") {
  writeMap(data);
  openMap();
  console.log(`Roast map written to ${path.join(outDir, "index.html")}`);
  console.log(`Summary written to ${path.join(outDir, "summary.md")}`);
} else {
  console.log(terminalReport(data));
}
