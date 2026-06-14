import { readFileSync, existsSync, writeFileSync, mkdirSync, rmSync, readdirSync, statSync } from "node:fs";
import { join, dirname, resolve, relative, basename, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const here = dirname(fileURLToPath(import.meta.url));
const skillRoot = resolve(here, "..");
const STRESS_UPDATES = 40;
const COMPONENT_EXT = new Set([".jsx", ".tsx", ".js", ".ts"]);
const SKIP_DIRS = new Set(["node_modules", "dist", "build", ".next", "out", "coverage", ".rerender-tmp", ".git", ".cache", "public"]);
const ENTRY_RE = /(createRoot|hydrateRoot|ReactDOM\.render|ReactDOM\.hydrate)\s*\(/;

function die(msg) {
  process.stderr.write(msg + "\n");
  process.exit(1);
}
function readJson(p) {
  try { return JSON.parse(readFileSync(p, "utf8")); } catch { return null; }
}
function read(p) {
  try { return readFileSync(p, "utf8"); } catch { return ""; }
}

function walk(dir, acc) {
  let entries;
  try { entries = readdirSync(dir, { withFileTypes: true }); } catch { return acc; }
  for (const e of entries) {
    if (e.name.startsWith(".") && e.name !== ".") continue;
    const full = join(dir, e.name);
    if (e.isDirectory()) {
      if (SKIP_DIRS.has(e.name)) continue;
      walk(full, acc);
    } else if (COMPONENT_EXT.has(extname(e.name))) {
      if (/\.(test|spec|stories|d)\./.test(e.name)) continue;
      acc.push(full);
    }
  }
  return acc;
}

function detectExportName(src) {
  let m = src.match(/export\s+default\s+function\s+([A-Z][A-Za-z0-9_]*)/);
  if (m) return m[1];
  m = src.match(/export\s+default\s+(?:React\.)?(?:memo|forwardRef)\s*\(\s*(?:function\s+)?([A-Z][A-Za-z0-9_]*)/);
  if (m) return m[1];
  m = src.match(/export\s+default\s+([A-Z][A-Za-z0-9_]*)\s*;/);
  if (m) return m[1];
  if (/export\s+default\s+(?:React\.)?(?:memo|forwardRef)\s*\(/.test(src)) return null;
  if (/export\s+default\s+(?:function|\()/.test(src)) return null;
  m = src.match(/export\s+(?:const|function)\s+([A-Z][A-Za-z0-9_]*)/);
  if (m) return m[1];
  return null;
}

function looksLikeComponent(src, ext) {
  const isReact = /from\s+["']react["']/.test(src) || /from\s+["']react\/jsx-runtime["']/.test(src);
  const hasJsx = /<[A-Za-z][^>]*>/.test(src) || /jsx\(/.test(src) || /createElement\(/.test(src);
  const hasDefault = /export\s+default/.test(src);
  const hasPascalExport = /export\s+(?:const|function)\s+[A-Z]/.test(src);
  if (ext === ".jsx" || ext === ".tsx") return (hasDefault || hasPascalExport) && (hasJsx || isReact);
  return isReact && hasJsx && (hasDefault || hasPascalExport);
}

function staticAnalysis(src) {
  const count = (re) => (src.match(re) || []).length;
  const hasMemo = /(?:React\.)?memo\s*\(/.test(src);
  const hooks = {
    useState: count(/\buseState\s*\(/g),
    useEffect: count(/\buseEffect\s*\(/g),
    useMemo: count(/\buseMemo\s*\(/g),
    useCallback: count(/\buseCallback\s*\(/g),
    useRef: count(/\buseRef\s*\(/g),
    useContext: count(/\buseContext\s*\(/g),
    useReducer: count(/\buseReducer\s*\(/g),
  };
  const inlineObj = count(/=\{\{/g);
  const inlineArr = count(/=\{\[/g);
  const inlineFn = count(/=\{\s*(?:\([^)]*\)|[A-Za-z0-9_,\s]+)\s*=>/g) + count(/=\{\s*function\b/g);
  const arrayOps = count(/\.(?:map|filter|sort|reduce|flatMap)\s*\(/g);
  const loc = src.split("\n").filter((l) => l.trim().length).length;
  return {
    hasMemo,
    hooks,
    inlineProps: inlineObj + inlineArr + inlineFn,
    arrayOps,
    memoizedComputations: hooks.useMemo + hooks.useCallback,
    loc,
  };
}

const WRAPPERS = new Set(["StrictMode", "Fragment", "Suspense", "Profiler", "ErrorBoundary", "React"]);

function detectRoot(files) {
  for (const f of files) {
    const src = read(f);
    if (!ENTRY_RE.test(src)) continue;
    const at = src.search(/\.render\s*\(/);
    if (at === -1) continue;
    const region = src.slice(at, at + 600);
    const names = [];
    for (const m of region.matchAll(/<\s*([A-Z][A-Za-z0-9_]*)/g)) names.push(m[1]);
    for (const m of region.matchAll(/createElement\s*\(\s*([A-Z][A-Za-z0-9_]*)/g)) names.push(m[1]);
    const root = names.find((n) => !WRAPPERS.has(n));
    if (root) return root;
  }
  return null;
}

function clamp(n, lo, hi) { return Math.max(lo, Math.min(hi, n)); }

function scoreComponent(c) {
  const s = c.static;
  const issues = [];
  let score = 100;

  if (c.runtime && c.runtime.stress) {
    const ratio = c.runtime.stressUpdates ? c.runtime.wastedRenders / c.runtime.stressUpdates : 0;
    score -= 60 * ratio;
    if (c.runtime.wastedRenders > 0) {
      issues.push({
        code: "WASTED_RENDERS",
        severity: ratio >= 0.5 ? "high" : "medium",
        title: c.runtime.wastedRenders + " of " + c.runtime.stressUpdates + " re-renders were wasted",
        detail: "Under " + c.runtime.stressUpdates + " parent updates with unchanged props the component re-rendered " + c.runtime.wastedRenders + " times.",
        fix: s.hasMemo
          ? "memo() is present but defeated — a prop changes identity each render. Memoize objects/arrays/callbacks passed in with useMemo / useCallback."
          : "Wrap the export in React.memo so it skips renders when its props are unchanged.",
      });
    }
    const dur = c.runtime.avgUpdateMs;
    score -= clamp(dur * 4, 0, 30);
    if (dur > 1.5 && s.arrayOps > 0 && s.useMemo === 0) {
      issues.push({
        code: "EXPENSIVE_RENDER",
        severity: dur > 5 ? "high" : "medium",
        title: "Heavy work on every render (" + dur.toFixed(2) + " ms avg)",
        detail: "The render body runs map/sort/reduce on each pass without memoization.",
        fix: "Move the derivation into useMemo so it only recomputes when its inputs change.",
      });
    }
  }

  if (c.runtime && c.runtime.mountMs > 6) {
    score -= clamp(c.runtime.mountMs / 4, 0, 12);
  }

  if (s.inlineProps > 0) {
    score -= 5;
    issues.push({
      code: "INLINE_PROPS",
      severity: "low",
      title: s.inlineProps + " inline object/array/function prop" + (s.inlineProps > 1 ? "s" : ""),
      detail: "Inline {{ }}, [ ] or arrow props are recreated every render and break a child's React.memo.",
      fix: "Hoist constants out of render, or memoize them with useMemo / useCallback.",
    });
  }
  if (c.runtime && c.runtime.stress && c.runtime.wastedRenders > 0 && !s.hasMemo && s.arrayOps > 0 && s.useMemo === 0) {
    score -= 5;
  }

  score = clamp(Math.round(score), 0, 100);
  const grade = score >= 90 ? "good" : score >= 50 ? "warn" : "poor";
  return { score, grade, issues };
}

function genDomModule() {
  return `import { JSDOM } from "jsdom";
const dom = new JSDOM("<!doctype html><html><body></body></html>", { pretendToBeVisual: true, url: "http://localhost/" });
const { window } = dom;
const set = (k, v) => { try { globalThis[k] = v; } catch {} };
set("window", window);
set("document", window.document);
set("HTMLElement", window.HTMLElement);
set("Node", window.Node);
set("Element", window.Element);
set("Event", window.Event);
set("getComputedStyle", window.getComputedStyle.bind(window));
set("requestAnimationFrame", window.requestAnimationFrame ? window.requestAnimationFrame.bind(window) : (cb) => setTimeout(() => cb(Date.now()), 0));
set("cancelAnimationFrame", window.cancelAnimationFrame ? window.cancelAnimationFrame.bind(window) : clearTimeout);
if (!globalThis.navigator || !globalThis.navigator.userAgent) set("navigator", window.navigator);
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
`;
}

function genHarness(specs) {
  const imports = specs.map((s, i) => `import * as M${i} from ${JSON.stringify(s.path)};`).join("\n");
  const list = specs
    .map((s, i) => `  { id: ${JSON.stringify(s.id)}, file: ${JSON.stringify(s.file)}, mod: M${i}, name: ${JSON.stringify(s.name)}, stress: ${s.stress}, isRoot: ${s.isRoot} }`)
    .join(",\n");
  return `import "./__dom.mjs";
import * as React from "react";
import { createRoot } from "react-dom/client";
${imports}

const K = ${STRESS_UPDATES};
const act = React.act || ((cb) => { const r = cb(); return Promise.resolve(r); });
const origError = console.error; const origWarn = console.warn;
console.error = () => {}; console.warn = () => {};

const SPECS = [
${list}
];

function pick(mod, name) {
  const cands = [mod && mod.default, name && mod && mod[name], ...(mod ? Object.values(mod) : [])];
  for (const c of cands) {
    if (typeof c === "function") return c;
    if (c && typeof c === "object" && c.$$typeof) return c;
  }
  return null;
}

async function measure(spec) {
  const Comp = pick(spec.mod, spec.name);
  if (!Comp) return { error: "no renderable component export found" };
  const samples = [];
  let setTick = null;
  const onRender = (id, phase, actualDuration) => samples.push({ phase, ms: actualDuration });

  function Stress() {
    const [, setT] = React.useState(0);
    setTick = setT;
    return React.createElement(React.Profiler, { id: spec.id, onRender }, React.createElement(Comp, null));
  }

  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  try {
    await act(async () => { root.render(React.createElement(Stress)); });
  } catch (e) {
    try { await act(async () => root.unmount()); } catch {}
    container.remove();
    return { error: String((e && e.message) || e) };
  }

  if (spec.stress && setTick) {
    for (let i = 0; i < K; i++) {
      await act(async () => { setTick(i + 1); });
    }
  }
  try { await act(async () => root.unmount()); } catch {}
  container.remove();
  return { samples, stressUpdates: spec.stress ? K : 0, isRoot: spec.isRoot };
}

async function main() {
  const out = [];
  for (const spec of SPECS) {
    let res;
    try { res = await measure(spec); } catch (e) { res = { error: String((e && e.message) || e) }; }
    out.push({ id: spec.id, file: spec.file, ...res });
  }
  origError; origWarn;
  process.stdout.write("RERENDER_JSON_START" + JSON.stringify(out) + "RERENDER_JSON_END\\n");
  process.exit(0);
}
main();
`;
}

async function bundle(harnessPath, outPath, root) {
  let esbuild;
  try { esbuild = await import("esbuild"); }
  catch { die("esbuild is not installed in the skill - run install.sh"); }
  const empty = {};
  for (const e of [".css", ".scss", ".sass", ".less", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".avif", ".ico", ".woff", ".woff2", ".ttf", ".otf", ".mp3", ".mp4", ".webm"]) empty[e] = "empty";
  try {
    await esbuild.build({
      entryPoints: [harnessPath],
      bundle: true,
      platform: "node",
      format: "cjs",
      target: "node20",
      outfile: outPath,
      absWorkingDir: root,
      nodePaths: [join(root, "node_modules"), join(skillRoot, "node_modules")],
      external: ["jsdom"],
      jsx: "automatic",
      loader: empty,
      logLevel: "silent",
      define: { "process.env.NODE_ENV": '"development"' },
    });
  } catch (e) {
    die("could not bundle the project's components:\n" + ((e && e.message) || String(e)));
  }
  return esbuild.version;
}

async function main() {
  const root = resolve(process.argv[2] || process.cwd());
  const pkg = readJson(join(root, "package.json")) || {};
  const projectName = pkg.name || basename(root);
  if (!existsSync(join(root, "node_modules", "react"))) {
    die("react not found in " + root + "/node_modules - run `npm install` in the project first");
  }
  const reactVersion = (readJson(join(root, "node_modules", "react", "package.json")) || {}).version || "unknown";

  const scanDir = existsSync(join(root, "src")) ? join(root, "src") : root;
  const files = walk(scanDir, []);
  const rootName = detectRoot(files);

  const specs = [];
  const components = [];
  for (const f of files) {
    const src = read(f);
    if (ENTRY_RE.test(src)) continue;
    const ext = extname(f);
    if (!looksLikeComponent(src, ext)) continue;
    const name = detectExportName(src);
    const rel = relative(root, f);
    const id = name || basename(f, ext);
    const isRoot = !!(name && rootName && name === rootName);
    specs.push({ id, name, path: f, file: rel, stress: !isRoot, isRoot });
    components.push({ id, name, file: rel, isRoot, static: staticAnalysis(src) });
  }

  if (specs.length === 0) {
    die("no React components found under " + scanDir);
  }

  const tmp = join(root, ".rerender-tmp");
  rmSync(tmp, { recursive: true, force: true });
  mkdirSync(tmp, { recursive: true });
  writeFileSync(join(tmp, "__dom.mjs"), genDomModule());
  writeFileSync(join(tmp, "harness.mjs"), genHarness(specs));
  const bundled = join(tmp, "harness.bundle.cjs");
  const esbuildVersion = await bundle(join(tmp, "harness.mjs"), bundled, root);

  const run = spawnSync(process.execPath, [bundled], {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
    env: { ...process.env, NODE_PATH: join(skillRoot, "node_modules") },
  });
  rmSync(tmp, { recursive: true, force: true });
  if (run.status !== 0 && !run.stdout) {
    die("the measurement harness failed:\n" + (run.stderr || "").slice(0, 4000));
  }
  const m = (run.stdout || "").match(/RERENDER_JSON_START([\s\S]*?)RERENDER_JSON_END/);
  if (!m) die("no measurement output produced:\n" + (run.stderr || "").slice(0, 2000));
  const raw = JSON.parse(m[1]);
  const byFile = new Map(raw.map((r) => [r.file, r]));

  for (const c of components) {
    const r = byFile.get(c.file);
    if (!r || r.error) {
      c.runtime = null;
      c.runtimeError = (r && r.error) || "not measured";
    } else {
      const samples = r.samples || [];
      const mount = samples[0] || { ms: 0 };
      const updates = samples.slice(1);
      const effective = updates.filter((s) => s.ms > 0.01);
      const durs = effective.map((s) => s.ms);
      const sum = durs.reduce((a, b) => a + b, 0);
      c.runtime = {
        stress: !!r.stressUpdates,
        stressUpdates: r.stressUpdates || 0,
        isRoot: !!r.isRoot,
        commits: samples.length,
        mountMs: +mount.ms.toFixed(4),
        wastedRenders: effective.length,
        avgUpdateMs: durs.length ? +(sum / durs.length).toFixed(4) : 0,
        maxUpdateMs: durs.length ? +Math.max(...durs).toFixed(4) : 0,
        totalUpdateMs: +sum.toFixed(4),
        series: durs.slice(0, STRESS_UPDATES).map((d) => +d.toFixed(4)),
      };
    }
    const sc = scoreComponent(c);
    c.score = sc.score;
    c.grade = sc.grade;
    c.issues = sc.issues;
  }

  components.sort((a, b) => a.score - b.score);

  const measured = components.filter((c) => c.runtime);
  const stressed = measured.filter((c) => c.runtime.stress);
  const avgScore = measured.length ? Math.round(measured.reduce((a, c) => a + c.score, 0) / measured.length) : 0;
  const totalWasted = stressed.reduce((a, c) => a + c.runtime.wastedRenders, 0);
  const totalStress = stressed.reduce((a, c) => a + c.runtime.stressUpdates, 0);
  const totalRenderMs = measured.reduce((a, c) => a + c.runtime.totalUpdateMs + c.runtime.mountMs, 0);
  const issueCount = components.filter((c) => c.issues.length > 0).length;
  const slowest = [...measured].sort((a, b) => b.runtime.maxUpdateMs - a.runtime.maxUpdateMs)[0] || null;
  const worst = components[0] || null;

  const data = {
    meta: {
      project: projectName,
      generatedAt: new Date().toISOString(),
      reactVersion,
      esbuildVersion,
      stressUpdates: STRESS_UPDATES,
      componentCount: components.length,
      measuredCount: measured.length,
      scanRoot: relative(root, scanDir) || ".",
    },
    summary: {
      avgScore,
      totalWasted,
      totalStress,
      totalRenderMs: +totalRenderMs.toFixed(2),
      issueCount,
      slowest: slowest ? { id: slowest.id, ms: slowest.runtime.maxUpdateMs } : null,
      worst: worst ? { id: worst.id, score: worst.score } : null,
      good: components.filter((c) => c.grade === "good").length,
      warn: components.filter((c) => c.grade === "warn").length,
      poor: components.filter((c) => c.grade === "poor").length,
    },
    components,
  };

  const outDir = join(process.cwd(), "rerender-report");
  mkdirSync(outDir, { recursive: true });
  const template = readFileSync(join(skillRoot, "assets", "template.html"), "utf8");
  const payload = JSON.stringify(data).replace(/<\//g, "<\\/");
  writeFileSync(join(outDir, "index.html"), template.replace("__RERENDER_DATA__", payload));
  writeFileSync(join(outDir, "data.json"), JSON.stringify(data, null, 2));

  const pad = (s, n) => String(s).padEnd(n);
  process.stdout.write("project        " + projectName + "  (React " + reactVersion + ")\n");
  process.stdout.write("components     " + components.length + " found, " + measured.length + " measured, stress " + STRESS_UPDATES + " parent updates\n");
  process.stdout.write("overall score  " + avgScore + "/100   good " + data.summary.good + "  warn " + data.summary.warn + "  poor " + data.summary.poor + "\n");
  process.stdout.write("wasted renders " + totalWasted + " of " + totalStress + " stressed updates\n");
  if (slowest) process.stdout.write("slowest render " + slowest.id + "  " + slowest.runtime.maxUpdateMs.toFixed(2) + " ms\n");
  process.stdout.write("\nper component (worst first):\n");
  for (const c of components) {
    const r = c.runtime;
    const tag = c.isRoot ? " [root]" : "";
    const w = r && r.stress ? r.wastedRenders + "/" + r.stressUpdates + " wasted" : r ? "root mount only" : "not measured";
    process.stdout.write("  " + pad(c.score, 4) + pad(c.id + tag, 20) + pad(w, 18) + (r ? "avg " + r.avgUpdateMs.toFixed(2) + "ms" : c.runtimeError) + "\n");
  }
  process.stdout.write("\nreport written to " + join(outDir, "index.html") + "\n");
}

main().catch((e) => die(String((e && e.stack) || e)));
