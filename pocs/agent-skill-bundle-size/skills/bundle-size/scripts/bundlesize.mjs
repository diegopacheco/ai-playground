import { gzipSync } from "node:zlib";
import { readFileSync, existsSync, writeFileSync, mkdirSync } from "node:fs";
import { join, dirname, resolve, isAbsolute } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ASSET_LOADERS = {
  ".png": "dataurl", ".jpg": "dataurl", ".jpeg": "dataurl", ".gif": "dataurl",
  ".webp": "dataurl", ".avif": "dataurl", ".svg": "dataurl", ".ico": "dataurl",
  ".woff": "dataurl", ".woff2": "dataurl", ".ttf": "dataurl", ".otf": "dataurl",
  ".eot": "dataurl", ".mp3": "dataurl", ".mp4": "dataurl", ".webm": "dataurl",
  ".js": "jsx", ".cjs": "jsx", ".mjs": "jsx",
};
const ENTRY_CANDIDATES = [
  "src/main.jsx", "src/main.tsx", "src/main.js", "src/main.ts",
  "src/index.jsx", "src/index.tsx", "src/index.js", "src/index.ts",
  "src/App.jsx", "src/App.tsx", "index.jsx", "index.tsx", "index.js", "index.ts",
];

function die(msg) {
  process.stderr.write(msg + "\n");
  process.exit(1);
}

function readJson(path) {
  try { return JSON.parse(readFileSync(path, "utf8")); } catch { return null; }
}

function findEntry(root, override) {
  if (override) {
    const p = isAbsolute(override) ? override : join(root, override);
    if (existsSync(p)) return p;
    die("entry not found: " + override);
  }
  const html = join(root, "index.html");
  if (existsSync(html)) {
    const m = readFileSync(html, "utf8").match(/<script[^>]+src=["']([^"']+)["']/i);
    if (m) {
      const src = m[1].replace(/^\//, "");
      const p = join(root, src);
      if (existsSync(p)) return p;
    }
  }
  for (const c of ENTRY_CANDIDATES) {
    const p = join(root, c);
    if (existsSync(p)) return p;
  }
  return null;
}

function packageOf(inputPath) {
  const marker = "node_modules/";
  const i = inputPath.lastIndexOf(marker);
  if (i === -1) return { name: "(application code)", kind: "app" };
  const rest = inputPath.slice(i + marker.length).split("/");
  const name = rest[0].startsWith("@") ? rest[0] + "/" + rest[1] : rest[0];
  return { name, kind: "vendor" };
}

function versionOf(root, name) {
  const p = join(root, "node_modules", name, "package.json");
  const pkg = readJson(p);
  return pkg && pkg.version ? pkg.version : null;
}

async function main() {
  const root = resolve(process.argv[2] || process.cwd());
  const entryOverride = process.argv[3] || null;

  const pkgJson = readJson(join(root, "package.json")) || {};
  const projectName = pkgJson.name || root.split("/").pop();

  const entry = findEntry(root, entryOverride);
  if (!entry) {
    die("could not locate a frontend entry point under " + root +
      "\ntried: " + ENTRY_CANDIDATES.join(", ") +
      "\npass one explicitly: bundlesize.mjs <project> <src/main.jsx>");
  }
  if (!existsSync(join(root, "node_modules"))) {
    die("node_modules not found in " + root + " - run `npm install` first");
  }

  let esbuild;
  try { esbuild = await import("esbuild"); }
  catch { die("esbuild is not installed in the skill - run install.sh"); }

  let result;
  try {
    result = await esbuild.build({
      entryPoints: [entry],
      bundle: true,
      minify: true,
      metafile: true,
      write: false,
      format: "esm",
      platform: "browser",
      target: "es2020",
      jsx: "automatic",
      logLevel: "silent",
      absWorkingDir: root,
      outfile: "bundle.js",
      loader: ASSET_LOADERS,
      define: { "process.env.NODE_ENV": '"production"', "import.meta.env.MODE": '"production"', "import.meta.env.DEV": "false", "import.meta.env.PROD": "true" },
    });
  } catch (e) {
    const text = (e && e.message) || String(e);
    die("esbuild could not bundle this project:\n" + text +
      "\n\nthis attributor bundles plain JS/TS/JSX/CSS. projects using SCSS/LESS, " +
      "vite path aliases, or other loaders need their own build's stats instead.");
  }

  const meta = result.metafile;
  const gzipBytes = gzipSync(Buffer.concat(result.outputFiles.map((f) => Buffer.from(f.contents)))).length;

  let totalMinified = 0;
  for (const out of Object.values(meta.outputs)) totalMinified += out.bytes;

  const perInput = new Map();
  for (const out of Object.values(meta.outputs)) {
    for (const [path, info] of Object.entries(out.inputs)) {
      const cur = perInput.get(path) || 0;
      perInput.set(path, cur + info.bytesInOutput);
    }
  }

  const packages = new Map();
  const modules = [];
  for (const [path, minified] of perInput) {
    if (minified <= 0) continue;
    const { name, kind } = packageOf(path);
    const raw = (meta.inputs[path] && meta.inputs[path].bytes) || 0;
    modules.push({ path, package: name, kind, minifiedBytes: minified, rawBytes: raw });
    let p = packages.get(name);
    if (!p) { p = { name, kind, version: kind === "vendor" ? versionOf(root, name) : null, minifiedBytes: 0, rawBytes: 0, fileCount: 0 }; packages.set(name, p); }
    p.minifiedBytes += minified;
    p.rawBytes += raw;
    p.fileCount += 1;
  }

  const pkgList = [...packages.values()].sort((a, b) => b.minifiedBytes - a.minifiedBytes);
  for (const p of pkgList) p.pct = totalMinified ? (p.minifiedBytes / totalMinified) * 100 : 0;
  modules.sort((a, b) => b.minifiedBytes - a.minifiedBytes);

  let appBytes = 0, vendorBytes = 0, appModules = 0, vendorModules = 0;
  for (const m of modules) {
    if (m.kind === "app") { appBytes += m.minifiedBytes; appModules++; }
    else { vendorBytes += m.minifiedBytes; vendorModules++; }
  }

  const headline = pkgList[0] || null;
  const heaviestVendor = pkgList.find((p) => p.kind === "vendor") || null;
  const rawTotal = modules.reduce((a, m) => a + m.rawBytes, 0);

  const data = {
    meta: {
      project: projectName,
      entry: entry.startsWith(root) ? entry.slice(root.length + 1) : entry,
      generatedAt: new Date().toISOString(),
      esbuildVersion: esbuild.version,
      totalPackages: pkgList.length,
      totalModules: modules.length,
    },
    totals: {
      rawBytes: rawTotal,
      minifiedBytes: totalMinified,
      gzipBytes,
      appBytes, vendorBytes, appModules, vendorModules,
    },
    headline, heaviestVendor,
    packages: pkgList,
    modules: modules.slice(0, 500),
    treemap: pkgList.map((p) => ({ name: p.name, value: p.minifiedBytes, kind: p.kind, pct: p.pct })),
  };

  const outDir = join(process.cwd(), "bundle-size-report");
  mkdirSync(outDir, { recursive: true });
  const template = readFileSync(join(here, "..", "assets", "template.html"), "utf8");
  const payload = JSON.stringify(data).replace(/<\//g, "<\\/");
  writeFileSync(join(outDir, "index.html"), template.replace("__BUNDLE_DATA__", payload));
  writeFileSync(join(outDir, "data.json"), JSON.stringify(data, null, 2));

  const kb = (n) => (n / 1024).toFixed(1) + " KB";
  process.stdout.write("project        " + projectName + "\n");
  process.stdout.write("entry          " + data.meta.entry + "\n");
  process.stdout.write("bundle (min)   " + kb(totalMinified) + "  gzip " + kb(gzipBytes) + "\n");
  process.stdout.write("modules        " + modules.length + " across " + pkgList.length + " packages\n");
  process.stdout.write("app vs vendor  " + kb(appBytes) + " app / " + kb(vendorBytes) + " vendor\n");
  if (headline) process.stdout.write("heaviest       " + headline.name + "  " + kb(headline.minifiedBytes) + " (" + headline.pct.toFixed(1) + "% of bundle)\n");
  process.stdout.write("\ntop imports by minified size:\n");
  for (const p of pkgList.slice(0, 10)) {
    process.stdout.write("  " + p.minifiedBytes.toString().padStart(9) + " B  " + p.pct.toFixed(1).padStart(5) + "%  " + p.name + (p.version ? "@" + p.version : "") + "\n");
  }
  process.stdout.write("\nreport written to " + join(outDir, "index.html") + "\n");
}

main().catch((e) => die(String(e && e.stack || e)));
