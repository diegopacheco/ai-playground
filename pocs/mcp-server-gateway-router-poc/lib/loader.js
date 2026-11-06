import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { toolRegistry } from "./mcp-server.js";
import { markdownTools } from "./md-server.js";

export function markdownUpstream(name, markdown, source) {
  return { name, kind: "md", source, local: toolRegistry(markdownTools(markdown)) };
}

export async function codeUpstream(name, file, source) {
  const absolute = path.resolve(file);
  const version = fs.statSync(absolute).mtimeMs;
  const { tools } = await import(`${pathToFileURL(absolute).href}?v=${version}`);
  if (!Array.isArray(tools)) throw new Error(`${absolute} must export a 'tools' array`);
  return { name, kind: "code", source, local: toolRegistry(tools) };
}

export async function scanDir(dir) {
  const found = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    try {
      if (entry.isFile() && entry.name.endsWith(".md")) {
        found.push(markdownUpstream(entry.name.slice(0, -3), fs.readFileSync(full, "utf8"), "dir"));
      } else if (entry.isDirectory() && fs.existsSync(path.join(full, "tools.js"))) {
        found.push(await codeUpstream(entry.name.replace(/-mcp$/, ""), path.join(full, "tools.js"), "dir"));
      }
    } catch (error) {
      console.error(`skipped ${full}: ${error.message}`);
    }
  }
  return found;
}

export async function syncDir(dir, router) {
  const found = await scanDir(dir);
  const names = new Set(found.map((upstream) => upstream.name));
  router.servers().filter((s) => s.source === "dir" && !names.has(s.name)).forEach((s) => router.remove(s.name));
  found.forEach((upstream) => {
    const existing = router.get(upstream.name);
    if (existing && existing.source !== "dir") return console.error(`skipped ${upstream.name}: name already registered`);
    try {
      router.add(upstream);
    } catch (error) {
      console.error(`skipped ${upstream.name}: ${error.message}`);
    }
  });
  return router.servers().filter((s) => s.source === "dir").map((s) => s.name);
}

export function watchDir(dir, router, delayMs = 150) {
  let timer = null;
  let running = Promise.resolve();
  const schedule = () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      running = running.then(() => syncDir(dir, router)).catch((error) => console.error(error.message));
    }, delayMs);
  };
  const watcher = fs.watch(dir, { recursive: true }, schedule);
  return {
    close: () => {
      clearTimeout(timer);
      watcher.close();
    },
    settled: () => running,
  };
}
