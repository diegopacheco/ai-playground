import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

export function skillFolder(root, source, folder, content) {
  const dir = join(root, source, folder);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "SKILL.md"), content);
  return dir;
}

export function skillMd(name, description, body) {
  return `---\nname: ${name}\ndescription: ${description}\n---\n\n${body}\n`;
}

export function tempRoot() {
  return mkdtempSync(join(tmpdir(), "skills-gateway-"));
}
