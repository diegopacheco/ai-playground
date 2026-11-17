import type { Playbook, PlaybookId, Principle, PrincipleGroup } from "../src/domain/types";

const GROUPS: readonly PrincipleGroup[] = ["Core", "Architecture", "Verification", "Delegation", "Meta"];

export function plain(markdown: string): string {
  return markdown.replace(/\*\*/g, "").replace(/`/g, "").trim();
}

export function parsePlaybook(id: string, markdown: string): Playbook {
  const lines = markdown.split("\n");
  const title = plain((lines.find((l) => l.startsWith("### ")) ?? id).replace(/^### /, ""));
  const numbered = lines.filter((l) => /^\d+\. /.test(l)).map((l) => plain(l.replace(/^\d+\. /, "")));
  const bullets = lines.filter((l) => l.startsWith("- ")).map((l) => plain(l.slice(2)));
  return { id: id as PlaybookId, title, steps: numbered.length > 0 ? numbered : bullets };
}

export function parsePrinciples(skillMarkdown: string): Principle[] {
  const section = skillMarkdown.split("## Principles")[1]?.split("\n## ")[0] ?? "";
  const principles: Principle[] = [];
  let group: PrincipleGroup | undefined;
  for (const line of section.split("\n")) {
    const heading = GROUPS.find((g) => line.trim() === `**${g}**`);
    if (heading) {
      group = heading;
      continue;
    }
    const match = /^- \*\*([^*]+?)\.?\*\*(?: \([^)]*\))?\.?\s+(.*)$/.exec(line);
    if (match && group && match[1] && match[2]) {
      principles.push({ name: match[1], group, when: plain(match[2]) });
    }
  }
  return principles;
}
