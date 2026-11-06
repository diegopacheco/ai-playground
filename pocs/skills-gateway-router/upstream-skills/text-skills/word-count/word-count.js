import { readFileSync } from "node:fs";

export function countText(text) {
  return {
    words: (text.match(/\S+/g) ?? []).length,
    lines: text === "" ? 0 : text.split(/\r?\n/).length - (text.endsWith("\n") ? 1 : 0),
    characters: [...text].length,
  };
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  const text = args[0] === "--file" ? readFileSync(args[1], "utf8") : args.join(" ");
  const { words, lines, characters } = countText(text);
  console.log(`words ${words}\nlines ${lines}\ncharacters ${characters}`);
}
