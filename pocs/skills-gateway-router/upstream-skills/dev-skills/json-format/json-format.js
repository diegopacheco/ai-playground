import { readFileSync } from "node:fs";

export function formatJson(text) {
  return JSON.stringify(JSON.parse(text), null, 2);
}

if (import.meta.main) {
  try {
    console.log(formatJson(readFileSync(process.argv[2], "utf8")));
  } catch (error) {
    console.error(`invalid JSON: ${error.message}`);
    process.exit(1);
  }
}
