export function ruleNames(yaml: string): string[] {
  const lines = yaml.split("\n");
  const start = lines.findIndex((line) => line.startsWith("network_policies:"));
  if (start === -1) {
    return [];
  }
  const names: string[] = [];
  for (const line of lines.slice(start + 1)) {
    if (/^\S/.test(line)) {
      break;
    }
    const match = /^  ([A-Za-z0-9_-]+):\s*$/.exec(line);
    if (match) {
      names.push(match[1]);
    }
  }
  return names;
}
