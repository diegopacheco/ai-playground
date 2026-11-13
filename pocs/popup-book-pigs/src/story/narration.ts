import type { StoryPage } from "./pages";

export const NARRATION_DELAY_MS = 1400;

function speakable(text: string): string {
  return text.replace(/\b([A-Z])([A-Z]+)\b/g, (_, first: string, rest: string) => first + rest.toLowerCase());
}

export function narrationText(page: StoryPage): string {
  return speakable([`${page.title}.`, ...page.lines].join(" "));
}
