import type { StoryPage } from "./pages";

export const NARRATION_DELAY_MS = 1400;

function speakable(text: string): string {
  return text
    .replace(/(?<!\p{L})(\p{Lu})(\p{Lu}+)(?!\p{L})/gu, (_, first: string, rest: string) => first + rest.toLowerCase())
    .replace(/[“”"]/g, "");
}

export function narrationText(page: StoryPage): string {
  return speakable([`${page.title}.`, ...page.lines].join(" "));
}

export function narrationParts(page: StoryPage): string[] {
  return narrationText(page)
    .split(/(?<=[.!?])\s+/)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}
