import { CHARACTERS } from "./catalog";
import type { ActivityStats, CharacterId, CharacterSpec, UnlockRule } from "./types";

export type UnlockProgress = {
  readonly current: number;
  readonly target: number;
  readonly done: boolean;
};

function ruleProgress(rule: UnlockRule, stats: ActivityStats): UnlockProgress {
  switch (rule.kind) {
    case "starter":
      return { current: 1, target: 1, done: true };
    case "cook":
      return {
        current: Math.min(stats.cookedCount, rule.target),
        target: rule.target,
        done: stats.cookedCount >= rule.target,
      };
    case "paint":
      return {
        current: Math.min(stats.paintedCount, rule.target),
        target: rule.target,
        done: stats.paintedCount >= rule.target,
      };
    case "move":
      return {
        current: Math.min(stats.movedCount, rule.target),
        target: rule.target,
        done: stats.movedCount >= rule.target,
      };
    case "call":
      return {
        current: Math.min(stats.calledCount, rule.target),
        target: rule.target,
        done: stats.calledCount >= rule.target,
      };
    default: {
      const unreachable: never = rule;
      return unreachable;
    }
  }
}

export function getUnlockProgress(spec: CharacterSpec, stats: ActivityStats): UnlockProgress {
  return ruleProgress(spec.unlockRule, stats);
}

export function evaluateUnlocks(
  stats: ActivityStats,
  alreadyUnlocked: readonly CharacterId[],
): {
  readonly unlocked: readonly CharacterId[];
  readonly newlyUnlocked: readonly CharacterId[];
} {
  const existing = new Set<CharacterId>(alreadyUnlocked);
  const newlyUnlocked: CharacterId[] = [];
  for (const spec of CHARACTERS) {
    if (!existing.has(spec.id) && ruleProgress(spec.unlockRule, stats).done) {
      existing.add(spec.id);
      newlyUnlocked.push(spec.id);
    }
  }
  const ordered = CHARACTERS.filter((c) => existing.has(c.id)).map((c) => c.id);
  return { unlocked: ordered, newlyUnlocked };
}
