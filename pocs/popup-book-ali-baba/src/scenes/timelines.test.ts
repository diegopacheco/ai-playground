import { describe, expect, test } from "bun:test";
import { ALI_DOOR, doorMarks, doorOpen, DOORS, lidLift, JAR_LOOP, JAR_X, lanternX, shoutOn, SESAME_LOOP, STREET_LOOP, THIEF_MARK_AT, WRONG_WORDS, wrongWord } from "./timelines";

const sample = (length: number, step = 0.05) => Array.from({ length: Math.floor(length / step) }, (_, i) => i * step);

describe("the chalk mark", () => {
  test("the thief marks only Ali Baba's door", () => {
    const marks = doorMarks(THIEF_MARK_AT + 0.1);
    expect(marks[ALI_DOOR]).toBe(true);
    expect(marks.filter(Boolean).length).toBe(1);
  });

  test("no door is marked before the thief comes", () => {
    expect(doorMarks(THIEF_MARK_AT - 0.1).some(Boolean)).toBe(false);
  });

  test("Morgiana marks every door before the night ends, so the thieves cannot tell them apart", () => {
    expect(doorMarks(STREET_LOOP - 0.05).every(Boolean)).toBe(true);
    expect(DOORS.length).toBeGreaterThanOrEqual(5);
  });

  test("once a door is marked it stays marked for the rest of the night", () => {
    let previous = doorMarks(0);
    for (const t of sample(STREET_LOOP - 0.05)) {
      const now = doorMarks(t);
      previous.forEach((was, i) => was && expect(now[i]).toBe(true));
      previous = now;
    }
  });
});

describe("open sesame", () => {
  test("the stone only starts to move after the captain shouts", () => {
    const firstOpen = sample(SESAME_LOOP).find((t) => doorOpen(t) > 0.01)!;
    const firstShout = sample(SESAME_LOOP).find((t) => shoutOn(t))!;
    expect(firstShout).toBeLessThan(firstOpen);
  });

  test("Kassim never remembers the right word", () => {
    for (const t of sample(30)) expect(WRONG_WORDS[wrongWord(t)]).not.toContain("Sesame");
    expect(new Set(sample(30).map(wrongWord)).size).toBe(WRONG_WORDS.length);
  });
});

describe("the jars", () => {
  test("a jar Morgiana has passed stays sealed", () => {
    for (const t of sample(JAR_LOOP)) {
      JAR_X.forEach((x, i) => {
        if (lanternX(t) > x + 0.5) expect(lidLift(t, i)).toBe(0);
      });
    }
  });

  test("before she arrives the thieves peek out of the jars", () => {
    const peeks = sample(2).some((t) => JAR_X.some((_, i) => lidLift(t, i) > 0.5));
    expect(peeks).toBe(true);
  });
});
