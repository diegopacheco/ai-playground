import { describe, expect, test } from "bun:test";
import { pickVoice } from "./voices";

const MAC_VOICES = [
  { name: "Luciana", lang: "pt-BR" },
  { name: "Samantha", lang: "en-US" },
  { name: "Rishi", lang: "en-IN" },
  { name: "Thomas", lang: "fr-FR" },
];

describe("voice picking", () => {
  test("an English voice reads the book, because the text is written in English", () => {
    expect(pickVoice(MAC_VOICES)?.name).toBe("Samantha");
  });

  test("a US or UK voice beats another English accent even when that one is listed first", () => {
    expect(pickVoice([{ name: "Rishi", lang: "en-IN" }, { name: "Tessa", lang: "en_GB" }])?.name).toBe("Tessa");
  });

  test("a natural neural voice beats the stock voices because it sounds the most human", () => {
    const voices = [...MAC_VOICES, { name: "Microsoft Jenny Online (Natural) - English (United States)", lang: "en-US" }];
    expect(pickVoice(voices)?.name).toContain("Jenny");
  });

  test("the story is never read by a Portuguese or French voice, which would mangle every word", () => {
    expect(pickVoice([{ name: "Luciana", lang: "pt-BR" }, { name: "Thomas", lang: "fr-FR" }])).toBeUndefined();
  });

  test("any English voice is still used when it is the only one, so narration keeps working", () => {
    expect(pickVoice([{ name: "Rishi", lang: "en-IN" }])?.name).toBe("Rishi");
  });

  test("no voices loaded yet gives no voice, so the browser default is used", () => {
    expect(pickVoice([])).toBeUndefined();
  });

  test("the novelty voices macOS lists for English are never picked, even when they come first", () => {
    const voices = [{ name: "Zarvox", lang: "en-US" }, { name: "Bad News", lang: "en-US" }, { name: "Fred", lang: "en-US" }, { name: "Samantha", lang: "en-US" }];
    expect(pickVoice(voices)?.name).toBe("Samantha");
    expect(pickVoice(voices.slice(0, 3))).toBeUndefined();
  });

  test("a downloaded premium voice beats every stock voice", () => {
    const voices = [{ name: "Google US English", lang: "en-US" }, { name: "Ava (Premium)", lang: "en-US" }, { name: "Samantha", lang: "en-US" }];
    expect(pickVoice(voices)?.name).toBe("Ava (Premium)");
  });

  test("an Android network voice is preferred over the local one because it is the neural voice", () => {
    expect(pickVoice([{ name: "en-us-x-sfg-local", lang: "en-US" }, { name: "en-us-x-sfg-network", lang: "en-US" }])?.name).toBe("en-us-x-sfg-network");
  });
});
