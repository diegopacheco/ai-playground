import { describe, expect, test } from "bun:test";
import { pickVoice } from "./voices";

const MAC_VOICES = [
  { name: "Albert", lang: "en-US" },
  { name: "Bad News", lang: "en-US" },
  { name: "Zarvox", lang: "en-US" },
  { name: "Daniel", lang: "en-GB" },
  { name: "Samantha", lang: "en-US" },
  { name: "Thomas", lang: "fr-FR" },
];

describe("voice picking", () => {
  test("a robotic novelty voice is never picked even when it is listed first", () => {
    expect(pickVoice(MAC_VOICES)?.name).toBe("Samantha");
  });

  test("a downloaded premium or neural voice beats the stock voices because it sounds the most human", () => {
    const voices = [...MAC_VOICES, { name: "Ava (Premium)", lang: "en-US" }, { name: "Google US English", lang: "en-US" }];
    expect(pickVoice(voices)?.name).toBe("Ava (Premium)");
  });

  test("Microsoft natural online voices win on Edge", () => {
    const voices = [
      { name: "Microsoft David - English (United States)", lang: "en-US" },
      { name: "Microsoft Aria Online (Natural) - English (United States)", lang: "en-US" },
    ];
    expect(pickVoice(voices)?.name).toContain("Aria");
  });

  test("a story in English is never read by a voice of another language", () => {
    expect(pickVoice([{ name: "Thomas", lang: "fr-FR" }])).toBeUndefined();
  });

  test("an unknown English voice is still used when it is the only one, so narration keeps working", () => {
    expect(pickVoice([{ name: "Fiona", lang: "en_GB" }])?.name).toBe("Fiona");
  });

  test("no voices loaded yet gives no voice, so the browser default is used", () => {
    expect(pickVoice([])).toBeUndefined();
  });
});
