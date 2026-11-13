import { describe, expect, test } from "bun:test";
import { pickVoice } from "./voices";

const MAC_VOICES = [
  { name: "Samantha", lang: "en-US" },
  { name: "Joana", lang: "pt-PT" },
  { name: "Luciana", lang: "pt-BR" },
  { name: "Thomas", lang: "fr-FR" },
];

describe("voice picking", () => {
  test("a Brazilian voice reads the book, because the text is written in Brazilian Portuguese", () => {
    expect(pickVoice(MAC_VOICES)?.name).toBe("Luciana");
  });

  test("a Brazilian voice beats a European Portuguese voice even when the European one is listed first", () => {
    expect(pickVoice([{ name: "Joana", lang: "pt-PT" }, { name: "Felipe", lang: "pt_BR" }])?.name).toBe("Felipe");
  });

  test("a natural neural voice beats the stock voices because it sounds the most human", () => {
    const voices = [...MAC_VOICES, { name: "Microsoft Francisca Online (Natural) - Portuguese (Brazil)", lang: "pt-BR" }];
    expect(pickVoice(voices)?.name).toContain("Francisca");
  });

  test("the story is never read by an English or French voice, which would mangle every word", () => {
    expect(pickVoice([{ name: "Samantha", lang: "en-US" }, { name: "Thomas", lang: "fr-FR" }])).toBeUndefined();
  });

  test("a European Portuguese voice is still used when it is the only one, so narration keeps working", () => {
    expect(pickVoice([{ name: "Joana", lang: "pt-PT" }])?.name).toBe("Joana");
  });

  test("no voices loaded yet gives no voice, so the browser default is used", () => {
    expect(pickVoice([])).toBeUndefined();
  });

  test("the robotic Eloquence voices macOS ships for pt-BR are never picked, even when they are the only ones listed first", () => {
    const voices = [{ name: "Eddy (Português (Brasil))", lang: "pt-BR" }, { name: "Shelley (Português (Brasil))", lang: "pt-BR" }, { name: "Luciana", lang: "pt-BR" }];
    expect(pickVoice(voices)?.name).toBe("Luciana");
    expect(pickVoice(voices.slice(0, 2))).toBeUndefined();
  });

  test("the Google Brazilian voice beats the compact Luciana voice on Chrome because it sounds more human", () => {
    expect(pickVoice([{ name: "Luciana", lang: "pt-BR" }, { name: "Google português do Brasil", lang: "pt-BR" }])?.name).toBe("Google português do Brasil");
  });

  test("a downloaded premium Luciana beats every stock voice", () => {
    const voices = [{ name: "Google português do Brasil", lang: "pt-BR" }, { name: "Luciana (Premium)", lang: "pt-BR" }, { name: "Felipe", lang: "pt-BR" }];
    expect(pickVoice(voices)?.name).toBe("Luciana (Premium)");
  });

  test("an Android network voice is preferred over the local one because it is the neural voice", () => {
    expect(pickVoice([{ name: "pt-br-x-afs-local", lang: "pt-BR" }, { name: "pt-br-x-afs-network", lang: "pt-BR" }])?.name).toBe("pt-br-x-afs-network");
  });
});
