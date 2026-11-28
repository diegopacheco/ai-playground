import { describe, expect, test } from "bun:test";
import { replyToTopic, replyToVoice } from "../src/features/phone/dialogue";
import { TALK_TOPICS } from "../src/features/phone/types";

describe("toy phone dialogue", () => {
  test("answers every picture talk topic with a playful reply and mood", () => {
    for (const topic of TALK_TOPICS) {
      const reply = replyToTopic("momo", topic);
      expect(reply.text.length).toBeGreaterThan(0);
      expect(reply.reactionEmoji.length).toBeGreaterThan(0);
    }
  });

  test("understands spoken keywords and gently echoes custom kid phrases", () => {
    const foodReply = replyToVoice("pip", "Are you hungry for cake?");
    expect(foodReply.mood).toBe("yummy");
    expect(foodReply.text).toContain("Berry Cupcake");

    const customReply = replyToVoice("momo", "rainbow dinosaur");
    expect(customReply.text).toContain("rainbow dinosaur");

    const emptyReply = replyToVoice("momo", "   ");
    expect(emptyReply.mood).toBe("dancing");
  });
});
