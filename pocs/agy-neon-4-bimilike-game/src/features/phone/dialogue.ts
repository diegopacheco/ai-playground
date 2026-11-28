import { getCharacterSpec } from "../characters/catalog";
import type { CharacterId } from "../characters/types";
import type { PhoneReply, TalkTopic, TopicButtonSpec } from "./types";

export const TOPIC_BUTTONS: readonly TopicButtonSpec[] = [
  { id: "hi", label: "Hi!", emoji: "👋", color: "#ff9ebb" },
  { id: "hungry", label: "Yummy?", emoji: "🧁", color: "#ffd166" },
  { id: "play", label: "Play!", emoji: "🎈", color: "#72efdd" },
  { id: "sing", label: "Sing!", emoji: "🎵", color: "#b8b8ff" },
  { id: "dance", label: "Dance!", emoji: "💃", color: "#ffb774" },
  { id: "bye", label: "Bye!", emoji: "💖", color: "#9bf6ff" },
] as const;

export function replyToTopic(characterId: CharacterId, topic: TalkTopic): PhoneReply {
  const char = getCharacterSpec(characterId);
  switch (topic) {
    case "hi":
      return {
        text: `${char.greeting}`,
        reactionEmoji: "👋✨",
        mood: "waving",
      };
    case "hungry":
      return {
        text: `Mmm! Can we bake ${char.favoriteFood} in the kitchen?`,
        reactionEmoji: "😋🧁",
        mood: "yummy",
      };
    case "play":
      return {
        text: `Yay! Let's paint the house and move the sofa!`,
        reactionEmoji: "🎨🎉",
        mood: "happy",
      };
    case "sing":
      return {
        text: `La la la! ${char.name} loves singing with you!`,
        reactionEmoji: "🎶🌟",
        mood: "dancing",
      };
    case "dance":
      return {
        text: `Wiggle wiggle! Watch me dance in the house!`,
        reactionEmoji: "💃🌈",
        mood: "dancing",
      };
    case "bye":
      return {
        text: `Bye bye best friend! Call ${char.name} again soon!`,
        reactionEmoji: "💖👋",
        mood: "waving",
      };
    default: {
      const unreachable: never = topic;
      return unreachable;
    }
  }
}

export function replyToVoice(characterId: CharacterId, spokenText: string): PhoneReply {
  const char = getCharacterSpec(characterId);
  const clean = spokenText.trim().toLowerCase();
  if (clean.includes("eat") || clean.includes("food") || clean.includes("cake") || clean.includes("hungry")) {
    return replyToTopic(characterId, "hungry");
  }
  if (clean.includes("dance") || clean.includes("jump")) {
    return replyToTopic(characterId, "dance");
  }
  if (clean.includes("sing") || clean.includes("song") || clean.includes("music")) {
    return replyToTopic(characterId, "sing");
  }
  if (clean.includes("bye")) {
    return replyToTopic(characterId, "bye");
  }
  if (clean.length > 0) {
    return {
      text: `Hehe! You said "${spokenText.trim()}"! ${char.name} loves talking with you!`,
      reactionEmoji: "💬💖",
      mood: "happy",
    };
  }
  return {
    text: `Hehe! ${char.name} heard your silly giggle!`,
    reactionEmoji: "😆✨",
    mood: "dancing",
  };
}
