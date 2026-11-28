export const TALK_TOPICS = ["hi", "hungry", "play", "sing", "dance", "bye"] as const;
export type TalkTopic = (typeof TALK_TOPICS)[number];

export type TopicButtonSpec = {
  readonly id: TalkTopic;
  readonly label: string;
  readonly emoji: string;
  readonly color: string;
};

export type PhoneReply = {
  readonly text: string;
  readonly reactionEmoji: string;
  readonly mood: "happy" | "yummy" | "dancing" | "waving";
};
