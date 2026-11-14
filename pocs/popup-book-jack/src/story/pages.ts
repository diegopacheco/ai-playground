export type SceneId =
  | "cover"
  | "cottage"
  | "beans"
  | "window"
  | "beanstalk"
  | "clouds"
  | "castle"
  | "treasure"
  | "chase"
  | "theEnd";

export interface StoryPage {
  scene: SceneId;
  title: string;
  lines: string[];
}

export const PAGES: StoryPage[] = [
  {
    scene: "cover",
    title: "Jack and the Beanstalk",
    lines: [
      "A pop-up tale of a brave boy, a magic bean,",
      "and a giant who lives high above the clouds.",
    ],
  },
  {
    scene: "cottage",
    title: "A Poor Little Cottage",
    lines: [
      "Once upon a time, Jack lived with his mother in a tiny cottage.",
      "They were so poor that all they had left was Milky-White, the cow.",
    ],
  },
  {
    scene: "beans",
    title: "Five Magic Beans",
    lines: [
      "On the way to market, a strange old man stopped Jack.",
      "“Trade me your cow for these five magic beans!” he said.",
    ],
  },
  {
    scene: "window",
    title: "Out the Window",
    lines: [
      "“Beans? You silly boy!” cried his mother.",
      "She threw the beans out the window and they glowed in the moonlight.",
    ],
  },
  {
    scene: "beanstalk",
    title: "The Beanstalk Grows",
    lines: [
      "By morning, a giant beanstalk had grown up and up,",
      "twisting all the way into the sky. WHOOSH!",
    ],
  },
  {
    scene: "clouds",
    title: "Above the Clouds",
    lines: [
      "Jack climbed and climbed until he was above the clouds.",
      "There, on a road of mist, stood an enormous castle.",
    ],
  },
  {
    scene: "castle",
    title: "Fee-Fi-Fo-Fum!",
    lines: [
      "The giant's wife hid Jack just as the floor began to shake.",
      "“Fee-fi-fo-fum! I smell a boy!” roared the giant.",
    ],
  },
  {
    scene: "treasure",
    title: "The Golden Hen",
    lines: [
      "When the giant fell asleep, Jack saw a hen lay golden eggs,",
      "and a magic harp that played all by itself.",
    ],
  },
  {
    scene: "chase",
    title: "The Great Chase",
    lines: [
      "Jack grabbed the hen and the harp and ran for the beanstalk.",
      "The harp cried “Master!” and the giant woke up. STOMP!",
    ],
  },
  {
    scene: "theEnd",
    title: "Happily Ever After",
    lines: [
      "Jack chopped the beanstalk and down it fell. CRASH!",
      "With the golden eggs, Jack and his mother were never poor again.",
    ],
  },
];
