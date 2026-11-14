export type SceneId = "riverbank" | "rabbit" | "fall" | "drinkMe" | "eatMe" | "caterpillar" | "cheshire" | "teaParty" | "croquet" | "waking";

export interface StoryPage {
  scene: SceneId;
  title: string;
  lines: string[];
}

export const PAGES: StoryPage[] = [
  {
    scene: "riverbank",
    title: "Alice's Adventures in Wonderland",
    lines: [
      "Alice was getting very tired of sitting by her sister on the bank.",
      "Her sister's book had no pictures and no conversations in it at all.",
    ],
  },
  {
    scene: "rabbit",
    title: "The White Rabbit",
    lines: [
      "Then a White Rabbit with pink eyes ran close by her.",
      "He took a watch out of his waistcoat pocket and cried, “I shall be late!”",
    ],
  },
  {
    scene: "fall",
    title: "Down the Rabbit Hole",
    lines: [
      "Alice followed him down the rabbit hole, and down, and down she fell.",
      "Past cupboards, maps and jars of marmalade, slowly, as if she were floating.",
    ],
  },
  {
    scene: "drinkMe",
    title: "Drink Me",
    lines: [
      "In a long hall she found a little bottle marked “DRINK ME”.",
      "She drank it and shut up like a telescope, small enough for the tiny door.",
    ],
  },
  {
    scene: "eatMe",
    title: "Eat Me",
    lines: [
      "Then she ate a little cake marked “EAT ME” and began to grow.",
      "“Curiouser and curiouser!” She grew until her head struck the roof.",
    ],
  },
  {
    scene: "caterpillar",
    title: "Advice from a Caterpillar",
    lines: [
      "On top of a great mushroom sat a blue Caterpillar.",
      "He looked at her a long while, then asked, “Who are you?”",
    ],
  },
  {
    scene: "cheshire",
    title: "The Cheshire Cat",
    lines: [
      "A Cat grinned at her from the bough of a tree.",
      "It vanished slowly, until only the grin remained.",
    ],
  },
  {
    scene: "teaParty",
    title: "A Mad Tea-Party",
    lines: [
      "The Hatter and the March Hare were having tea, a Dormouse asleep between them.",
      "“No room! No room!” they cried, and everyone moved one place on.",
    ],
  },
  {
    scene: "croquet",
    title: "The Queen's Croquet",
    lines: [
      "Card gardeners were painting the white roses red.",
      "The Queen played croquet with flamingos and hedgehogs. “Off with their heads!”",
    ],
  },
  {
    scene: "waking",
    title: "Only a Pack of Cards",
    lines: [
      "“You're nothing but a pack of cards!” The whole pack flew up into the air.",
      "Alice woke on the bank, with dead leaves falling softly on her face.",
    ],
  },
];
