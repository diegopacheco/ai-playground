export type SceneId =
  | "cover"
  | "farewell"
  | "straw"
  | "sticks"
  | "bricks"
  | "knock"
  | "blowStraw"
  | "blowSticks"
  | "brickStands"
  | "roof"
  | "pot"
  | "theEnd";

export interface StoryPage {
  scene: SceneId;
  title: string;
  lines: string[];
}

export const PAGES: StoryPage[] = [
  {
    scene: "cover",
    title: "The Three Little Pigs",
    lines: [
      "A pop-up tale of three little pigs,",
      "three little houses, and one Big Bad Wolf.",
    ],
  },
  {
    scene: "farewell",
    title: "Off Into the World",
    lines: [
      "Once upon a time, Mother Pig hugged her three little pigs goodbye.",
      "\"Build strong homes,\" she said, \"and beware of the Big Bad Wolf!\"",
    ],
  },
  {
    scene: "straw",
    title: "The House of Straw",
    lines: [
      "The first little pig met a farmer with a cart full of straw.",
      "In one lazy afternoon he built a golden little house of straw.",
    ],
  },
  {
    scene: "sticks",
    title: "The House of Sticks",
    lines: [
      "The second little pig gathered sticks from the deep forest.",
      "Tap, tap, tap! His house of sticks was done by suppertime.",
    ],
  },
  {
    scene: "bricks",
    title: "The House of Bricks",
    lines: [
      "The third little pig worked for days, brick by heavy brick.",
      "His brothers laughed at him, but his house was built to last.",
    ],
  },
  {
    scene: "knock",
    title: "Knock, Knock",
    lines: [
      "Along came the Big Bad Wolf. \"Little pig, let me come in!\"",
      "\"Not by the hair of my chinny chin chin!\" squeaked the pig.",
    ],
  },
  {
    scene: "blowStraw",
    title: "Huff and Puff",
    lines: [
      "\"Then I'll huff, and I'll puff, and I'll blow your house in!\"",
      "WHOOSH! Straw flew everywhere and the little pig ran for his life.",
    ],
  },
  {
    scene: "blowSticks",
    title: "Crack Go the Sticks",
    lines: [
      "At the house of sticks the wolf huffed and puffed once more.",
      "CRACK! Down it came, and two little pigs raced to their brother.",
    ],
  },
  {
    scene: "brickStands",
    title: "Strong as Bricks",
    lines: [
      "The wolf huffed, and puffed, and HUFFED, and PUFFED...",
      "but the house of bricks stood strong. Not a single brick moved.",
    ],
  },
  {
    scene: "roof",
    title: "Up on the Roof",
    lines: [
      "Furious, the Big Bad Wolf climbed up onto the roof.",
      "\"I'll sneak down the chimney!\" he snarled with a wicked grin.",
    ],
  },
  {
    scene: "pot",
    title: "The Boiling Pot",
    lines: [
      "But the clever pig hung a big pot of water over the fire.",
      "SPLASH! Down fell the wolf, and up he shot, howling, to the hills.",
    ],
  },
  {
    scene: "theEnd",
    title: "Happily Ever After",
    lines: [
      "The three little pigs lived safe and happy in the house of bricks,",
      "and the Big Bad Wolf was never seen again. The End.",
    ],
  },
];
