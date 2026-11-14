export type SceneId =
  | "title"
  | "woodcutter"
  | "riders"
  | "sesame"
  | "treasure"
  | "kassim"
  | "chalk"
  | "merchant"
  | "jars"
  | "dance"
  | "wedding"
  | "theEnd";

export interface StoryPage {
  scene: SceneId;
  title: string;
  lines: string[];
}

export const PAGES: StoryPage[] = [
  {
    scene: "title",
    title: "Ali Baba and the Forty Thieves",
    lines: [
      "Light the lamp and hush the room.",
      "A tale from the Thousand and One Nights comes to life in shadows.",
    ],
  },
  {
    scene: "woodcutter",
    title: "The Poor Woodcutter",
    lines: [
      "In a city of Persia lived Ali Baba, a poor and honest woodcutter.",
      "Every day he cut wood in the hills and loaded it on his donkeys.",
    ],
  },
  {
    scene: "riders",
    title: "Forty Riders",
    lines: [
      "One day a cloud of dust rolled over the hills: forty riders!",
      "Ali Baba climbed a tree and held his breath.",
    ],
  },
  {
    scene: "sesame",
    title: "Open, Sesame!",
    lines: [
      "The captain of the thieves faced a wall of rock and cried, “Open, Sesame!”",
      "The great stone rolled away, and the thieves rode into the hill.",
    ],
  },
  {
    scene: "treasure",
    title: "The Treasure Cave",
    lines: [
      "When they had gone, Ali Baba said the magic words himself.",
      "Inside he found gold, jewels and silks piled up to the roof.",
    ],
  },
  {
    scene: "kassim",
    title: "Kassim Forgets the Word",
    lines: [
      "His greedy brother Kassim filled ten sacks, then forgot the word.",
      "“Open, Barley! Open, Wheat!” But the stone stayed shut.",
    ],
  },
  {
    scene: "chalk",
    title: "The Chalk Mark",
    lines: [
      "By night a thief marked Ali Baba's door with a cross of chalk.",
      "But clever Morgiana marked every door in the street the same way.",
    ],
  },
  {
    scene: "merchant",
    title: "The Oil Merchant",
    lines: [
      "The captain came back dressed as a merchant of oil.",
      "His mules carried big jars, and he asked Ali Baba for a bed.",
    ],
  },
  {
    scene: "jars",
    title: "The Jars in the Courtyard",
    lines: [
      "Morgiana took her lamp to fetch some oil. A voice whispered, “Is it time?”",
      "Thieves hid in the jars! She sealed every lid before dawn.",
    ],
  },
  {
    scene: "dance",
    title: "Morgiana's Dance",
    lines: [
      "At supper Morgiana danced with veils for the merchant.",
      "She spun, saw the dagger under his cloak, and unmasked the captain.",
    ],
  },
  {
    scene: "wedding",
    title: "The Wedding Feast",
    lines: [
      "Ali Baba gave Morgiana her freedom and his son's hand.",
      "Lanterns glowed and the whole city danced at the wedding.",
    ],
  },
  {
    scene: "theEnd",
    title: "The End",
    lines: [
      "Only Ali Baba knew the secret of the cave, and he shared it wisely.",
      "Blow out the lamp. The shadows will be here tomorrow night.",
    ],
  },
];
