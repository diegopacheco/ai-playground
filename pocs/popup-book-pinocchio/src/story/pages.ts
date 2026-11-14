export type SceneId =
  | "cover"
  | "workshop"
  | "alive"
  | "cricket"
  | "puppetShow"
  | "foxCat"
  | "field"
  | "fairy"
  | "toyland"
  | "shark"
  | "inside"
  | "theEnd";

export interface StoryPage {
  scene: SceneId;
  title: string;
  lines: string[];
}

export const PAGES: StoryPage[] = [
  {
    scene: "cover",
    title: "Pinocchio",
    lines: [
      "Tonight in the little puppet theater:",
      "the tale of a wooden boy who wished to be real.",
    ],
  },
  {
    scene: "workshop",
    title: "Geppetto's Workshop",
    lines: [
      "Once upon a time, an old carver named Geppetto found a talking log.",
      "Chip, chip, chip! He carved it into a puppet and called him Pinocchio.",
    ],
  },
  {
    scene: "alive",
    title: "The Puppet Comes Alive",
    lines: [
      "The moment the last chip fell, the puppet hopped off the bench!",
      "He kicked, he danced, and he ran laughing around the room.",
    ],
  },
  {
    scene: "cricket",
    title: "The Talking Cricket",
    lines: [
      "A wise old cricket chirped: “Listen to your father and go to school!”",
      "But Pinocchio covered his ears and only wanted to play.",
    ],
  },
  {
    scene: "puppetShow",
    title: "The Great Puppet Show",
    lines: [
      "Pinocchio sold his schoolbook to see the puppets dance.",
      "The fierce Fire-Eater sneezed AH-CHOO and gave him five gold coins.",
    ],
  },
  {
    scene: "foxCat",
    title: "The Fox and the Cat",
    lines: [
      "On the road he met a limping Fox and a Cat who pretended to be blind.",
      "“Plant your coins,” they said, “and they will grow into a tree of gold!”",
    ],
  },
  {
    scene: "field",
    title: "The Field of Miracles",
    lines: [
      "Under the moon, Pinocchio buried his coins and dreamed of riches.",
      "But the sly Fox and Cat dug them up and ran off with every one.",
    ],
  },
  {
    scene: "fairy",
    title: "The Growing Nose",
    lines: [
      "The kind Fairy with blue hair asked him where his coins were.",
      "He told a lie, and his nose grew so long the birds sat on it!",
    ],
  },
  {
    scene: "toyland",
    title: "The Land of Toys",
    lines: [
      "Pinocchio ran away to a land where nobody ever went to school.",
      "But after days of only play, he grew two long donkey ears. HEE-HAW!",
    ],
  },
  {
    scene: "shark",
    title: "The Terrible Shark",
    lines: [
      "He swam out to sea to find Geppetto, who was lost in a storm.",
      "Then a giant shark opened its jaws and GULP! swallowed him whole.",
    ],
  },
  {
    scene: "inside",
    title: "Inside the Shark",
    lines: [
      "Deep in the dark belly, a little candle was glowing. It was Geppetto!",
      "When the shark yawned, Pinocchio carried his father out to sea.",
    ],
  },
  {
    scene: "theEnd",
    title: "A Real Boy",
    lines: [
      "Pinocchio worked hard and cared for his father every day.",
      "One morning the Fairy made him a real boy. The End.",
    ],
  },
];
