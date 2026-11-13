export type SceneId =
  | "cover"
  | "woodcutter"
  | "crumbs"
  | "lost"
  | "candyHouse"
  | "witch"
  | "cage"
  | "bone"
  | "oven"
  | "theEnd";

export interface StoryPage {
  scene: SceneId;
  title: string;
  lines: string[];
}

export const PAGES: StoryPage[] = [
  {
    scene: "cover",
    title: "João e Maria",
    lines: [
      "Um conto pop-up de dois irmãos corajosos,",
      "uma floresta escura e uma casa feita de doces.",
    ],
  },
  {
    scene: "woodcutter",
    title: "A Casa do Lenhador",
    lines: [
      "Era uma vez um lenhador pobre que morava na beira da floresta.",
      "Com ele viviam seus dois filhos, João e Maria.",
    ],
  },
  {
    scene: "crumbs",
    title: "Migalhas de Pão",
    lines: [
      "Um dia, no fundo da mata, João jogou migalhas pelo caminho.",
      "Mas os passarinhos famintos comeram todas elas. Piu, piu!",
    ],
  },
  {
    scene: "lost",
    title: "Perdidos na Floresta",
    lines: [
      "A noite caiu e os irmãos não acharam mais o caminho de casa.",
      "A coruja piscava, e Maria segurou firme a mão de João.",
    ],
  },
  {
    scene: "candyHouse",
    title: "A Casa de Doces",
    lines: [
      "De manhã, encontraram uma casinha feita de pão de mel!",
      "O telhado era de chocolate e as janelas de açúcar.",
    ],
  },
  {
    scene: "witch",
    title: "A Bruxa",
    lines: [
      "A porta se abriu e uma velhinha sorriu: “Entrem, queridos!”",
      "Mas ela era uma bruxa malvada que adorava comer crianças.",
    ],
  },
  {
    scene: "cage",
    title: "João na Gaiola",
    lines: [
      "A bruxa trancou João numa gaiola para engordá-lo.",
      "E Maria tinha que varrer, cozinhar e buscar água o dia todo.",
    ],
  },
  {
    scene: "bone",
    title: "O Osso de Galinha",
    lines: [
      "A bruxa enxergava mal e pedia: “Mostre o seu dedo!”",
      "Esperto, João mostrava um ossinho. “Ainda tão magrinho!”",
    ],
  },
  {
    scene: "oven",
    title: "O Forno",
    lines: [
      "Um dia a bruxa acendeu o forno e se curvou para olhar o fogo.",
      "Maria deu um empurrão e PLAFT! Fechou a porta do forno!",
    ],
  },
  {
    scene: "theEnd",
    title: "Felizes Para Sempre",
    lines: [
      "Com o tesouro da bruxa, os irmãos acharam o caminho de casa.",
      "O pai os abraçou, e nunca mais passaram fome. Fim.",
    ],
  },
];
