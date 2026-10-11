export type Hit = {
  id: string;
  score: number;
  name: string;
  category: string;
  price: number | null;
  about: string;
  image: string;
  url: string;
};

export type Category = {
  name: string;
  count: number;
};

export type Stats = {
  products: number;
  collection: string;
};

export type Filters = {
  category: string;
  minPrice: string;
  maxPrice: string;
};

export type SearchResponse = {
  hits: Hit[];
};

export type AskResponse = {
  answer: string;
  sources: Hit[];
};

export type Tab = "ask" | "search";
