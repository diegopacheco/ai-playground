export type Hit = {
  url: string;
  title: string;
  published: string;
  labels: string[];
  score: number;
  text: string;
};

export type Facet = {
  name: string;
  count: number;
};

export type Stats = {
  posts: number;
  chunks: number;
  labels: Facet[];
  years: Facet[];
};

export type Filters = {
  label: string;
  year: string;
};

export type SearchResult = {
  hits: Hit[];
};

export type Answer = {
  answer: string;
  sources: Hit[];
};

export type TabName = "ask" | "search";
