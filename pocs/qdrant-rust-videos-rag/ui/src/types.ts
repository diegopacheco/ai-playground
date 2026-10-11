export type Hit = {
  id: string;
  score: number;
  video_id: string;
  title: string;
  description: string;
  url: string;
  thumbnail: string;
  duration_text: string;
  views: number;
  age_text: string;
};

export type Stats = {
  videos: number;
  collection: string;
};

export type SearchRequest = {
  query: string;
  limit?: number;
  min_minutes?: number;
  max_minutes?: number;
};

export type Answer = {
  answer: string;
  sources: Hit[];
};

export type TabName = "ask" | "search" | "similar";
