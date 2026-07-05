import type { Media } from "../shared/types.ts"

type Seed = Omit<Media, "inLibrary" | "watched" | "watchedAt" | "episodes"> & {
  seasons?: { season: number; titles: string[] }[]
}

export const catalog: Seed[] = [
  {
    id: "local-severance",
    providerId: "severance",
    provider: "local",
    type: "show",
    title: "Severance",
    year: 2022,
    overview: "Office workers whose memories have been surgically divided uncover the truth about the work that consumes them.",
    genres: ["Drama", "Mystery", "Science Fiction"],
    runtime: 48,
    poster: null,
    backdrop: null,
    color: "#e86d52",
    status: "Returning",
    rating: 8.7,
    seasons: [
      { season: 1, titles: ["Good News About Hell", "Half Loop", "In Perpetuity", "The You You Are", "The Grim Barbarity of Optics and Design", "Hide and Seek", "Defiant Jazz", "What's for Dinner?", "The We We Are"] },
      { season: 2, titles: ["Hello, Ms. Cobel", "Goodbye, Mrs. Selvig", "Who Is Alive?", "Woe's Hollow", "Trojan's Horse", "Attila", "Chikhai Bardo", "Sweet Vitriol", "The After Hours", "Cold Harbor"] }
    ]
  },
  {
    id: "local-bear",
    providerId: "the-bear",
    provider: "local",
    type: "show",
    title: "The Bear",
    year: 2022,
    overview: "A young chef returns to Chicago to run his family's sandwich shop and finds a crew ready to become something more.",
    genres: ["Drama", "Comedy"],
    runtime: 31,
    poster: null,
    backdrop: null,
    color: "#2f6d64",
    status: "Returning",
    rating: 8.6,
    seasons: [
      { season: 1, titles: ["System", "Hands", "Brigade", "Dogs", "Sheridan", "Ceres", "Review", "Braciole"] },
      { season: 2, titles: ["Beef", "Pasta", "Sundae", "Honeydew", "Pop", "Fishes", "Forks", "Bolognese", "Omelette", "The Bear"] }
    ]
  },
  {
    id: "local-andor",
    providerId: "andor",
    provider: "local",
    type: "show",
    title: "Andor",
    year: 2022,
    overview: "A thief becomes a revolutionary as ordinary people choose what they are willing to risk against an empire.",
    genres: ["Drama", "Science Fiction", "Adventure"],
    runtime: 43,
    poster: null,
    backdrop: null,
    color: "#a55f35",
    status: "Ended",
    rating: 8.5,
    seasons: [
      { season: 1, titles: ["Kassa", "That Would Be Me", "Reckoning", "Aldhani", "The Axe Forgets", "The Eye", "Announcement", "Narkina 5", "Nobody's Listening!", "One Way Out", "Daughter of Ferrix", "Rix Road"] }
    ]
  },
  {
    id: "local-past-lives",
    providerId: "past-lives",
    provider: "local",
    type: "movie",
    title: "Past Lives",
    year: 2023,
    overview: "Two childhood friends reunite in New York for one week and confront the paths their lives might have taken.",
    genres: ["Drama", "Romance"],
    runtime: 106,
    poster: null,
    backdrop: null,
    color: "#d9947c",
    status: "Released",
    rating: 7.8
  },
  {
    id: "local-perfect-days",
    providerId: "perfect-days",
    provider: "local",
    type: "movie",
    title: "Perfect Days",
    year: 2023,
    overview: "A quiet Tokyo caretaker finds wonder in routine, music, books and the shifting light through trees.",
    genres: ["Drama"],
    runtime: 124,
    poster: null,
    backdrop: null,
    color: "#3f7890",
    status: "Released",
    rating: 7.9
  },
  {
    id: "local-dune-two",
    providerId: "dune-part-two",
    provider: "local",
    type: "movie",
    title: "Dune: Part Two",
    year: 2024,
    overview: "Paul Atreides unites with Chani and the Fremen while seeking revenge against the conspirators who destroyed his family.",
    genres: ["Science Fiction", "Adventure"],
    runtime: 166,
    poster: null,
    backdrop: null,
    color: "#c46f42",
    status: "Released",
    rating: 8.5
  },
  {
    id: "local-holdovers",
    providerId: "the-holdovers",
    provider: "local",
    type: "movie",
    title: "The Holdovers",
    year: 2023,
    overview: "A curmudgeonly teacher remains on campus during the holidays with a grieving cook and a student with nowhere to go.",
    genres: ["Comedy", "Drama"],
    runtime: 133,
    poster: null,
    backdrop: null,
    color: "#875b45",
    status: "Released",
    rating: 8.0
  },
  {
    id: "local-mononoke",
    providerId: "princess-mononoke",
    provider: "local",
    type: "movie",
    title: "Princess Mononoke",
    year: 1997,
    overview: "A prince caught between an iron town and the gods of the forest seeks a path without hatred.",
    genres: ["Animation", "Fantasy", "Adventure"],
    runtime: 134,
    poster: null,
    backdrop: null,
    color: "#496c4f",
    status: "Released",
    rating: 8.4
  }
]
