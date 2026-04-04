export interface Airport {
  code: string;
  city: string;
  country: string;
  name: string;
}

export interface FlightResult {
  id: string;
  date: string;
  origin: string;
  destination: string;
  airline: string;
  price: string;
  cabin: string;
  stops: string;
  duration: string;
  source: string;
  booking_url: string;
}

export interface SearchResponse {
  results: FlightResult[];
  agent_used: string;
  raw_output: string;
  error: string | null;
}

export interface SearchRequest {
  origin: string;
  origin_city: string;
  destination: string;
  destination_city: string;
  date: string;
  agent: string;
}

export interface AgentInfo {
  id: string;
  name: string;
}
