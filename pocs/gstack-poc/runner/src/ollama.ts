export interface OllamaChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
  images?: string[];
}

export interface OllamaChatRequest {
  model: string;
  messages: OllamaChatMessage[];
  stream: false;
  format?: object | "json";
  options?: {
    temperature?: number;
    num_ctx?: number;
  };
}

export interface OllamaChatResponse {
  model: string;
  message: { role: "assistant"; content: string };
  done: boolean;
  total_duration?: number;
  eval_count?: number;
  prompt_eval_count?: number;
}

export interface OllamaClient {
  chat(request: OllamaChatRequest): Promise<OllamaChatResponse>;
}

export class HttpOllamaClient implements OllamaClient {
  constructor(
    private readonly baseUrl: string,
    private readonly fetchImpl: typeof fetch = fetch,
  ) {}

  async chat(request: OllamaChatRequest): Promise<OllamaChatResponse> {
    const response = await this.fetchImpl(`${this.baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const body = await response.text();
      throw new OllamaError(
        `Ollama /api/chat failed: ${response.status} ${response.statusText} — ${body}`,
        response.status,
      );
    }
    return (await response.json()) as OllamaChatResponse;
  }
}

export class OllamaError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "OllamaError";
  }
}
