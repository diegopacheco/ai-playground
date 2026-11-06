export interface Agent {
  readonly name: string;
  readonly description?: string;
  readonly role?: string;
  readonly model?: string;
  readonly tools: readonly string[];
  readonly source?: string;
  readonly content?: string;
}

export interface Stage {
  readonly name: string;
  readonly description?: string;
  readonly agents: readonly string[];
  readonly next: readonly string[];
  readonly artifacts: readonly string[];
  readonly content?: string;
}

export interface Loop {
  readonly name: string;
  readonly description?: string;
  readonly from: string;
  readonly to: string;
  readonly until?: string;
  readonly max?: number;
}

export interface Factory {
  readonly name: string;
  readonly description?: string;
  readonly entry?: string;
  readonly agents: readonly Agent[];
  readonly stages: readonly Stage[];
  readonly loops: readonly Loop[];
}

export interface Factoryfile {
  readonly version: string;
  readonly factories: readonly Factory[];
}
