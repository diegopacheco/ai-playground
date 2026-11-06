export interface Agent {
  readonly name: string;
  readonly description?: string;
  readonly role?: string;
  readonly provider?: string;
  readonly model?: string;
  readonly tools: readonly string[];
  readonly source?: string;
  readonly content?: string;
}

export interface Stage {
  readonly name: string;
  readonly label?: string;
  readonly description?: string;
  readonly agents: readonly string[];
  readonly run?: string;
  readonly script?: string;
  readonly next: readonly string[];
  readonly artifacts: readonly string[];
  readonly output?: string;
  readonly check?: string;
  readonly verdict?: Readonly<Record<string, string>>;
  readonly source?: string;
  readonly content?: string;
  readonly skills: readonly string[];
  readonly tasks?: string;
  readonly parallel?: number;
  readonly visits?: number;
  readonly end?: string;
  readonly at: readonly number[];
  readonly routes: readonly Route[];
}

export interface Route {
  readonly outcome: string;
  readonly to: string;
  readonly label?: string;
  readonly reason?: string;
  readonly cycle: boolean;
  readonly exhausted?: Exhausted;
}

export interface Exhausted {
  readonly to: string;
  readonly label?: string;
  readonly reason?: string;
  readonly exhausted?: Exhausted;
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
  readonly title?: string;
  readonly description?: string;
  readonly entry?: string;
  readonly flow?: string;
  readonly backlog?: string;
  readonly agents: readonly Agent[];
  readonly stages: readonly Stage[];
  readonly loops: readonly Loop[];
}

export interface Factoryfile {
  readonly version: string;
  readonly factories: readonly Factory[];
}
