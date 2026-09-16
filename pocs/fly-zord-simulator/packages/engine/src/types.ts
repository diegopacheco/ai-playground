export type Side = "left" | "right";

export type MoveId = "jab" | "saber" | "missiles" | "guard" | "charge";

export type MoveKind = "attack" | "defend" | "recharge";

export interface Move {
  readonly id: MoveId;
  readonly name: string;
  readonly kind: MoveKind;
  readonly energy: number;
  readonly power: number;
  readonly frames: number;
}

export interface Pilot {
  readonly name: string;
  readonly kind: "human" | "fly";
  readonly provider: string;
  readonly model: string;
}

export interface Zord {
  readonly name: string;
  readonly palette: string;
  readonly hp: number;
  readonly energy: number;
  readonly guarding: boolean;
  readonly pilot: Pilot;
}

export interface TurnEvent {
  readonly turn: number;
  readonly side: Side;
  readonly move: MoveId;
  readonly damage: number;
  readonly critical: boolean;
  readonly blocked: boolean;
  readonly energyAfter: number;
  readonly taunt: string;
}

export interface Battle {
  readonly seed: number;
  readonly turn: number;
  readonly active: Side;
  readonly left: Zord;
  readonly right: Zord;
  readonly log: readonly TurnEvent[];
  readonly winner: Side | "none";
}
