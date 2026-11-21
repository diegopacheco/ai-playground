import type { Cents } from "../../money";
import type { User } from "../people/types";

export type Charge = { readonly userId: string; readonly amount: Cents };

export type Meeting = {
  readonly id: string;
  readonly title: string;
  readonly organizerId: string;
  readonly attendeeIds: readonly string[];
  readonly start: Date;
  readonly end: Date;
  readonly charges: readonly Charge[];
  readonly totalCost: Cents;
};

export type MeetingRequest = {
  readonly title: string;
  readonly organizerId: string;
  readonly attendeeIds: readonly string[];
  readonly start: Date;
  readonly end: Date;
};

export type QuoteLine = {
  readonly user: User;
  readonly cost: Cents;
  readonly balanceBefore: Cents;
  readonly balanceAfter: Cents;
  readonly allowed: boolean;
};

export type Quote = {
  readonly minutes: number;
  readonly lines: readonly QuoteLine[];
  readonly totalCost: Cents;
  readonly allowed: boolean;
};

export function durationMinutes(m: { readonly start: Date; readonly end: Date }): number {
  return Math.round((m.end.getTime() - m.start.getTime()) / 60000);
}
