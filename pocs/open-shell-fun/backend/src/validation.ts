import type { Access, Enforcement, NetworkRule } from "./types";

const NAME = /^[a-z0-9][a-z0-9-]{0,40}$/;
const RULE = /^[a-z0-9][a-z0-9_]{0,40}$/;
const HOST = /^(\*\.)?[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$/;
const BINARY = /^\/[A-Za-z0-9._\/-]+$/;
const ACCESS: Access[] = ["read-only", "read-write", "full"];
const ENFORCEMENT: Enforcement[] = ["enforce", "audit"];

export class ValidationError extends Error {}

export function requireName(value: unknown, field: string): string {
  if (typeof value !== "string" || !NAME.test(value)) {
    throw new ValidationError(`${field} must be lowercase letters, digits or dashes`);
  }
  return value;
}

export function requireRuleName(value: unknown): string {
  if (typeof value !== "string" || !RULE.test(value)) {
    throw new ValidationError("ruleName must be lowercase letters, digits or underscores");
  }
  return value;
}

export function requireCommand(value: unknown): string {
  if (typeof value !== "string" || value.trim() === "" || value.length > 2000) {
    throw new ValidationError("command must be a non empty string up to 2000 chars");
  }
  return value;
}

export function parseRule(input: unknown): NetworkRule {
  if (typeof input !== "object" || input === null) {
    throw new ValidationError("rule must be an object");
  }
  const raw = input as Record<string, unknown>;
  const port = Number(raw.port);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new ValidationError("port must be between 1 and 65535");
  }
  if (typeof raw.host !== "string" || !HOST.test(raw.host)) {
    throw new ValidationError("host must be a valid hostname");
  }
  if (typeof raw.binary !== "string" || !BINARY.test(raw.binary)) {
    throw new ValidationError("binary must be an absolute path");
  }
  if (!ACCESS.includes(raw.access as Access)) {
    throw new ValidationError(`access must be one of ${ACCESS.join(", ")}`);
  }
  if (!ENFORCEMENT.includes(raw.enforcement as Enforcement)) {
    throw new ValidationError(`enforcement must be one of ${ENFORCEMENT.join(", ")}`);
  }
  return {
    ruleName: requireRuleName(raw.ruleName),
    binary: raw.binary,
    host: raw.host,
    port,
    access: raw.access as Access,
    enforcement: raw.enforcement as Enforcement,
  };
}
