let token: string | null = null;

export function setToken(value: string | null): void {
  const trimmed = value?.trim();
  token = trimmed ? trimmed : null;
}

export function getToken(): string | null {
  return token;
}

export function hasToken(): boolean {
  return token !== null;
}
