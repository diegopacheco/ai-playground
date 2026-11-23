type User = { readonly id: string };

export function load(raw: string): User {
  const parsed = JSON.parse(raw);
  return parsed as User;
}
