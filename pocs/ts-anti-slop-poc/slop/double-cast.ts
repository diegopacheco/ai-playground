type User = { readonly id: string };

export function toUser(input: string): User {
  // SAFETY: the payload always comes from our own api
  return input as unknown as User;
}
