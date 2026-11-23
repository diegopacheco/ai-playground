type User = { readonly active: boolean; readonly email: string };

export function loadUsers(): User[] {
  return [{ active: true, email: "a@x.io" }];
}
