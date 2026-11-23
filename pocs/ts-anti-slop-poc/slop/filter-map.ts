type User = { readonly active: boolean; readonly email: string };

const users: User[] = [{ active: true, email: "a@x.io" }];

export const emails = users.filter((user) => user.active).map((user) => user.email);
