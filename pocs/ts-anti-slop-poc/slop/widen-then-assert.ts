type User = { readonly id: string };

const user: User = { id: "1" };
const loose: unknown = user;

// SAFETY: it was a user a moment ago
export const back = loose as User;
