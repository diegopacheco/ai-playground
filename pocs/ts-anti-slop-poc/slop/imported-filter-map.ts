import { loadUsers } from "./support/users-source.ts";

export const emails = loadUsers().filter((user) => user.active).map((user) => user.email);
