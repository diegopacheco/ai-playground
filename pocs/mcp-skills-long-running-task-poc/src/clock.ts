export type Clock = { readonly now: () => Date };
export type IdGenerator = () => string;

export const systemClock: Clock = { now: () => new Date() };
export const randomIds: IdGenerator = () => crypto.randomUUID();
