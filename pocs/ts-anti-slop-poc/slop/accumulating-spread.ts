type Item = { readonly id: string };

const items: Item[] = [{ id: "1" }];

export const byId = items.reduce((acc, item) => ({ ...acc, [item.id]: item }), {});
