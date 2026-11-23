type Item = { readonly id: string };

const items: Item[] = [{ id: "1" }];

export const ids = items.reduce((acc: string[], item) => acc.concat([item.id]), []);
