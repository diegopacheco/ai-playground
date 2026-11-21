export type Span = { readonly id: string; readonly start: Date; readonly end: Date };

export type Placed<T extends Span> = { readonly item: T; readonly lane: number; readonly lanes: number };

export function layoutDay<T extends Span>(items: readonly T[]): readonly Placed<T>[] {
  const sorted = [...items].sort((a, b) => a.start.getTime() - b.start.getTime() || b.end.getTime() - a.end.getTime());
  const out: Placed<T>[] = [];
  let cluster: { item: T; lane: number }[] = [];
  let clusterEnd = 0;
  const flush = () => {
    const lanes = cluster.reduce((n, c) => Math.max(n, c.lane + 1), 0);
    for (const c of cluster) out.push({ item: c.item, lane: c.lane, lanes });
    cluster = [];
  };
  for (const item of sorted) {
    if (cluster.length > 0 && item.start.getTime() >= clusterEnd) flush();
    const laneEnds = new Map<number, number>();
    for (const c of cluster) laneEnds.set(c.lane, Math.max(laneEnds.get(c.lane) ?? 0, c.item.end.getTime()));
    let lane = 0;
    while ((laneEnds.get(lane) ?? 0) > item.start.getTime()) lane++;
    cluster.push({ item, lane });
    clusterEnd = Math.max(clusterEnd, item.end.getTime());
  }
  flush();
  return out;
}
