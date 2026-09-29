export function clusterPoints(points, radius) {
  const clusters = [];
  for (const point of points) {
    const near = clusters.find(c => Math.hypot(c.seedX - point.x, c.seedY - point.y) < radius);
    if (near) near.members.push(point);
    else clusters.push({ seedX: point.x, seedY: point.y, members: [point] });
  }
  return clusters.map(({ members }) => ({
    x: members.reduce((sum, p) => sum + p.x, 0) / members.length,
    y: members.reduce((sum, p) => sum + p.y, 0) / members.length,
    members
  }));
}
