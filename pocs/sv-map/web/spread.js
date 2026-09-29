export function spreadPoints(points, gap, iterations = 120) {
  const placed = points.map((p, i) => ({ ...p, anchorX: p.x, anchorY: p.y, x: p.x + Math.cos(i) * 0.01, y: p.y + Math.sin(i) * 0.01 }));
  for (let step = 0; step < iterations; step++) {
    let moved = false;
    for (let i = 0; i < placed.length; i++) {
      for (let j = i + 1; j < placed.length; j++) {
        const a = placed[i];
        const b = placed[j];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const distance = Math.hypot(dx, dy);
        if (distance >= gap) continue;
        const push = (gap - distance) / 2 / distance;
        a.x -= dx * push;
        a.y -= dy * push;
        b.x += dx * push;
        b.y += dy * push;
        moved = true;
      }
    }
    if (!moved) break;
  }
  return placed;
}
