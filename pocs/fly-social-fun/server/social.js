export function extractHashtags(text) {
  return [...new Set(text.match(/#[A-Za-z]\w*/g) || [])];
}

export function trendingHashtags(buzzes, limit = 6) {
  const counts = new Map();
  for (const buzz of buzzes) {
    for (const tag of buzz.hashtags) counts.set(tag, (counts.get(tag) || 0) + 1);
  }
  return [...counts.entries()]
    .map(([tag, count]) => ({ tag, count }))
    .sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag))
    .slice(0, limit);
}

export function rankFlies(flies, limit = 10) {
  return flies
    .filter((fly) => fly.alive)
    .sort((a, b) => b.followers.size - a.followers.size || b.likes - a.likes || a.name.localeCompare(b.name))
    .slice(0, limit);
}
