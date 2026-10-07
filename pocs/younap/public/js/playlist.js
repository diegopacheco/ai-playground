export function formatTime(seconds) {
  const total = Math.max(0, Math.floor(seconds || 0));
  const minutes = String(Math.floor(total / 60)).padStart(2, '0');
  return `${minutes}:${String(total % 60).padStart(2, '0')}`;
}

export function totalSeconds(tracks) {
  return tracks.reduce((sum, track) => sum + track.seconds, 0);
}

export function nextIndex(current, length, { shuffle, repeat }, random = Math.random) {
  if (!length) return -1;
  if (shuffle && length > 1) {
    if (current < 0) return Math.floor(random() * length);
    const pick = Math.floor(random() * (length - 1));
    return pick >= current ? pick + 1 : pick;
  }
  if (current + 1 < length) return current + 1;
  return repeat ? 0 : -1;
}

export function previousIndex(current, length) {
  if (!length) return -1;
  return current > 0 ? current - 1 : length - 1;
}

export function removeTrack(tracks, current, index) {
  if (index < 0 || index >= tracks.length) return { tracks, current };
  const remaining = tracks.filter((_, position) => position !== index);
  if (index === current) return { tracks: remaining, current: -1 };
  return { tracks: remaining, current: index < current ? current - 1 : current };
}

export function scrollText(text, offset, width) {
  if (text.length <= width) return text;
  const loop = `${text}  ***  `;
  const start = offset % loop.length;
  return (loop + loop).slice(start, start + width);
}
