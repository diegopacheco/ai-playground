export const BLOCKLIST = [
  /(^|\/)node_modules(\/|$)/,
  /(^|\/)dist(\/|$)/,
  /(^|\/)build(\/|$)/,
  /(^|\/)\.git(\/|$)/,
  /\.lock$/,
  /\.min\.js$/,
  /\.min\.css$/,
  /(^|\/)package-lock\.json$/,
  /(^|\/)yarn\.lock$/,
  /(^|\/)pnpm-lock\.yaml$/,
];

export function isBlocked(relPath: string): boolean {
  return BLOCKLIST.some(re => re.test(relPath));
}

const BINARY_EXT = new Set([
  'png', 'jpg', 'jpeg', 'gif', 'webp', 'ico', 'bmp', 'tiff', 'svg',
  'pdf', 'zip', 'gz', 'tar', '7z', 'rar', 'bz2', 'xz',
  'mp3', 'mp4', 'mov', 'avi', 'mkv', 'wav', 'ogg', 'flac',
  'woff', 'woff2', 'ttf', 'eot', 'otf',
  'exe', 'dll', 'so', 'dylib', 'class', 'jar', 'wasm',
  'pyc', 'pyo',
]);

export function looksBinaryByExt(path: string): boolean {
  const i = path.lastIndexOf('.');
  if (i < 0) return false;
  return BINARY_EXT.has(path.slice(i + 1).toLowerCase());
}

export function isBinaryBuffer(buf: Buffer): boolean {
  const n = Math.min(buf.length, 4096);
  for (let i = 0; i < n; i++) {
    if (buf[i] === 0) return true;
  }
  return false;
}
