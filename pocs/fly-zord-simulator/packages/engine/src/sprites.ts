export type Palette = Readonly<Record<string, string>>;

export interface Sprite {
  readonly width: number;
  readonly height: number;
  readonly rows: readonly string[];
}

function sprite(rows: readonly string[]): Sprite {
  const width = rows[0]?.length ?? 0;
  for (const row of rows) if (row.length !== width) throw new Error("every sprite row must share one width");
  return { width, height: rows.length, rows };
}

export const ZORD_PALETTES: Readonly<Record<string, Palette>> = {
  crimson: { H: "#b81d2e", B: "#e03c3c", A: "#f2a03d", S: "#6b0f1a", V: "#ffe14d", W: "#e8e8f0", F: "#f2a03d", C: "#3de0e0" },
  cobalt: { H: "#1d3fb8", B: "#3c7ce0", A: "#3de0a0", S: "#0f1a6b", V: "#ffe14d", W: "#e8e8f0", F: "#3de0a0", C: "#ff5ad0" }
};

export const ZORD_IDLE = sprite([
  "....HHHHHHHH....",
  "...HHHHHHHHHH...",
  "...HVVVVVVVVH...",
  "...HHWWWWWWHH...",
  "....HHHHHHHH....",
  "......SSSS......",
  "..AAABBBBBBAAA..",
  ".AAAABBBBBBAAAA.",
  ".AAAABBCCBBAAAA.",
  ".SSAABBCCBBAASS.",
  "...AABBBBBBAA...",
  "...AABBBBBBAA...",
  "...SSBBBBBBSS...",
  "...FF.BBBB.FF...",
  "......BBBB......",
  ".....SSSSSS.....",
  ".....BB..BB.....",
  ".....BB..BB.....",
  ".....SS..SS.....",
  ".....BB..BB.....",
  "....WWW..WWW....",
  "....WWW..WWW...."
]);

export const ZORD_ATTACK = sprite([
  "....HHHHHHHH....",
  "...HHHHHHHHHH...",
  "...HVVVVVVVVH...",
  "...HHWWWWWWHH...",
  "....HHHHHHHH....",
  "......SSSS......",
  "..AAABBBBBBAAA..",
  ".AAAABBBBBBAAAA.",
  ".AAAABBCCBBAAAA.",
  "..SAABBCCBBAAAAF",
  "...AABBBBBBAAFFF",
  "...AABBBBBBAA.FF",
  "...SSBBBBBBSS...",
  "...FF.BBBB......",
  "......BBBB......",
  ".....SSSSSS.....",
  "....BB....BB....",
  "....BB.....BB...",
  "...SS.......BB..",
  "...BB.......SS..",
  "..WWW.......BB..",
  "..WWW......WWW.."
]);

export const ZORD_GUARD = sprite([
  "....HHHHHHHH....",
  "...HHHHHHHHHH...",
  "...HVVVVVVVVH...",
  "...HHWWWWWWHH...",
  "....HHHHHHHH....",
  "......SSSS......",
  "..AAABBBBBBAAA..",
  ".AAAABBBBBBAAAA.",
  ".AAAABBCCBBAAAA.",
  ".SSAABBCCBBAASS.",
  "...AAFFFFFFAA...",
  "...AAFFFFFFAA...",
  "...SSFFFFFFSS...",
  "....F.BBBB.F....",
  "......BBBB......",
  ".....SSSSSS.....",
  ".....BB..BB.....",
  "....BBB..BBB....",
  "....SS....SS....",
  "....BB....BB....",
  "...WWW....WWW...",
  "...WWW....WWW..."
]);

export const FLY_PALETTE: Palette = {
  E: "#ff3b3b", e: "#ffb0b0", B: "#3a3a4a", D: "#1a1a24", W: "#bcd8ff", w: "#e8f2ff",
  L: "#7a6a4a", G: "#2be06a", H: "#f2a03d", S: "#6b6b7a"
};

export const FLY_CALM = sprite([
  "..WWW........WWW..",
  ".WwwwW......WwwwW.",
  "..WWWW......WWWW..",
  "....DDDDDDDDDD....",
  "...DEEEEDDEEEED...",
  "...DEeEEDDEEeED...",
  "...DEEEEDDEEEED...",
  "....DDDSSSSDDD....",
  "...LDDDDDDDDDDL...",
  "..LLDBBBBBBBBDLL..",
  "..L.DBBGGGGBBD.L..",
  "....DBBBBBBBBD....",
  "....DDSSSSSSDD....",
  ".....D.D..D.D.....",
  "....HH.HH.HH.HH..."
]);

export const FLY_BUZZ = sprite([
  "...WWWW....WWWW...",
  "..WwwwwW..WwwwwW..",
  "...WWWW....WWWW...",
  "....DDDDDDDDDD....",
  "...DEEEEDDEEEED...",
  "...DEEeEDDEeEED...",
  "...DEEEEDDEEEED...",
  "....DDDSSSSDDD....",
  "..LDDDDDDDDDDDDL..",
  ".LLLDBBBBBBBBDLLL.",
  "L...DBBGGGGBBD...L",
  "....DBBBBBBBBD....",
  "....DDSSSSSSDD....",
  "....D.D....D.D....",
  "...HH.HH..HH.HH..."
]);

export const RANGER_PALETTE: Palette = {
  H: "#ffffff", V: "#1a1a24", B: "#e03c3c", A: "#ffe14d", S: "#6b0f1a", L: "#7a6a4a", G: "#2be06a"
};

export const RANGER_CALM = sprite([
  "......HHHHHH......",
  ".....HHHHHHHH.....",
  ".....HVVVVVVH.....",
  ".....HVVVVVVH.....",
  ".....HHHHHHHH.....",
  "......AAAAAA......",
  "...SSBBBBBBBBSS...",
  "..LSBBBBBBBBBBSL..",
  "..LLBBBAAAABBBLL..",
  "..L.BBBBBBBBBB.L..",
  "....BBBGGGGBBB....",
  "....BBBBBBBBBB....",
  "....SSSSSSSSSS....",
  ".....S.S..S.S.....",
  "....SS.SS.SS.SS..."
]);

export const CITY_PALETTE: Palette = {
  d: "#141428", m: "#22224a", l: "#2e2e63", y: "#ffe14d", g: "#3de0a0", r: "#ff5ad0", s: "#0a0a18"
};

export interface Building {
  readonly x: number;
  readonly width: number;
  readonly height: number;
  readonly shade: string;
  readonly windows: readonly number[];
}

export function skyline(seed: number, width: number, count: number): Building[] {
  const shades = ["d", "m", "l"];
  const buildings: Building[] = [];
  const step = Math.max(8, Math.floor(width / count));
  for (let index = 0; index < count; index += 1) {
    const wobble = (seed + index * 37) % 13;
    const bodyWidth = step - 2 + (wobble % 5);
    const height = 24 + ((seed + index * 61) % 48);
    const windows: number[] = [];
    for (let slot = 0; slot < Math.floor((bodyWidth * height) / 26); slot += 1) if ((seed + index * 17 + slot * 29) % 3 === 0) windows.push(slot);
    buildings.push({ x: index * step, width: bodyWidth, height, shade: shades[(seed + index) % shades.length] ?? "m", windows });
  }
  return buildings;
}
