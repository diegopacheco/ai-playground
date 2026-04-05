const spriteCache = new Map<string, HTMLImageElement>()

export function loadImage(src: string): Promise<HTMLImageElement> {
  if (spriteCache.has(src)) {
    return Promise.resolve(spriteCache.get(src)!)
  }
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      spriteCache.set(src, img)
      resolve(img)
    }
    img.onerror = reject
    img.src = src
  })
}

export async function loadCharacterSprites(): Promise<HTMLImageElement[]> {
  const sprites: HTMLImageElement[] = []
  for (let i = 0; i < 6; i++) {
    sprites.push(await loadImage(`/assets/characters/char_${i}.png`))
  }
  return sprites
}

export async function loadFloorTiles(): Promise<HTMLImageElement[]> {
  const tiles: HTMLImageElement[] = []
  for (let i = 0; i < 9; i++) {
    tiles.push(await loadImage(`/assets/floors/floor_${i}.png`))
  }
  return tiles
}

export async function loadWallTile(): Promise<HTMLImageElement> {
  return loadImage('/assets/walls/wall_0.png')
}

export async function loadFurnitureSprite(path: string): Promise<HTMLImageElement> {
  return loadImage(`/assets/furniture/${path}`)
}
