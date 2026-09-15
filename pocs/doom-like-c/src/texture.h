#ifndef TEXTURE_H
#define TEXTURE_H

#include <stdint.h>

#define TEX_SIZE 32

typedef enum {
    TEX_BRICK,
    TEX_TILE,
    TEX_METAL,
    TEX_MOSS,
    TEX_FLOOR,
    TEX_CEILING,
    TEX_COUNT
} TextureId;

void textures_init(void);
uint8_t texture_sample(TextureId id, int u, int v);

#endif
