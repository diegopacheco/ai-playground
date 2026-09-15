#ifndef SPRITE_H
#define SPRITE_H

#include <stdint.h>

#include "framebuffer.h"

#define SPRITE_CLEAR 255

typedef enum {
    SPR_GOBLIN_WALK_A,
    SPR_GOBLIN_WALK_B,
    SPR_GOBLIN_THROW,
    SPR_GOBLIN_SOAKED,
    SPR_DUCK,
    SPR_WATER_DROP,
    SPR_SPLASH,
    SPR_PICKUP_WATER,
    SPR_PICKUP_HEALTH,
    SPR_WATER_GUN,
    SPR_COUNT
} SpriteId;

typedef struct {
    int width;
    int height;
    const char *const *rows;
} Sprite;

const Sprite *sprite_get(SpriteId id);
uint8_t sprite_pixel(SpriteId id, int x, int y);
void sprite_draw(Framebuffer *fb, SpriteId id, int x, int y, int scale, uint8_t tint);

#endif
