#include "texture.h"

#include "palette.h"

static uint8_t textures[TEX_COUNT][TEX_SIZE * TEX_SIZE];

static uint32_t hash2(uint32_t x, uint32_t y, uint32_t seed)
{
    uint32_t h = x * 374761393u + y * 668265263u + seed * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}

static uint8_t brick_pixel(int u, int v)
{
    int row = v / 8;
    int shifted = u + (row % 2) * 8;
    if (v % 8 == 7 || shifted % 16 == 15) {
        return PAL_GRAY;
    }
    if (hash2((uint32_t)u, (uint32_t)v, 2) % 9 == 0) {
        return PAL_BROWN;
    }
    return hash2((uint32_t)(shifted / 16), (uint32_t)row, 1) % 3 == 0 ? PAL_DARK_RED : PAL_RED;
}

static uint8_t tile_pixel(int u, int v)
{
    if (u % 8 == 0 || v % 8 == 0) {
        return PAL_LIGHT_GRAY;
    }
    if (u % 8 == 1 && v % 8 == 1) {
        return PAL_FOAM;
    }
    return ((u / 8 + v / 8) % 2) != 0 ? PAL_WHITE : PAL_SKY;
}

static uint8_t metal_pixel(int u, int v)
{
    int cu = u % 16;
    int cv = v % 16;
    if (cv == 0) {
        return PAL_SLATE;
    }
    if (cv == 1) {
        return PAL_LIGHT_GRAY;
    }
    if ((cu == 3 || cu == 12) && (cv == 4 || cv == 12)) {
        return PAL_WHITE;
    }
    if ((cu == 4 || cu == 13) && (cv == 5 || cv == 13)) {
        return PAL_SLATE;
    }
    return hash2((uint32_t)u, (uint32_t)v, 3) % 5 == 0 ? PAL_GRAY : PAL_STEEL;
}

static uint8_t moss_pixel(int u, int v)
{
    int drip = (int)(hash2((uint32_t)(u / 2), 0, 4) % 14);
    if (v < drip) {
        return hash2((uint32_t)u, (uint32_t)v, 5) % 3 == 0 ? PAL_LIGHT_GREEN : PAL_GREEN;
    }
    if (v == drip) {
        return PAL_DARK_GREEN;
    }
    if (v % 11 == 10 || (u + (v / 11) * 5) % 13 == 0) {
        return PAL_DARK_GRAY;
    }
    return hash2((uint32_t)u, (uint32_t)v, 6) % 4 == 0 ? PAL_SLATE : PAL_GRAY;
}

static uint8_t floor_pixel(int u, int v)
{
    if (u % 16 == 0 || v % 16 == 0) {
        return PAL_SLATE;
    }
    if (hash2((uint32_t)u, (uint32_t)v, 7) % 23 == 0) {
        return PAL_TEAL;
    }
    return ((u / 16 + v / 16) % 2) != 0 ? PAL_DARK_TEAL : PAL_DARK_BLUE;
}

static uint8_t ceiling_pixel(int u, int v)
{
    if (u % 16 == 0 || v % 16 == 0) {
        return PAL_BLACK;
    }
    if (u % 16 >= 7 && u % 16 <= 9 && v % 16 >= 7 && v % 16 <= 9) {
        return PAL_LIGHT_YELLOW;
    }
    return PAL_DARK_GRAY;
}

static uint8_t generate_pixel(TextureId id, int u, int v)
{
    switch (id) {
    case TEX_BRICK:
        return brick_pixel(u, v);
    case TEX_TILE:
        return tile_pixel(u, v);
    case TEX_METAL:
        return metal_pixel(u, v);
    case TEX_MOSS:
        return moss_pixel(u, v);
    case TEX_FLOOR:
        return floor_pixel(u, v);
    case TEX_CEILING:
        return ceiling_pixel(u, v);
    default:
        return PAL_BLACK;
    }
}

void textures_init(void)
{
    for (int id = 0; id < TEX_COUNT; id++) {
        for (int v = 0; v < TEX_SIZE; v++) {
            for (int u = 0; u < TEX_SIZE; u++) {
                textures[id][v * TEX_SIZE + u] = generate_pixel((TextureId)id, u, v);
            }
        }
    }
}

uint8_t texture_sample(TextureId id, int u, int v)
{
    return textures[id % TEX_COUNT][(v & (TEX_SIZE - 1)) * TEX_SIZE + (u & (TEX_SIZE - 1))];
}
