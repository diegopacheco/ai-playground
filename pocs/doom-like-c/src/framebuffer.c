#include "framebuffer.h"

#include <string.h>

#include "palette.h"

void fb_clear(Framebuffer *fb, uint8_t color)
{
    memset(fb->pixels, color, sizeof fb->pixels);
}

void fb_put(Framebuffer *fb, int x, int y, uint8_t color)
{
    if (x < 0 || y < 0 || x >= SCREEN_W || y >= SCREEN_H) {
        return;
    }
    fb->pixels[y * SCREEN_W + x] = color;
}

uint8_t fb_get(const Framebuffer *fb, int x, int y)
{
    if (x < 0 || y < 0 || x >= SCREEN_W || y >= SCREEN_H) {
        return PAL_BLACK;
    }
    return fb->pixels[y * SCREEN_W + x];
}

void fb_fill_rect(Framebuffer *fb, int x, int y, int width, int height, uint8_t color)
{
    for (int row = y; row < y + height; row++) {
        for (int col = x; col < x + width; col++) {
            fb_put(fb, col, row, color);
        }
    }
}

void fb_dither_rect(Framebuffer *fb, int x, int y, int width, int height, uint8_t color)
{
    for (int row = y; row < y + height; row++) {
        for (int col = x; col < x + width; col++) {
            if ((row + col) % 2 == 0) {
                fb_put(fb, col, row, color);
            }
        }
    }
}

void fb_to_argb(const Framebuffer *fb, uint32_t *out)
{
    for (int i = 0; i < SCREEN_W * SCREEN_H; i++) {
        out[i] = palette_argb(fb->pixels[i]);
    }
}
