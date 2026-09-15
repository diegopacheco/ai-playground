#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <stdint.h>

#include "config.h"

typedef struct {
    uint8_t pixels[SCREEN_W * SCREEN_H];
} Framebuffer;

void fb_clear(Framebuffer *fb, uint8_t color);
void fb_put(Framebuffer *fb, int x, int y, uint8_t color);
uint8_t fb_get(const Framebuffer *fb, int x, int y);
void fb_fill_rect(Framebuffer *fb, int x, int y, int width, int height, uint8_t color);
void fb_dither_rect(Framebuffer *fb, int x, int y, int width, int height, uint8_t color);
void fb_to_argb(const Framebuffer *fb, uint32_t *out);

#endif
