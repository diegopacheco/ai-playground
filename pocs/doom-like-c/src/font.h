#ifndef FONT_H
#define FONT_H

#include <stdint.h>

#include "framebuffer.h"

#define FONT_GLYPH_W 5
#define FONT_GLYPH_H 7

void font_draw(Framebuffer *fb, int x, int y, const char *text, int scale, uint8_t color);
void font_draw_shadow(Framebuffer *fb, int x, int y, const char *text, int scale, uint8_t color, uint8_t shadow);
void font_draw_centered(Framebuffer *fb, int y, const char *text, int scale, uint8_t color, uint8_t shadow);
int font_width(const char *text, int scale);

#endif
