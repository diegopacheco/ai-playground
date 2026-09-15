#ifndef PALETTE_H
#define PALETTE_H

#include <stdint.h>

#define SHADE_LEVELS 8

typedef enum {
    PAL_BLACK,
    PAL_DARK_GRAY,
    PAL_GRAY,
    PAL_LIGHT_GRAY,
    PAL_WHITE,
    PAL_DARK_RED,
    PAL_RED,
    PAL_ORANGE,
    PAL_YELLOW,
    PAL_LIGHT_YELLOW,
    PAL_DARK_BROWN,
    PAL_BROWN,
    PAL_TAN,
    PAL_DARK_GREEN,
    PAL_GREEN,
    PAL_LIGHT_GREEN,
    PAL_NAVY,
    PAL_DARK_BLUE,
    PAL_BLUE,
    PAL_SKY,
    PAL_CYAN,
    PAL_PURPLE,
    PAL_MAGENTA,
    PAL_PINK,
    PAL_DARK_TEAL,
    PAL_TEAL,
    PAL_SKIN,
    PAL_DARK_PURPLE,
    PAL_OLIVE,
    PAL_SLATE,
    PAL_STEEL,
    PAL_FOAM,
    PAL_COUNT
} PaletteColor;

void palette_init(void);
uint8_t palette_shade(uint8_t color, int level);
uint32_t palette_argb(uint8_t color);
int palette_luminance(uint8_t color);

#endif
