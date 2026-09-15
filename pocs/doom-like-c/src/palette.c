#include "palette.h"

typedef struct {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} Rgb;

static const Rgb colors[PAL_COUNT] = {
    [PAL_BLACK] = { 0, 0, 0 },
    [PAL_DARK_GRAY] = { 29, 29, 33 },
    [PAL_GRAY] = { 83, 86, 93 },
    [PAL_LIGHT_GRAY] = { 150, 152, 160 },
    [PAL_WHITE] = { 240, 240, 232 },
    [PAL_DARK_RED] = { 110, 20, 30 },
    [PAL_RED] = { 200, 40, 45 },
    [PAL_ORANGE] = { 240, 130, 20 },
    [PAL_YELLOW] = { 255, 210, 30 },
    [PAL_LIGHT_YELLOW] = { 255, 240, 140 },
    [PAL_DARK_BROWN] = { 60, 36, 22 },
    [PAL_BROWN] = { 120, 72, 40 },
    [PAL_TAN] = { 200, 160, 110 },
    [PAL_DARK_GREEN] = { 20, 70, 40 },
    [PAL_GREEN] = { 50, 140, 60 },
    [PAL_LIGHT_GREEN] = { 140, 210, 90 },
    [PAL_NAVY] = { 14, 20, 60 },
    [PAL_DARK_BLUE] = { 30, 50, 130 },
    [PAL_BLUE] = { 40, 100, 210 },
    [PAL_SKY] = { 110, 180, 250 },
    [PAL_CYAN] = { 170, 240, 255 },
    [PAL_PURPLE] = { 100, 40, 140 },
    [PAL_MAGENTA] = { 200, 60, 170 },
    [PAL_PINK] = { 250, 160, 190 },
    [PAL_DARK_TEAL] = { 16, 70, 80 },
    [PAL_TEAL] = { 40, 140, 150 },
    [PAL_SKIN] = { 230, 180, 140 },
    [PAL_DARK_PURPLE] = { 50, 20, 70 },
    [PAL_OLIVE] = { 150, 130, 30 },
    [PAL_SLATE] = { 50, 60, 80 },
    [PAL_STEEL] = { 120, 140, 160 },
    [PAL_FOAM] = { 220, 250, 255 },
};

static uint8_t shade_table[SHADE_LEVELS][PAL_COUNT];

static int color_distance(Rgb color, int r, int g, int b)
{
    int dr = color.r - r;
    int dg = color.g - g;
    int db = color.b - b;
    return dr * dr * 3 + dg * dg * 4 + db * db * 2;
}

static uint8_t nearest_color(int r, int g, int b)
{
    uint8_t best = 0;
    int best_distance = color_distance(colors[0], r, g, b);
    for (int i = 1; i < PAL_COUNT; i++) {
        int distance = color_distance(colors[i], r, g, b);
        if (distance < best_distance) {
            best = (uint8_t)i;
            best_distance = distance;
        }
    }
    return best;
}

void palette_init(void)
{
    for (int level = 0; level < SHADE_LEVELS; level++) {
        float brightness = 1.0f - 0.9f * (float)level / (float)(SHADE_LEVELS - 1);
        for (int c = 0; c < PAL_COUNT; c++) {
            if (level == 0) {
                shade_table[level][c] = (uint8_t)c;
                continue;
            }
            shade_table[level][c] = nearest_color((int)(colors[c].r * brightness),
                (int)(colors[c].g * brightness),
                (int)(colors[c].b * brightness));
        }
    }
}

uint8_t palette_shade(uint8_t color, int level)
{
    if (color >= PAL_COUNT) {
        return color;
    }
    if (level < 0) {
        level = 0;
    }
    if (level >= SHADE_LEVELS) {
        level = SHADE_LEVELS - 1;
    }
    return shade_table[level][color];
}

uint32_t palette_argb(uint8_t color)
{
    Rgb c = colors[color % PAL_COUNT];
    return 0xFF000000u | ((uint32_t)c.r << 16) | ((uint32_t)c.g << 8) | (uint32_t)c.b;
}

int palette_luminance(uint8_t color)
{
    Rgb c = colors[color % PAL_COUNT];
    return c.r * 299 + c.g * 587 + c.b * 114;
}
