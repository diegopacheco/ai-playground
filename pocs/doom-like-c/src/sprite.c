#include "sprite.h"

#include "palette.h"

#define SPRITE_DEF(width, rows) { (width), (int)(sizeof(rows) / sizeof((rows)[0])), (rows) }

static const char *const goblin_walk_a[] = {
    "......PPPP......",
    ".....PPPPPP.....",
    "....gGGGGGGg....",
    "....GRKGGKRG....",
    "....GGGGGGGG....",
    "....gGKKKKGg....",
    ".....gGGGGg.....",
    "...pPPPPPPPPp...",
    "..GpPPPPPPPPpG..",
    "..G.pPPPPPPp.G..",
    "..g.pPPPPPPp.g..",
    "....pPPPPPPp....",
    ".....pp..pp.....",
    ".....NN..NN.....",
    "....NNN..NNN....",
    "................",
};

static const char *const goblin_walk_b[] = {
    "......PPPP......",
    ".....PPPPPP.....",
    "....gGGGGGGg....",
    "....GRKGGKRG....",
    "....GGGGGGGG....",
    "....gGKKKKGg....",
    ".....gGGGGg.....",
    "...pPPPPPPPPp...",
    "..GpPPPPPPPPpG..",
    "..G.pPPPPPPp.G..",
    "..g.pPPPPPPp.g..",
    "....pPPPPPPp....",
    "....pp....pp....",
    "....NN....NN....",
    "...NNN....NNN...",
    "................",
};

static const char *const goblin_throw[] = {
    "......PPPP..YY..",
    ".....PPPPPPYYYO.",
    "....gGGGGGGgYY..",
    "....GRKGGKRG.G..",
    "....GGGGGGGG.G..",
    "....gGKKKKGg.G..",
    ".....gGGGGg..G..",
    "...pPPPPPPPPpG..",
    "..GpPPPPPPPPp...",
    "..G.pPPPPPPp....",
    "..g.pPPPPPPp....",
    "....pPPPPPPp....",
    ".....pp..pp.....",
    ".....NN..NN.....",
    "....NNN..NNN....",
    "................",
};

static const char *const goblin_soaked[] = {
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "......PPPP......",
    "....gGGGGGGg....",
    "...GGKGGGGKGG...",
    "..SSgGGGGGGgSS..",
    ".SBSSSSSSSSSSBS.",
    "..SSSSSSSSSSSS..",
};

static const char *const duck[] = {
    "................",
    "................",
    "......YYYY......",
    ".....YYYYYY.....",
    ".....YKYYYY.....",
    "...OOYYYYYY.....",
    "...OOOYYYY......",
    "......YYY.......",
    "..YY.YYYYYYY....",
    "..YYYYYYYYYYY...",
    "..YYYYYYYYYYYY..",
    "...YYYYYYyYYYY..",
    "...yYYYYYYYYYy..",
    "....yYYYYYYYy...",
    ".....yyyyyyy....",
    "................",
};

static const char *const water_drop[] = {
    "...SS...",
    "..SCCS..",
    ".SCWCCS.",
    ".SCCCCS.",
    "SCWCCCBS",
    "SCCCCCBS",
    ".SCCCBS.",
    "..SSSS..",
};

static const char *const splash[] = {
    "..S..S..",
    "S..CC..S",
    "..SWCS..",
    ".SCCCCS.",
    "S.SCCS.S",
    "..S..S..",
    ".S....S.",
    "........",
};

static const char *const pickup_water[] = {
    "................",
    "................",
    "................",
    "....LLLLLLLL....",
    "...L........L...",
    "...L........L...",
    "..DDDDDDDDDDDD..",
    "..DCCWCCCCCCSD..",
    "..DSSSSSSSSSSD..",
    "...DBBBBBBBBD...",
    "...DBBBBBBBBD...",
    "...DBBBBBBBBD...",
    "....DBBBBBBD....",
    "....DDDDDDDD....",
    "................",
    "................",
};

static const char *const pickup_health[] = {
    "................",
    "................",
    "................",
    "................",
    "......DDDD......",
    ".....D....D.....",
    "..WWWWWWWWWWWW..",
    "..WWWWWRRWWWWW..",
    "..WWWWWRRWWWWW..",
    "..WWWRRRRRRWWW..",
    "..WWWRRRRRRWWW..",
    "..WWWWWRRWWWWW..",
    "..WWWWWRRWWWWW..",
    "..LLLLLLLLLLLL..",
    "................",
    "................",
};

static const char *const water_gun[] = {
    "..............CC................",
    ".............DLLD...............",
    ".............DLLD...............",
    "..........BBBBBBBBBB............",
    ".........BSSSSSSSSSSB...........",
    ".........BSWSSSSSSSSB...........",
    ".........BSSSSSSSSSSB...........",
    "..........BBBBBBBBBB............",
    "...........OOOOOOOO.............",
    "..........OYYYYYYYYO............",
    ".........OYYOOOOOOYYO...........",
    ".........OYOOOOOOOOYO...........",
    ".........OOOOOOOOOOOO...........",
    "........OOOOOOOOOOOOOO..........",
    "........ORRROOOOOOORRO..........",
    "........OOOOOOOOOOOOOO..........",
    ".........OOOOOOOOOOOO...........",
    "..........kkOOOOOOkk............",
    ".........kkkkOOOOkkkk...........",
    "........kkkkkkOOkkkkkk..........",
    "........kkkkkkkkkkkkkk..........",
    ".......kkkkkkkkkkkkkkkk.........",
    ".......TkkkkkkkkkkkkkkT.........",
    "......TTkkkkkkkkkkkkkkTT........",
};

static const Sprite sprites[SPR_COUNT] = {
    [SPR_GOBLIN_WALK_A] = SPRITE_DEF(16, goblin_walk_a),
    [SPR_GOBLIN_WALK_B] = SPRITE_DEF(16, goblin_walk_b),
    [SPR_GOBLIN_THROW] = SPRITE_DEF(16, goblin_throw),
    [SPR_GOBLIN_SOAKED] = SPRITE_DEF(16, goblin_soaked),
    [SPR_DUCK] = SPRITE_DEF(16, duck),
    [SPR_WATER_DROP] = SPRITE_DEF(8, water_drop),
    [SPR_SPLASH] = SPRITE_DEF(8, splash),
    [SPR_PICKUP_WATER] = SPRITE_DEF(16, pickup_water),
    [SPR_PICKUP_HEALTH] = SPRITE_DEF(16, pickup_health),
    [SPR_WATER_GUN] = SPRITE_DEF(32, water_gun),
};

static uint8_t color_for(char code)
{
    switch (code) {
    case 'K': return PAL_BLACK;
    case 'D': return PAL_DARK_GRAY;
    case 'L': return PAL_LIGHT_GRAY;
    case 'W': return PAL_WHITE;
    case 'r': return PAL_DARK_RED;
    case 'R': return PAL_RED;
    case 'O': return PAL_ORANGE;
    case 'Y': return PAL_YELLOW;
    case 'y': return PAL_OLIVE;
    case 'N': return PAL_BROWN;
    case 'T': return PAL_TAN;
    case 'k': return PAL_SKIN;
    case 'g': return PAL_DARK_GREEN;
    case 'G': return PAL_GREEN;
    case 'P': return PAL_PURPLE;
    case 'p': return PAL_DARK_PURPLE;
    case 'B': return PAL_BLUE;
    case 'S': return PAL_SKY;
    case 'C': return PAL_CYAN;
    default: return SPRITE_CLEAR;
    }
}

const Sprite *sprite_get(SpriteId id)
{
    return &sprites[id % SPR_COUNT];
}

uint8_t sprite_pixel(SpriteId id, int x, int y)
{
    const Sprite *sprite = sprite_get(id);
    if (x < 0 || y < 0 || x >= sprite->width || y >= sprite->height) {
        return SPRITE_CLEAR;
    }
    return color_for(sprite->rows[y][x]);
}

void sprite_draw(Framebuffer *fb, SpriteId id, int x, int y, int scale, uint8_t tint)
{
    const Sprite *sprite = sprite_get(id);
    for (int row = 0; row < sprite->height; row++) {
        for (int col = 0; col < sprite->width; col++) {
            uint8_t color = sprite_pixel(id, col, row);
            if (color == SPRITE_CLEAR) {
                continue;
            }
            fb_fill_rect(fb, x + col * scale, y + row * scale, scale, scale, tint != SPRITE_CLEAR ? tint : color);
        }
    }
}
