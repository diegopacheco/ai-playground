#include "render_screens.h"

#include <stdio.h>

#include "font.h"
#include "palette.h"
#include "sprite.h"

#define WATER_LINE 118

static void draw_bubbles(Framebuffer *fb, int tick)
{
    for (int i = 0; i < 24; i++) {
        int x = (i * 53 + 17) % SCREEN_W;
        int y = WATER_LINE - ((i * 37 + tick) % WATER_LINE);
        fb_put(fb, x, y, i % 3 == 0 ? PAL_SKY : PAL_DARK_BLUE);
    }
}

static void draw_waves(Framebuffer *fb, int tick)
{
    for (int x = 0; x < SCREEN_W; x++) {
        float wave = sinf((float)x * 0.08f + (float)tick * 0.15f) * 3.0f + sinf((float)x * 0.03f - (float)tick * 0.05f) * 2.0f;
        int surface = WATER_LINE + (int)wave;
        for (int y = surface; y < SCREEN_H; y++) {
            int depth = y - surface;
            uint8_t color = depth == 0 ? PAL_FOAM : depth < 6 ? PAL_BLUE : PAL_DARK_BLUE;
            if (depth >= 6 && depth < 10 && (x + y) % 2 == 0) {
                color = PAL_BLUE;
            }
            fb_put(fb, x, y, color);
        }
    }
}

static void draw_spray(Framebuffer *fb, int tick)
{
    for (int i = 0; i < 5; i++) {
        int phase = (tick + i * 6) % 30;
        int x = 70 + i * 10 + phase;
        int y = 96 - (phase * (30 - phase)) / 12;
        sprite_draw(fb, SPR_WATER_DROP, x, y, 1, SPRITE_CLEAR);
        sprite_draw(fb, SPR_WATER_DROP, SCREEN_W - x - 8, y, 1, SPRITE_CLEAR);
    }
}

void render_logo(Framebuffer *fb, int tick)
{
    fb_clear(fb, PAL_NAVY);
    draw_bubbles(fb, tick);
    font_draw_centered(fb, 16, GAME_NAME, 4, PAL_YELLOW, PAL_ORANGE);
    font_draw_centered(fb, 56, "WATER GUNS VS RUBBER DUCKS", 1, PAL_CYAN, PAL_BLACK);
    draw_spray(fb, tick);
    int bob = (int)(sinf((float)tick * 0.12f) * 3.0f);
    sprite_draw(fb, SPR_DUCK, SCREEN_W / 2 - 32, WATER_LINE - 54 + bob, 4, SPRITE_CLEAR);
    draw_waves(fb, tick);
}

void render_title(Framebuffer *fb, int tick)
{
    render_logo(fb, tick);
    if ((tick / 18) % 2 == 0) {
        font_draw_centered(fb, 146, "PRESS ENTER TO START", 1, PAL_WHITE, PAL_BLACK);
    }
    font_draw_centered(fb, 166, "WASD MOVE - ARROWS TURN", 1, PAL_LIGHT_GRAY, PAL_BLACK);
    font_draw_centered(fb, 178, "SPACE SHOOTS WATER - ESC QUITS", 1, PAL_LIGHT_GRAY, PAL_BLACK);
}

void render_end_overlay(Framebuffer *fb, const Game *game)
{
    bool won = game->mode == GAME_WON;
    char stats[32];
    fb_dither_rect(fb, 0, 0, SCREEN_W, VIEW_H, PAL_BLACK);
    if (won) {
        font_draw_centered(fb, 36, "ALL SOAKED!", 3, PAL_CYAN, PAL_NAVY);
        sprite_draw(fb, SPR_GOBLIN_SOAKED, SCREEN_W / 2 - 16, 64, 2, SPRITE_CLEAR);
    } else {
        font_draw_centered(fb, 36, "YOU GOT DUCKED", 3, PAL_YELLOW, PAL_DARK_RED);
        sprite_draw(fb, SPR_DUCK, SCREEN_W / 2 - 16, 64, 2, SPRITE_CLEAR);
    }
    snprintf(stats, sizeof stats, "GOBLINS SOAKED %d/%d", game->kills, game->enemy_count);
    font_draw_centered(fb, 104, stats, 1, PAL_WHITE, PAL_BLACK);
    if ((game->tick / 18) % 2 == 0) {
        font_draw_centered(fb, 124, "PRESS ENTER TO PLAY AGAIN", 1, PAL_LIGHT_YELLOW, PAL_BLACK);
    }
}
