#include "render_hud.h"

#include <stdio.h>

#include "font.h"
#include "palette.h"
#include "sprite.h"

#define GUN_SCALE 3
#define HURT_BORDER 8

void render_weapon(Framebuffer *fb, const Game *game)
{
    const Player *player = &game->player;
    const Sprite *gun = sprite_get(SPR_WATER_GUN);
    int bob_x = (int)(sinf(player->bob) * 6.0f);
    int bob_y = (int)(fabsf(cosf(player->bob)) * 4.0f);
    int recoil = player->shot_timer * 2;
    int x = SCREEN_W / 2 - gun->width * GUN_SCALE / 2 + bob_x;
    int y = VIEW_H - gun->height * GUN_SCALE + 8 + bob_y + recoil;
    sprite_draw(fb, SPR_WATER_GUN, x, y, GUN_SCALE, SPRITE_CLEAR);
    if (player->shot_timer > 0) {
        sprite_draw(fb, SPR_SPLASH, x + 15 * GUN_SCALE - 12, y - 20, GUN_SCALE, SPRITE_CLEAR);
    }
}

static void draw_stat(Framebuffer *fb, int x, int width, const char *label, const char *value, uint8_t color)
{
    font_draw(fb, x + (width - font_width(label, 1)) / 2, VIEW_H + 4, label, 1, PAL_LIGHT_GRAY);
    font_draw_shadow(fb, x + (width - font_width(value, 2)) / 2, VIEW_H + 14, value, 2, color, PAL_BLACK);
}

static void draw_panel(Framebuffer *fb, const Game *game)
{
    int section = SCREEN_W / 3;
    char text[16];
    fb_fill_rect(fb, 0, VIEW_H, SCREEN_W, HUD_H, PAL_SLATE);
    fb_fill_rect(fb, 0, VIEW_H, SCREEN_W, 1, PAL_STEEL);
    fb_fill_rect(fb, section, VIEW_H + 3, 1, HUD_H - 6, PAL_DARK_GRAY);
    fb_fill_rect(fb, section * 2, VIEW_H + 3, 1, HUD_H - 6, PAL_DARK_GRAY);
    snprintf(text, sizeof text, "%d%%", game->player.health);
    draw_stat(fb, 0, section, "HEALTH", text, PAL_RED);
    snprintf(text, sizeof text, "%d", game->player.water);
    draw_stat(fb, section, section, "WATER", text, PAL_SKY);
    snprintf(text, sizeof text, "%d/%d", game->kills, game->enemy_count);
    draw_stat(fb, section * 2, SCREEN_W - section * 2, "SOAKED", text, PAL_YELLOW);
}

static void draw_hurt_flash(Framebuffer *fb)
{
    fb_dither_rect(fb, 0, 0, SCREEN_W, HURT_BORDER, PAL_YELLOW);
    fb_dither_rect(fb, 0, VIEW_H - HURT_BORDER, SCREEN_W, HURT_BORDER, PAL_YELLOW);
    fb_dither_rect(fb, 0, HURT_BORDER, HURT_BORDER, VIEW_H - HURT_BORDER * 2, PAL_YELLOW);
    fb_dither_rect(fb, SCREEN_W - HURT_BORDER, HURT_BORDER, HURT_BORDER, VIEW_H - HURT_BORDER * 2, PAL_YELLOW);
}

void render_hud(Framebuffer *fb, const Game *game)
{
    if (game->mode == GAME_PLAYING && game->player.hurt_timer > 0) {
        draw_hurt_flash(fb);
    }
    if (game->mode == GAME_PLAYING && game->message_timer > 0 && game->message != NULL) {
        font_draw_shadow(fb, 12, 12, game->message, 1, PAL_WHITE, PAL_BLACK);
    }
    draw_panel(fb, game);
}
