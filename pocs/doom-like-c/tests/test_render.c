#include "check.h"

#include <string.h>

#include "font.h"
#include "game.h"
#include "palette.h"
#include "render.h"
#include "render_world.h"
#include "sprite.h"

#define ROWS(rows) (int)(sizeof(rows) / sizeof((rows)[0]))

static Game game;
static Framebuffer with_goblin;
static Framebuffer without_goblin;

static void distance_shading_keeps_near_colors_and_darkens_far_ones(void)
{
    for (int color = 0; color < PAL_COUNT; color++) {
        CHECK(palette_shade((uint8_t)color, 0) == color);
        CHECK(palette_luminance(palette_shade((uint8_t)color, SHADE_LEVELS - 1)) <= palette_luminance((uint8_t)color));
    }
    CHECK(palette_luminance(palette_shade(PAL_WHITE, SHADE_LEVELS - 1)) < palette_luminance(PAL_WHITE));
    CHECK(palette_shade(SPRITE_CLEAR, 3) == SPRITE_CLEAR);
}

static void every_sprite_row_matches_its_declared_width(void)
{
    for (int id = 0; id < SPR_COUNT; id++) {
        const Sprite *sprite = sprite_get((SpriteId)id);
        CHECK(sprite->height > 0);
        for (int row = 0; row < sprite->height; row++) {
            CHECK((int)strlen(sprite->rows[row]) == sprite->width);
        }
    }
}

static void font_width_excludes_trailing_gap(void)
{
    CHECK(font_width("", 2) == 0);
    CHECK(font_width("A", 1) == FONT_GLYPH_W);
    CHECK(font_width("AB", 2) == (FONT_GLYPH_W * 2 + 1) * 2);
}

static void goblin_straight_ahead_is_drawn_at_the_crosshair(void)
{
    static const char *const rows[] = {
        "###########",
        "#.........#",
        "#P..E.....#",
        "#.........#",
        "###########",
    };
    CHECK(game_init(&game, 4));
    CHECK(game_load_level(&game, rows, ROWS(rows)));
    game_start(&game);
    render_world(&with_goblin, &game);
    game.enemies[0].pos = vec2(-50.0f, -50.0f);
    render_world(&without_goblin, &game);
    int center_changed = 0;
    int edge_changed = 0;
    for (int y = 0; y < VIEW_H; y++) {
        center_changed += fb_get(&with_goblin, SCREEN_W / 2, y) != fb_get(&without_goblin, SCREEN_W / 2, y);
        edge_changed += fb_get(&with_goblin, 4, y) != fb_get(&without_goblin, 4, y);
    }
    CHECK(center_changed > 10);
    CHECK(edge_changed == 0);
}

static void every_screen_renders_with_the_hud_in_place(void)
{
    Input start = { 0 };
    start.start = true;
    CHECK(game_init(&game, 11));
    render_frame(&with_goblin, &game);
    CHECK(fb_get(&with_goblin, 0, 0) != PAL_SLATE);
    game_update(&game, &start);
    for (int mode = GAME_PLAYING; mode <= GAME_WON; mode++) {
        game.mode = (GameMode)mode;
        render_frame(&with_goblin, &game);
        CHECK(fb_get(&with_goblin, 1, VIEW_H + 2) == PAL_SLATE);
    }
}

void run_render_tests(void)
{
    RUN_TEST(distance_shading_keeps_near_colors_and_darkens_far_ones);
    RUN_TEST(every_sprite_row_matches_its_declared_width);
    RUN_TEST(font_width_excludes_trailing_gap);
    RUN_TEST(goblin_straight_ahead_is_drawn_at_the_crosshair);
    RUN_TEST(every_screen_renders_with_the_hud_in_place);
}
