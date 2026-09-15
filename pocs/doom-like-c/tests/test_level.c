#include "check.h"

#include "level.h"
#include "texture.h"

#define ROWS(rows) (int)(sizeof(rows) / sizeof((rows)[0]))

static void default_level_is_playable(void)
{
    Level level;
    int row_count;
    const char *const *rows = level_default_rows(&row_count);
    CHECK(level_parse(&level, rows, row_count));
    CHECK(level.enemy_count > 0);
    CHECK(level.pickup_count > 0);
    CHECK(!map_is_solid(&level.map, (int)level.player_start.x, (int)level.player_start.y));
}

static void open_border_is_rejected_so_rays_never_leave_the_map(void)
{
    static const char *const rows[] = { "#####", "#P...", "#####" };
    Level level;
    CHECK(!level_parse(&level, rows, ROWS(rows)));
}

static void ragged_rows_are_rejected(void)
{
    static const char *const rows[] = { "#####", "#P.#", "#####" };
    Level level;
    CHECK(!level_parse(&level, rows, ROWS(rows)));
}

static void exactly_one_player_start_is_required(void)
{
    static const char *const none[] = { "####", "#..#", "####" };
    static const char *const two[] = { "####", "#PP#", "####" };
    Level level;
    CHECK(!level_parse(&level, none, ROWS(none)));
    CHECK(!level_parse(&level, two, ROWS(two)));
}

static void spawns_land_on_cell_centers_with_their_textures(void)
{
    static const char *const rows[] = { "#TMG#", "#PEW#", "#H..#", "#####" };
    Level level;
    CHECK(level_parse(&level, rows, ROWS(rows)));
    CHECK_NEAR(level.player_start.x, 1.5, 0.001);
    CHECK_NEAR(level.player_start.y, 1.5, 0.001);
    CHECK(level.enemy_count == 1);
    CHECK_NEAR(level.enemies[0].x, 2.5, 0.001);
    CHECK(level.pickup_count == 2);
    CHECK(level.pickups[0].kind == PICKUP_WATER);
    CHECK(level.pickups[1].kind == PICKUP_HEALTH);
    CHECK(level.pickups[1].active);
    CHECK(map_cell(&level.map, 1, 0) == TEX_TILE + 1);
    CHECK(map_cell(&level.map, 2, 0) == TEX_METAL + 1);
    CHECK(map_cell(&level.map, 3, 0) == TEX_MOSS + 1);
    CHECK(map_cell(&level.map, 2, 1) == MAP_EMPTY);
}

static void unknown_tiles_are_rejected(void)
{
    static const char *const rows[] = { "####", "#P?#", "####" };
    Level level;
    CHECK(!level_parse(&level, rows, ROWS(rows)));
}

void run_level_tests(void)
{
    RUN_TEST(default_level_is_playable);
    RUN_TEST(open_border_is_rejected_so_rays_never_leave_the_map);
    RUN_TEST(ragged_rows_are_rejected);
    RUN_TEST(exactly_one_player_start_is_required);
    RUN_TEST(spawns_land_on_cell_centers_with_their_textures);
    RUN_TEST(unknown_tiles_are_rejected);
}
