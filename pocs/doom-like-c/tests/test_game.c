#include "check.h"

#include "game.h"

#define ROWS(rows) (int)(sizeof(rows) / sizeof((rows)[0]))

static Game game;

static void run_until_mode_changes(const Input *input, int limit)
{
    for (int tick = 0; tick < limit && game.mode == GAME_PLAYING; tick++) {
        game_update(&game, input);
    }
}

static void enter_starts_the_game_from_the_title(void)
{
    Input idle = { 0 };
    Input start = { 0 };
    start.start = true;
    CHECK(game_init(&game, 1));
    CHECK(game.mode == GAME_TITLE);
    game_update(&game, &idle);
    CHECK(game.mode == GAME_TITLE);
    game_update(&game, &start);
    CHECK(game.mode == GAME_PLAYING);
    CHECK(game.player.health == PLAYER_MAX_HEALTH);
    CHECK_NEAR(game.player.pos.x, game.level.player_start.x, 0.001);
    CHECK(game.enemy_count == game.level.enemy_count);
}

static void soaking_every_goblin_wins(void)
{
    static const char *const rows[] = { "######", "#P..E#", "######" };
    Input fire = { 0 };
    fire.fire = true;
    CHECK(game_init(&game, 5));
    CHECK(game_load_level(&game, rows, ROWS(rows)));
    game_start(&game);
    run_until_mode_changes(&fire, 400);
    CHECK(game.mode == GAME_WON);
    CHECK(game.kills == 1);
    CHECK(game.enemies[0].state == ENEMY_DEAD);
    CHECK(game.player.water == PLAYER_START_WATER - ENEMY_MAX_HEALTH);
}

static void standing_still_under_duck_fire_loses(void)
{
    static const char *const rows[] = { "#######", "#P...E#", "#######" };
    Input idle = { 0 };
    CHECK(game_init(&game, 9));
    CHECK(game_load_level(&game, rows, ROWS(rows)));
    game_start(&game);
    run_until_mode_changes(&idle, 5000);
    CHECK(game.mode == GAME_OVER);
    CHECK(game.player.health == 0);
    CHECK(game.kills == 0);
}

static void enter_after_game_over_restarts_fresh(void)
{
    static const char *const rows[] = { "#######", "#P...E#", "#######" };
    Input idle = { 0 };
    Input start = { 0 };
    start.start = true;
    CHECK(game_init(&game, 9));
    CHECK(game_load_level(&game, rows, ROWS(rows)));
    game_start(&game);
    run_until_mode_changes(&idle, 5000);
    CHECK(game.mode == GAME_OVER);
    game_update(&game, &start);
    CHECK(game.mode == GAME_PLAYING);
    CHECK(game.player.health == PLAYER_MAX_HEALTH);
    CHECK(game.player.water == PLAYER_START_WATER);
    CHECK(projectiles_active(&game.projectiles, PROJECTILE_DUCK) == 0);
    CHECK(game.enemies[0].state != ENEMY_DEAD);
}

static void empty_tank_shows_out_of_water(void)
{
    static const char *const rows[] = { "######", "#P...#", "######" };
    Input fire = { 0 };
    fire.fire = true;
    CHECK(game_init(&game, 2));
    CHECK(game_load_level(&game, rows, ROWS(rows)));
    game_start(&game);
    game.player.water = 0;
    game_update(&game, &fire);
    CHECK(projectiles_active(&game.projectiles, PROJECTILE_WATER) == 0);
    CHECK(game.message != NULL && game.message[0] == 'O');
}

void run_game_tests(void)
{
    RUN_TEST(enter_starts_the_game_from_the_title);
    RUN_TEST(soaking_every_goblin_wins);
    RUN_TEST(standing_still_under_duck_fire_loses);
    RUN_TEST(enter_after_game_over_restarts_fresh);
    RUN_TEST(empty_tank_shows_out_of_water);
}
