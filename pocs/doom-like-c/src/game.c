#include "game.h"

#include <string.h>

#include "combat.h"

bool game_init(Game *game, uint32_t seed)
{
    memset(game, 0, sizeof *game);
    rng_seed(&game->rng, seed);
    game->mode = GAME_TITLE;
    int row_count;
    const char *const *rows = level_default_rows(&row_count);
    return game_load_level(game, rows, row_count);
}

bool game_load_level(Game *game, const char *const *rows, int row_count)
{
    Level parsed;
    if (!level_parse(&parsed, rows, row_count)) {
        return false;
    }
    game->level = parsed;
    return true;
}

void game_start(Game *game)
{
    const Level *level = &game->level;
    player_spawn(&game->player, level->player_start, 0.0f);
    game->enemy_count = level->enemy_count;
    for (int i = 0; i < level->enemy_count; i++) {
        enemy_spawn(&game->enemies[i], level->enemies[i], &game->rng);
    }
    game->pickup_count = level->pickup_count;
    memcpy(game->pickups, level->pickups, sizeof level->pickups);
    projectiles_clear(&game->projectiles);
    splashes_clear(&game->splashes);
    game->kills = 0;
    game->message = NULL;
    game->message_timer = 0;
    game->mode = GAME_PLAYING;
}

static void show_message(Game *game, const char *text)
{
    game->message = text;
    game->message_timer = GAME_MESSAGE_TICKS;
}

static void fire_water(Game *game)
{
    Vec2 direction = player_direction(&game->player);
    Vec2 nozzle = vec2_add(game->player.pos, vec2_scale(direction, 0.3f));
    projectiles_spawn(&game->projectiles, PROJECTILE_WATER, nozzle, direction);
}

static void throw_duck(Game *game, const Enemy *enemy)
{
    Vec2 direction = vec2_normalize(vec2_sub(game->player.pos, enemy->pos));
    Vec2 hand = vec2_add(enemy->pos, vec2_scale(direction, ENEMY_RADIUS + 0.1f));
    projectiles_spawn(&game->projectiles, PROJECTILE_DUCK, hand, direction);
}

static void update_enemies(Game *game)
{
    for (int i = 0; i < game->enemy_count; i++) {
        Enemy *enemy = &game->enemies[i];
        if (enemy_update(enemy, &game->level.map, game->player.pos, &game->rng) == ENEMY_ACTION_THROW) {
            throw_duck(game, enemy);
        }
    }
}

static void collect_pickups(Game *game)
{
    for (int i = 0; i < game->pickup_count; i++) {
        Pickup *pickup = &game->pickups[i];
        if (pickup_try_collect(pickup, &game->player)) {
            show_message(game, pickup->kind == PICKUP_WATER ? "+25 WATER" : "+25 HEALTH");
        }
    }
}

static void update_playing(Game *game, const Input *input)
{
    Player *player = &game->player;
    player_update(player, &game->level.map, input);
    if (input->fire && player_try_fire(player)) {
        fire_water(game);
    } else if (input->fire && player->water == 0) {
        show_message(game, "OUT OF WATER!");
    }
    update_enemies(game);
    projectiles_update(&game->projectiles, &game->level.map, &game->splashes);
    int kills = combat_water_hits(&game->projectiles, game->enemies, game->enemy_count, &game->splashes);
    if (kills > 0) {
        game->kills += kills;
        show_message(game, "GOBLIN SOAKED!");
    }
    if (combat_duck_hits(&game->projectiles, player) > 0) {
        show_message(game, "QUACK! DUCKED!");
    }
    collect_pickups(game);
    splashes_update(&game->splashes);
    if (game->message_timer > 0) {
        game->message_timer--;
    }
    if (player->health <= 0) {
        game->mode = GAME_OVER;
    } else if (game->kills >= game->enemy_count) {
        game->mode = GAME_WON;
    }
}

void game_update(Game *game, const Input *input)
{
    game->tick++;
    if (game->mode == GAME_PLAYING) {
        update_playing(game, input);
        return;
    }
    if (input->start) {
        game_start(game);
    }
}
