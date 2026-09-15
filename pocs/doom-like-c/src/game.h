#ifndef GAME_H
#define GAME_H

#include <stdbool.h>
#include <stdint.h>

#include "enemy.h"
#include "input.h"
#include "level.h"
#include "pickup.h"
#include "player.h"
#include "projectile.h"
#include "rng.h"
#include "splash.h"

#define GAME_MESSAGE_TICKS 50

typedef enum {
    GAME_TITLE,
    GAME_PLAYING,
    GAME_OVER,
    GAME_WON
} GameMode;

typedef struct {
    GameMode mode;
    Level level;
    Player player;
    Enemy enemies[LEVEL_MAX_ENEMIES];
    int enemy_count;
    Pickup pickups[LEVEL_MAX_PICKUPS];
    int pickup_count;
    ProjectilePool projectiles;
    SplashPool splashes;
    Rng rng;
    int kills;
    int tick;
    const char *message;
    int message_timer;
} Game;

bool game_init(Game *game, uint32_t seed);
bool game_load_level(Game *game, const char *const *rows, int row_count);
void game_start(Game *game);
void game_update(Game *game, const Input *input);

#endif
