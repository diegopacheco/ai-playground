#ifndef ENEMY_H
#define ENEMY_H

#include <stdbool.h>

#include "map.h"
#include "rng.h"
#include "vec.h"

#define ENEMY_MAX_HEALTH 3
#define ENEMY_RADIUS 0.3f
#define ENEMY_SPEED 0.03f
#define ENEMY_SIGHT_RANGE 12.0f
#define ENEMY_KEEP_DISTANCE 2.5f
#define ENEMY_WINDUP_TICKS 12
#define ENEMY_HURT_TICKS 8
#define ENEMY_FIRST_THROW_MIN 20
#define ENEMY_FIRST_THROW_MAX 60
#define ENEMY_THROW_COOLDOWN_MIN 45
#define ENEMY_THROW_COOLDOWN_MAX 90

typedef enum {
    ENEMY_IDLE,
    ENEMY_CHASE,
    ENEMY_WINDUP,
    ENEMY_HURT,
    ENEMY_DEAD
} EnemyState;

typedef enum {
    ENEMY_ACTION_NONE,
    ENEMY_ACTION_THROW
} EnemyAction;

typedef struct {
    Vec2 pos;
    EnemyState state;
    int health;
    int timer;
    int throw_cooldown;
    int anim;
} Enemy;

void enemy_spawn(Enemy *enemy, Vec2 pos, Rng *rng);
bool enemy_can_see(const Enemy *enemy, const Map *map, Vec2 target);
EnemyAction enemy_update(Enemy *enemy, const Map *map, Vec2 target, Rng *rng);
bool enemy_soak(Enemy *enemy, int amount);

#endif
