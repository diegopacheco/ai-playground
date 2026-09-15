#ifndef PLAYER_H
#define PLAYER_H

#include <stdbool.h>

#include "input.h"
#include "map.h"
#include "vec.h"

#define PLAYER_MAX_HEALTH 100
#define PLAYER_MAX_WATER 99
#define PLAYER_START_WATER 60
#define PLAYER_RADIUS 0.25f
#define PLAYER_MOVE_SPEED 0.085f
#define PLAYER_TURN_SPEED 0.06f
#define PLAYER_FOV_SCALE 0.66f
#define PLAYER_FIRE_COOLDOWN 7
#define PLAYER_SHOT_TICKS 4
#define PLAYER_HURT_TICKS 8

typedef struct {
    Vec2 pos;
    float angle;
    int health;
    int water;
    int fire_cooldown;
    int shot_timer;
    int hurt_timer;
    float bob;
} Player;

void player_spawn(Player *player, Vec2 pos, float angle);
void player_update(Player *player, const Map *map, const Input *input);
bool player_try_fire(Player *player);
void player_hurt(Player *player, int damage);
Vec2 player_direction(const Player *player);
Vec2 player_camera_plane(const Player *player);

#endif
