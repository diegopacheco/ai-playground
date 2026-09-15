#include "player.h"

#include <string.h>

#include "collision.h"
#include "config.h"

static void tick_down(int *timer)
{
    if (*timer > 0) {
        (*timer)--;
    }
}

void player_spawn(Player *player, Vec2 pos, float angle)
{
    memset(player, 0, sizeof *player);
    player->pos = pos;
    player->angle = angle;
    player->health = PLAYER_MAX_HEALTH;
    player->water = PLAYER_START_WATER;
}

void player_update(Player *player, const Map *map, const Input *input)
{
    tick_down(&player->fire_cooldown);
    tick_down(&player->shot_timer);
    tick_down(&player->hurt_timer);

    if (input->turn_left) {
        player->angle -= PLAYER_TURN_SPEED;
    }
    if (input->turn_right) {
        player->angle += PLAYER_TURN_SPEED;
    }
    player->angle = fmodf(player->angle + 2.0f * PI_F, 2.0f * PI_F);

    Vec2 forward = player_direction(player);
    Vec2 right = vec2(-forward.y, forward.x);
    Vec2 move = vec2(0.0f, 0.0f);
    if (input->forward) {
        move = vec2_add(move, forward);
    }
    if (input->back) {
        move = vec2_sub(move, forward);
    }
    if (input->strafe_right) {
        move = vec2_add(move, right);
    }
    if (input->strafe_left) {
        move = vec2_sub(move, right);
    }
    if (move.x != 0.0f || move.y != 0.0f) {
        move = vec2_scale(vec2_normalize(move), PLAYER_MOVE_SPEED);
        player->pos = collision_move(map, player->pos, move, PLAYER_RADIUS);
        player->bob += 0.25f;
    }
}

bool player_try_fire(Player *player)
{
    if (player->fire_cooldown > 0 || player->water <= 0) {
        return false;
    }
    player->water--;
    player->fire_cooldown = PLAYER_FIRE_COOLDOWN;
    player->shot_timer = PLAYER_SHOT_TICKS;
    return true;
}

void player_hurt(Player *player, int damage)
{
    player->health -= damage;
    if (player->health < 0) {
        player->health = 0;
    }
    player->hurt_timer = PLAYER_HURT_TICKS;
}

Vec2 player_direction(const Player *player)
{
    return vec2_from_angle(player->angle);
}

Vec2 player_camera_plane(const Player *player)
{
    Vec2 forward = player_direction(player);
    return vec2(-forward.y * PLAYER_FOV_SCALE, forward.x * PLAYER_FOV_SCALE);
}
