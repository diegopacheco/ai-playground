#include "pickup.h"

static bool refill(int *value, int amount, int max)
{
    if (*value >= max) {
        return false;
    }
    *value = *value + amount > max ? max : *value + amount;
    return true;
}

bool pickup_try_collect(Pickup *pickup, Player *player)
{
    if (!pickup->active || vec2_distance(pickup->pos, player->pos) > PICKUP_RADIUS) {
        return false;
    }
    bool used = pickup->kind == PICKUP_WATER
        ? refill(&player->water, PICKUP_WATER_AMOUNT, PLAYER_MAX_WATER)
        : refill(&player->health, PICKUP_HEALTH_AMOUNT, PLAYER_MAX_HEALTH);
    if (used) {
        pickup->active = false;
    }
    return used;
}
