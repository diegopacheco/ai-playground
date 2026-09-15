#ifndef PICKUP_H
#define PICKUP_H

#include <stdbool.h>

#include "player.h"
#include "vec.h"

#define PICKUP_RADIUS 0.5f
#define PICKUP_WATER_AMOUNT 25
#define PICKUP_HEALTH_AMOUNT 25

typedef enum {
    PICKUP_WATER,
    PICKUP_HEALTH
} PickupKind;

typedef struct {
    PickupKind kind;
    Vec2 pos;
    bool active;
} Pickup;

bool pickup_try_collect(Pickup *pickup, Player *player);

#endif
