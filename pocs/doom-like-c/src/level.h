#ifndef LEVEL_H
#define LEVEL_H

#include <stdbool.h>

#include "map.h"
#include "pickup.h"
#include "vec.h"

#define LEVEL_MAX_ENEMIES 32
#define LEVEL_MAX_PICKUPS 32

typedef struct {
    Map map;
    Vec2 player_start;
    int enemy_count;
    Vec2 enemies[LEVEL_MAX_ENEMIES];
    int pickup_count;
    Pickup pickups[LEVEL_MAX_PICKUPS];
} Level;

bool level_parse(Level *level, const char *const *rows, int row_count);
const char *const *level_default_rows(int *row_count);

#endif
