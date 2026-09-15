#ifndef RAYCAST_H
#define RAYCAST_H

#include <stdbool.h>
#include <stdint.h>

#include "map.h"
#include "vec.h"

typedef struct {
    bool hit;
    float distance;
    int side;
    uint8_t cell;
    float wall_x;
} RayHit;

RayHit raycast_cast(const Map *map, Vec2 origin, Vec2 direction, float max_distance);
bool raycast_line_of_sight(const Map *map, Vec2 from, Vec2 to);

#endif
