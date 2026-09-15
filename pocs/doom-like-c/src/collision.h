#ifndef COLLISION_H
#define COLLISION_H

#include <stdbool.h>

#include "map.h"
#include "vec.h"

bool collision_blocked(const Map *map, Vec2 pos, float radius);
Vec2 collision_move(const Map *map, Vec2 pos, Vec2 delta, float radius);

#endif
