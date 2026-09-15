#include "collision.h"

bool collision_blocked(const Map *map, Vec2 pos, float radius)
{
    int left = (int)floorf(pos.x - radius);
    int right = (int)floorf(pos.x + radius);
    int top = (int)floorf(pos.y - radius);
    int bottom = (int)floorf(pos.y + radius);
    return map_is_solid(map, left, top) || map_is_solid(map, right, top)
        || map_is_solid(map, left, bottom) || map_is_solid(map, right, bottom);
}

Vec2 collision_move(const Map *map, Vec2 pos, Vec2 delta, float radius)
{
    Vec2 moved_x = vec2(pos.x + delta.x, pos.y);
    if (!collision_blocked(map, moved_x, radius)) {
        pos = moved_x;
    }
    Vec2 moved_y = vec2(pos.x, pos.y + delta.y);
    if (!collision_blocked(map, moved_y, radius)) {
        pos = moved_y;
    }
    return pos;
}
