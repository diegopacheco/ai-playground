#include "raycast.h"

#define RAY_FAR 1e30f

static float axis_delta(float component)
{
    return component == 0.0f ? RAY_FAR : fabsf(1.0f / component);
}

RayHit raycast_cast(const Map *map, Vec2 origin, Vec2 direction, float max_distance)
{
    RayHit result = { false, max_distance, 0, MAP_EMPTY, 0.0f };
    int map_x = (int)floorf(origin.x);
    int map_y = (int)floorf(origin.y);
    float delta_x = axis_delta(direction.x);
    float delta_y = axis_delta(direction.y);
    int step_x = direction.x < 0.0f ? -1 : 1;
    int step_y = direction.y < 0.0f ? -1 : 1;
    float side_x = direction.x < 0.0f ? (origin.x - (float)map_x) * delta_x : ((float)map_x + 1.0f - origin.x) * delta_x;
    float side_y = direction.y < 0.0f ? (origin.y - (float)map_y) * delta_y : ((float)map_y + 1.0f - origin.y) * delta_y;

    for (;;) {
        int side;
        float distance;
        if (side_x < side_y) {
            distance = side_x;
            side_x += delta_x;
            map_x += step_x;
            side = 0;
        } else {
            distance = side_y;
            side_y += delta_y;
            map_y += step_y;
            side = 1;
        }
        if (distance > max_distance) {
            return result;
        }
        if (map_is_solid(map, map_x, map_y)) {
            float wall = side == 0 ? origin.y + distance * direction.y : origin.x + distance * direction.x;
            result.hit = true;
            result.distance = distance;
            result.side = side;
            result.cell = map_cell(map, map_x, map_y);
            result.wall_x = wall - floorf(wall);
            return result;
        }
    }
}

bool raycast_line_of_sight(const Map *map, Vec2 from, Vec2 to)
{
    Vec2 delta = vec2_sub(to, from);
    float length = vec2_length(delta);
    if (length <= 0.0f) {
        return true;
    }
    return !raycast_cast(map, from, vec2_scale(delta, 1.0f / length), length).hit;
}
