#include "map.h"

#include "texture.h"

static bool in_bounds(const Map *map, int x, int y)
{
    return x >= 0 && y >= 0 && x < map->width && y < map->height;
}

uint8_t map_cell(const Map *map, int x, int y)
{
    if (!in_bounds(map, x, y)) {
        return TEX_BRICK + 1;
    }
    return map->cells[y * MAP_MAX + x];
}

bool map_is_solid(const Map *map, int x, int y)
{
    return map_cell(map, x, y) != MAP_EMPTY;
}

void map_set(Map *map, int x, int y, uint8_t cell)
{
    if (in_bounds(map, x, y)) {
        map->cells[y * MAP_MAX + x] = cell;
    }
}
