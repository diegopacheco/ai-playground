#ifndef MAP_H
#define MAP_H

#include <stdbool.h>
#include <stdint.h>

#define MAP_MAX 32
#define MAP_EMPTY 0

typedef struct {
    int width;
    int height;
    uint8_t cells[MAP_MAX * MAP_MAX];
} Map;

uint8_t map_cell(const Map *map, int x, int y);
bool map_is_solid(const Map *map, int x, int y);
void map_set(Map *map, int x, int y, uint8_t cell);

#endif
