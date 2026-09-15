#include "level.h"

#include <string.h>

#include "texture.h"

static const char *const default_rows[] = {
    "########################",
    "#P.....#.......E.......#",
    "#......#...............#",
    "#..W...T....TTTT...E...#",
    "#......T....T..T.......#",
    "###.####....T..T....H..#",
    "#......#....TTTT.......#",
    "#..E...#...............#",
    "#......####MM..MM####..#",
    "#......#..........E.#..#",
    "#.H....#..W.........#..#",
    "###..###............#..#",
    "#.......M....E......G..#",
    "#.......M...........G..#",
    "#..E....MMMM..MMMMMMG..#",
    "#......................#",
    "#....GGGG......TTTT....#",
    "#....G..G..E...T..T.E..#",
    "#....G..G......T..T....#",
    "#.W..GG.G......TTTT..W.#",
    "#..........E...........#",
    "#...H.......E......H...#",
    "#.....E................#",
    "########################",
};

static uint8_t wall_cell(char code)
{
    switch (code) {
    case '#': return TEX_BRICK + 1;
    case 'T': return TEX_TILE + 1;
    case 'M': return TEX_METAL + 1;
    case 'G': return TEX_MOSS + 1;
    default: return MAP_EMPTY;
    }
}

static bool add_pickup(Level *level, PickupKind kind, Vec2 pos)
{
    if (level->pickup_count >= LEVEL_MAX_PICKUPS) {
        return false;
    }
    Pickup pickup = { kind, pos, true };
    level->pickups[level->pickup_count++] = pickup;
    return true;
}

static bool add_enemy(Level *level, Vec2 pos)
{
    if (level->enemy_count >= LEVEL_MAX_ENEMIES) {
        return false;
    }
    level->enemies[level->enemy_count++] = pos;
    return true;
}

static bool parse_cell(Level *level, char code, Vec2 center, int *players)
{
    switch (code) {
    case '#':
    case 'T':
    case 'M':
    case 'G':
    case '.':
        return true;
    case 'P':
        level->player_start = center;
        (*players)++;
        return true;
    case 'E':
        return add_enemy(level, center);
    case 'W':
        return add_pickup(level, PICKUP_WATER, center);
    case 'H':
        return add_pickup(level, PICKUP_HEALTH, center);
    default:
        return false;
    }
}

bool level_parse(Level *level, const char *const *rows, int row_count)
{
    memset(level, 0, sizeof *level);
    if (row_count <= 0 || row_count > MAP_MAX) {
        return false;
    }
    int width = (int)strlen(rows[0]);
    if (width <= 0 || width > MAP_MAX) {
        return false;
    }
    level->map.width = width;
    level->map.height = row_count;
    int players = 0;
    for (int y = 0; y < row_count; y++) {
        if ((int)strlen(rows[y]) != width) {
            return false;
        }
        for (int x = 0; x < width; x++) {
            char code = rows[y][x];
            bool border = x == 0 || y == 0 || x == width - 1 || y == row_count - 1;
            if (border && wall_cell(code) == MAP_EMPTY) {
                return false;
            }
            map_set(&level->map, x, y, wall_cell(code));
            if (!parse_cell(level, code, vec2((float)x + 0.5f, (float)y + 0.5f), &players)) {
                return false;
            }
        }
    }
    return players == 1;
}

const char *const *level_default_rows(int *row_count)
{
    *row_count = (int)(sizeof default_rows / sizeof default_rows[0]);
    return default_rows;
}
