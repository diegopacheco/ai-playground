#include "check.h"

#include "collision.h"
#include "level.h"
#include "raycast.h"
#include "texture.h"

#define ROWS(rows) (int)(sizeof(rows) / sizeof((rows)[0]))

static const char *const pillar_rows[] = {
    "#######",
    "#P....#",
    "#..M..#",
    "#.....#",
    "#######",
};

static Map pillar_map(void)
{
    Level level;
    level_parse(&level, pillar_rows, ROWS(pillar_rows));
    return level.map;
}

static void ray_reports_distance_to_the_first_wall(void)
{
    Map map = pillar_map();
    RayHit hit = raycast_cast(&map, vec2(1.5f, 1.5f), vec2(1.0f, 0.0f), 64.0f);
    CHECK(hit.hit);
    CHECK_NEAR(hit.distance, 4.5, 0.001);
    CHECK(hit.cell == TEX_BRICK + 1);
    CHECK(hit.side == 0);
}

static void ray_stops_at_the_pillar_and_knows_its_texture(void)
{
    Map map = pillar_map();
    RayHit hit = raycast_cast(&map, vec2(1.5f, 2.5f), vec2(1.0f, 0.0f), 64.0f);
    CHECK(hit.hit);
    CHECK_NEAR(hit.distance, 1.5, 0.001);
    CHECK(hit.cell == TEX_METAL + 1);
    CHECK_NEAR(hit.wall_x, 0.5, 0.001);
}

static void ray_beyond_max_distance_misses(void)
{
    Map map = pillar_map();
    RayHit hit = raycast_cast(&map, vec2(1.5f, 1.5f), vec2(1.0f, 0.0f), 2.0f);
    CHECK(!hit.hit);
}

static void pillar_blocks_line_of_sight_so_goblins_cannot_throw_through_walls(void)
{
    Map map = pillar_map();
    CHECK(!raycast_line_of_sight(&map, vec2(1.5f, 2.5f), vec2(5.5f, 2.5f)));
    CHECK(raycast_line_of_sight(&map, vec2(1.5f, 1.5f), vec2(5.5f, 1.5f)));
}

static void player_cannot_walk_into_a_wall(void)
{
    Map map = pillar_map();
    Vec2 pos = collision_move(&map, vec2(1.5f, 1.5f), vec2(-1.0f, 0.0f), 0.25f);
    CHECK_NEAR(pos.x, 1.5, 0.001);
    CHECK(!collision_blocked(&map, pos, 0.25f));
}

static void player_slides_along_a_wall_instead_of_sticking(void)
{
    Map map = pillar_map();
    Vec2 pos = collision_move(&map, vec2(1.3f, 1.5f), vec2(-0.2f, 0.2f), 0.25f);
    CHECK_NEAR(pos.x, 1.3, 0.001);
    CHECK_NEAR(pos.y, 1.7, 0.001);
}

void run_world_tests(void)
{
    RUN_TEST(ray_reports_distance_to_the_first_wall);
    RUN_TEST(ray_stops_at_the_pillar_and_knows_its_texture);
    RUN_TEST(ray_beyond_max_distance_misses);
    RUN_TEST(pillar_blocks_line_of_sight_so_goblins_cannot_throw_through_walls);
    RUN_TEST(player_cannot_walk_into_a_wall);
    RUN_TEST(player_slides_along_a_wall_instead_of_sticking);
}
