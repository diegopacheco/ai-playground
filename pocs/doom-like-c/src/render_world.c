#include "render_world.h"

#include <stdlib.h>

#include "palette.h"
#include "raycast.h"
#include "sprite.h"
#include "texture.h"

#define HORIZON (VIEW_H / 2)
#define MAX_VIEW_DISTANCE 64.0f
#define MAX_BILLBOARDS (LEVEL_MAX_ENEMIES + LEVEL_MAX_PICKUPS + MAX_PROJECTILES + MAX_SPLASHES)

typedef struct {
    Vec2 pos;
    SpriteId sprite;
    float size;
    float elevation;
    uint8_t tint;
    float distance;
} Billboard;

typedef struct {
    Vec2 pos;
    Vec2 direction;
    Vec2 plane;
} Camera;

static float depth_buffer[SCREEN_W];

static int shade_for_distance(float distance)
{
    return (int)(distance * 0.45f);
}

static void draw_floor_and_ceiling(Framebuffer *fb, const Camera *camera)
{
    Vec2 left = vec2_sub(camera->direction, camera->plane);
    Vec2 right = vec2_add(camera->direction, camera->plane);
    for (int y = HORIZON + 1; y < VIEW_H; y++) {
        float row_distance = (0.5f * (float)VIEW_H) / (float)(y - HORIZON);
        Vec2 step = vec2_scale(vec2_sub(right, left), row_distance / (float)SCREEN_W);
        Vec2 world = vec2_add(camera->pos, vec2_scale(left, row_distance));
        int level = shade_for_distance(row_distance);
        int ceiling_y = VIEW_H - 1 - y;
        for (int x = 0; x < SCREEN_W; x++) {
            int u = (int)((world.x - floorf(world.x)) * TEX_SIZE);
            int v = (int)((world.y - floorf(world.y)) * TEX_SIZE);
            fb_put(fb, x, y, palette_shade(texture_sample(TEX_FLOOR, u, v), level));
            fb_put(fb, x, ceiling_y, palette_shade(texture_sample(TEX_CEILING, u, v), level + 1));
            world = vec2_add(world, step);
        }
    }
}

static void draw_wall_column(Framebuffer *fb, int x, const RayHit *hit)
{
    float distance = hit->distance < 0.05f ? 0.05f : hit->distance;
    int line_height = (int)((float)VIEW_H / distance);
    int start = HORIZON - line_height / 2;
    int u = (int)(hit->wall_x * TEX_SIZE);
    TextureId texture = (TextureId)(hit->cell - 1);
    int level = shade_for_distance(distance) + hit->side;
    int top = start < 0 ? 0 : start;
    int bottom = start + line_height > VIEW_H ? VIEW_H : start + line_height;
    for (int y = top; y < bottom; y++) {
        int v = (int)((float)(y - start) * TEX_SIZE / (float)line_height);
        fb_put(fb, x, y, palette_shade(texture_sample(texture, u, v), level));
    }
}

static void draw_walls(Framebuffer *fb, const Camera *camera, const Map *map)
{
    for (int x = 0; x < SCREEN_W; x++) {
        float camera_x = 2.0f * (float)x / (float)SCREEN_W - 1.0f;
        Vec2 ray = vec2_add(camera->direction, vec2_scale(camera->plane, camera_x));
        RayHit hit = raycast_cast(map, camera->pos, ray, MAX_VIEW_DISTANCE);
        depth_buffer[x] = hit.hit ? hit.distance : MAX_VIEW_DISTANCE;
        if (hit.hit) {
            draw_wall_column(fb, x, &hit);
        }
    }
}

static SpriteId goblin_sprite(const Enemy *enemy)
{
    switch (enemy->state) {
    case ENEMY_DEAD:
        return SPR_GOBLIN_SOAKED;
    case ENEMY_WINDUP:
        return SPR_GOBLIN_THROW;
    case ENEMY_CHASE:
        return (enemy->anim / 8) % 2 == 0 ? SPR_GOBLIN_WALK_A : SPR_GOBLIN_WALK_B;
    default:
        return SPR_GOBLIN_WALK_A;
    }
}

static int add_billboard(Billboard *list, int count, Vec2 pos, SpriteId sprite, float size, float elevation, uint8_t tint)
{
    Billboard billboard = { pos, sprite, size, elevation, tint, 0.0f };
    list[count] = billboard;
    return count + 1;
}

static int collect_billboards(const Game *game, Billboard *list)
{
    int count = 0;
    for (int i = 0; i < game->enemy_count; i++) {
        const Enemy *enemy = &game->enemies[i];
        uint8_t tint = enemy->state == ENEMY_HURT ? PAL_CYAN : SPRITE_CLEAR;
        count = add_billboard(list, count, enemy->pos, goblin_sprite(enemy), 0.8f, 0.0f, tint);
    }
    for (int i = 0; i < game->pickup_count; i++) {
        const Pickup *pickup = &game->pickups[i];
        if (pickup->active) {
            SpriteId sprite = pickup->kind == PICKUP_WATER ? SPR_PICKUP_WATER : SPR_PICKUP_HEALTH;
            count = add_billboard(list, count, pickup->pos, sprite, 0.45f, 0.0f, SPRITE_CLEAR);
        }
    }
    for (int i = 0; i < MAX_PROJECTILES; i++) {
        const Projectile *projectile = &game->projectiles.items[i];
        if (projectile->active) {
            bool water = projectile->kind == PROJECTILE_WATER;
            count = add_billboard(list, count, projectile->pos, water ? SPR_WATER_DROP : SPR_DUCK,
                water ? 0.1f : 0.3f, projectile_elevation(projectile), SPRITE_CLEAR);
        }
    }
    for (int i = 0; i < MAX_SPLASHES; i++) {
        const Splash *splash = &game->splashes.items[i];
        if (splash->timer > 0) {
            count = add_billboard(list, count, splash->pos, SPR_SPLASH, 0.3f, 0.3f, SPRITE_CLEAR);
        }
    }
    return count;
}

static int farther_first(const void *a, const void *b)
{
    float da = ((const Billboard *)a)->distance;
    float db = ((const Billboard *)b)->distance;
    return (da < db) - (da > db);
}

static void draw_billboard(Framebuffer *fb, const Camera *camera, const Billboard *billboard)
{
    Vec2 relative = vec2_sub(billboard->pos, camera->pos);
    float inverse = 1.0f / (camera->plane.x * camera->direction.y - camera->direction.x * camera->plane.y);
    float side = inverse * (camera->direction.y * relative.x - camera->direction.x * relative.y);
    float depth = inverse * (-camera->plane.y * relative.x + camera->plane.x * relative.y);
    if (depth <= 0.1f) {
        return;
    }
    const Sprite *sprite = sprite_get(billboard->sprite);
    float scale = (float)VIEW_H / depth;
    int height = (int)(billboard->size * scale);
    int width = height * sprite->width / sprite->height;
    if (height < 1 || width < 1) {
        return;
    }
    int center_x = (int)((float)SCREEN_W / 2.0f * (1.0f + side / depth));
    int bottom = (int)((float)HORIZON + scale * 0.5f - billboard->elevation * scale);
    int top = bottom - height;
    int left = center_x - width / 2;
    int level = shade_for_distance(depth);
    for (int x = left < 0 ? 0 : left; x < left + width && x < SCREEN_W; x++) {
        if (depth >= depth_buffer[x]) {
            continue;
        }
        int u = (x - left) * sprite->width / width;
        for (int y = top < 0 ? 0 : top; y < bottom && y < VIEW_H; y++) {
            uint8_t color = sprite_pixel(billboard->sprite, u, (y - top) * sprite->height / height);
            if (color == SPRITE_CLEAR) {
                continue;
            }
            if (billboard->tint != SPRITE_CLEAR && (x + y) % 2 == 0) {
                color = billboard->tint;
            }
            fb_put(fb, x, y, palette_shade(color, level));
        }
    }
}

static void draw_billboards(Framebuffer *fb, const Camera *camera, const Game *game)
{
    static Billboard billboards[MAX_BILLBOARDS];
    int count = collect_billboards(game, billboards);
    for (int i = 0; i < count; i++) {
        billboards[i].distance = vec2_distance(billboards[i].pos, camera->pos);
    }
    qsort(billboards, (size_t)count, sizeof billboards[0], farther_first);
    for (int i = 0; i < count; i++) {
        draw_billboard(fb, camera, &billboards[i]);
    }
}

void render_world(Framebuffer *fb, const Game *game)
{
    Camera camera = { game->player.pos, player_direction(&game->player), player_camera_plane(&game->player) };
    fb_fill_rect(fb, 0, 0, SCREEN_W, VIEW_H, PAL_BLACK);
    draw_floor_and_ceiling(fb, &camera);
    draw_walls(fb, &camera, &game->level.map);
    draw_billboards(fb, &camera, game);
}
