#include "projectile.h"

#include <string.h>

void projectiles_clear(ProjectilePool *pool)
{
    memset(pool, 0, sizeof *pool);
}

bool projectiles_spawn(ProjectilePool *pool, ProjectileKind kind, Vec2 pos, Vec2 direction)
{
    float speed = kind == PROJECTILE_WATER ? WATER_SPEED : DUCK_SPEED;
    for (int i = 0; i < MAX_PROJECTILES; i++) {
        Projectile *projectile = &pool->items[i];
        if (!projectile->active) {
            projectile->kind = kind;
            projectile->pos = pos;
            projectile->velocity = vec2_scale(vec2_normalize(direction), speed);
            projectile->active = true;
            projectile->age = 0;
            return true;
        }
    }
    return false;
}

void projectiles_update(ProjectilePool *pool, const Map *map, SplashPool *splashes)
{
    for (int i = 0; i < MAX_PROJECTILES; i++) {
        Projectile *projectile = &pool->items[i];
        if (!projectile->active) {
            continue;
        }
        projectile->age++;
        Vec2 next = vec2_add(projectile->pos, projectile->velocity);
        if (map_is_solid(map, (int)floorf(next.x), (int)floorf(next.y))) {
            projectile->active = false;
            if (projectile->kind == PROJECTILE_WATER) {
                splashes_spawn(splashes, projectile->pos);
            }
            continue;
        }
        if (projectile->age > PROJECTILE_LIFETIME) {
            projectile->active = false;
            continue;
        }
        projectile->pos = next;
    }
}

int projectiles_active(const ProjectilePool *pool, ProjectileKind kind)
{
    int count = 0;
    for (int i = 0; i < MAX_PROJECTILES; i++) {
        count += pool->items[i].active && pool->items[i].kind == kind;
    }
    return count;
}

float projectile_elevation(const Projectile *projectile)
{
    if (projectile->kind == PROJECTILE_WATER) {
        return 0.45f;
    }
    return 0.34f + 0.06f * sinf((float)projectile->age * 0.35f);
}
