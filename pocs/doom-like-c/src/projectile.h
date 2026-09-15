#ifndef PROJECTILE_H
#define PROJECTILE_H

#include <stdbool.h>

#include "map.h"
#include "splash.h"
#include "vec.h"

#define MAX_PROJECTILES 96
#define WATER_SPEED 0.32f
#define DUCK_SPEED 0.15f
#define PROJECTILE_LIFETIME 140

typedef enum {
    PROJECTILE_WATER,
    PROJECTILE_DUCK
} ProjectileKind;

typedef struct {
    ProjectileKind kind;
    Vec2 pos;
    Vec2 velocity;
    bool active;
    int age;
} Projectile;

typedef struct {
    Projectile items[MAX_PROJECTILES];
} ProjectilePool;

void projectiles_clear(ProjectilePool *pool);
bool projectiles_spawn(ProjectilePool *pool, ProjectileKind kind, Vec2 pos, Vec2 direction);
void projectiles_update(ProjectilePool *pool, const Map *map, SplashPool *splashes);
int projectiles_active(const ProjectilePool *pool, ProjectileKind kind);
float projectile_elevation(const Projectile *projectile);

#endif
