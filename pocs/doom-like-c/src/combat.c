#include "combat.h"

#include <stddef.h>

static Enemy *enemy_hit_by(const Projectile *projectile, Enemy *enemies, int enemy_count)
{
    for (int i = 0; i < enemy_count; i++) {
        if (enemies[i].state != ENEMY_DEAD && vec2_distance(projectile->pos, enemies[i].pos) < ENEMY_HIT_RADIUS) {
            return &enemies[i];
        }
    }
    return NULL;
}

int combat_water_hits(ProjectilePool *pool, Enemy *enemies, int enemy_count, SplashPool *splashes)
{
    int kills = 0;
    for (int i = 0; i < MAX_PROJECTILES; i++) {
        Projectile *projectile = &pool->items[i];
        if (!projectile->active || projectile->kind != PROJECTILE_WATER) {
            continue;
        }
        Enemy *enemy = enemy_hit_by(projectile, enemies, enemy_count);
        if (enemy == NULL) {
            continue;
        }
        projectile->active = false;
        splashes_spawn(splashes, projectile->pos);
        kills += enemy_soak(enemy, WATER_DAMAGE);
    }
    return kills;
}

int combat_duck_hits(ProjectilePool *pool, Player *player)
{
    int hits = 0;
    for (int i = 0; i < MAX_PROJECTILES; i++) {
        Projectile *projectile = &pool->items[i];
        if (!projectile->active || projectile->kind != PROJECTILE_DUCK) {
            continue;
        }
        if (vec2_distance(projectile->pos, player->pos) < PLAYER_HIT_RADIUS) {
            projectile->active = false;
            player_hurt(player, DUCK_DAMAGE);
            hits++;
        }
    }
    return hits;
}
