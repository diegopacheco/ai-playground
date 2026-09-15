#ifndef COMBAT_H
#define COMBAT_H

#include "enemy.h"
#include "player.h"
#include "projectile.h"
#include "splash.h"

#define WATER_DAMAGE 1
#define DUCK_DAMAGE 10
#define ENEMY_HIT_RADIUS 0.45f
#define PLAYER_HIT_RADIUS 0.4f

int combat_water_hits(ProjectilePool *pool, Enemy *enemies, int enemy_count, SplashPool *splashes);
int combat_duck_hits(ProjectilePool *pool, Player *player);

#endif
