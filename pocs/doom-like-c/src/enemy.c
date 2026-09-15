#include "enemy.h"

#include "collision.h"
#include "raycast.h"

void enemy_spawn(Enemy *enemy, Vec2 pos, Rng *rng)
{
    enemy->pos = pos;
    enemy->state = ENEMY_IDLE;
    enemy->health = ENEMY_MAX_HEALTH;
    enemy->timer = 0;
    enemy->throw_cooldown = rng_range(rng, ENEMY_FIRST_THROW_MIN, ENEMY_FIRST_THROW_MAX);
    enemy->anim = 0;
}

bool enemy_can_see(const Enemy *enemy, const Map *map, Vec2 target)
{
    return vec2_distance(enemy->pos, target) <= ENEMY_SIGHT_RANGE
        && raycast_line_of_sight(map, enemy->pos, target);
}

static EnemyAction finish_windup(Enemy *enemy, Rng *rng)
{
    enemy->timer--;
    if (enemy->timer > 0) {
        return ENEMY_ACTION_NONE;
    }
    enemy->state = ENEMY_CHASE;
    enemy->throw_cooldown = rng_range(rng, ENEMY_THROW_COOLDOWN_MIN, ENEMY_THROW_COOLDOWN_MAX);
    return ENEMY_ACTION_THROW;
}

static void chase(Enemy *enemy, const Map *map, Vec2 target)
{
    Vec2 to_target = vec2_sub(target, enemy->pos);
    if (vec2_length(to_target) > ENEMY_KEEP_DISTANCE) {
        Vec2 step = vec2_scale(vec2_normalize(to_target), ENEMY_SPEED);
        enemy->pos = collision_move(map, enemy->pos, step, ENEMY_RADIUS);
    }
}

EnemyAction enemy_update(Enemy *enemy, const Map *map, Vec2 target, Rng *rng)
{
    if (enemy->state == ENEMY_DEAD) {
        return ENEMY_ACTION_NONE;
    }
    enemy->anim++;
    if (enemy->throw_cooldown > 0) {
        enemy->throw_cooldown--;
    }
    if (enemy->state == ENEMY_HURT) {
        enemy->timer--;
        if (enemy->timer <= 0) {
            enemy->state = ENEMY_CHASE;
        }
        return ENEMY_ACTION_NONE;
    }
    if (enemy->state == ENEMY_WINDUP) {
        return finish_windup(enemy, rng);
    }
    if (!enemy_can_see(enemy, map, target)) {
        enemy->state = ENEMY_IDLE;
        return ENEMY_ACTION_NONE;
    }
    enemy->state = ENEMY_CHASE;
    chase(enemy, map, target);
    if (enemy->throw_cooldown == 0) {
        enemy->state = ENEMY_WINDUP;
        enemy->timer = ENEMY_WINDUP_TICKS;
    }
    return ENEMY_ACTION_NONE;
}

bool enemy_soak(Enemy *enemy, int amount)
{
    if (enemy->state == ENEMY_DEAD) {
        return false;
    }
    enemy->health -= amount;
    if (enemy->health <= 0) {
        enemy->health = 0;
        enemy->state = ENEMY_DEAD;
        return true;
    }
    enemy->state = ENEMY_HURT;
    enemy->timer = ENEMY_HURT_TICKS;
    return false;
}
