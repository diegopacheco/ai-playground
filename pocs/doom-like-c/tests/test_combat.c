#include "check.h"

#include "combat.h"
#include "level.h"

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

static Enemy goblin_at(Vec2 pos)
{
    Rng rng;
    Enemy enemy;
    rng_seed(&rng, 42);
    enemy_spawn(&enemy, pos, &rng);
    return enemy;
}

static void three_water_hits_soak_a_goblin(void)
{
    ProjectilePool pool;
    SplashPool splashes;
    Enemy enemy = goblin_at(vec2(3.5f, 1.5f));
    projectiles_clear(&pool);
    splashes_clear(&splashes);
    for (int hit = 1; hit <= ENEMY_MAX_HEALTH; hit++) {
        projectiles_spawn(&pool, PROJECTILE_WATER, enemy.pos, vec2(1.0f, 0.0f));
        int kills = combat_water_hits(&pool, &enemy, 1, &splashes);
        CHECK(kills == (hit == ENEMY_MAX_HEALTH ? 1 : 0));
        CHECK(projectiles_active(&pool, PROJECTILE_WATER) == 0);
    }
    CHECK(enemy.state == ENEMY_DEAD);
    CHECK(splashes_active(&splashes) == ENEMY_MAX_HEALTH);
}

static void soaked_goblins_do_not_absorb_more_water(void)
{
    ProjectilePool pool;
    SplashPool splashes;
    Enemy enemy = goblin_at(vec2(3.5f, 1.5f));
    enemy.state = ENEMY_DEAD;
    projectiles_clear(&pool);
    splashes_clear(&splashes);
    projectiles_spawn(&pool, PROJECTILE_WATER, enemy.pos, vec2(1.0f, 0.0f));
    CHECK(combat_water_hits(&pool, &enemy, 1, &splashes) == 0);
    CHECK(projectiles_active(&pool, PROJECTILE_WATER) == 1);
}

static void rubber_duck_hurts_the_player_once(void)
{
    ProjectilePool pool;
    Player player;
    player_spawn(&player, vec2(1.5f, 1.5f), 0.0f);
    projectiles_clear(&pool);
    projectiles_spawn(&pool, PROJECTILE_DUCK, player.pos, vec2(1.0f, 0.0f));
    CHECK(combat_duck_hits(&pool, &player) == 1);
    CHECK(player.health == PLAYER_MAX_HEALTH - DUCK_DAMAGE);
    CHECK(player.hurt_timer > 0);
    CHECK(combat_duck_hits(&pool, &player) == 0);
    CHECK(player.health == PLAYER_MAX_HEALTH - DUCK_DAMAGE);
}

static void water_splashes_on_walls_but_ducks_do_not(void)
{
    Map map = pillar_map();
    ProjectilePool pool;
    SplashPool splashes;
    projectiles_clear(&pool);
    splashes_clear(&splashes);
    projectiles_spawn(&pool, PROJECTILE_WATER, vec2(1.5f, 2.5f), vec2(1.0f, 0.0f));
    projectiles_spawn(&pool, PROJECTILE_DUCK, vec2(1.5f, 2.5f), vec2(1.0f, 0.0f));
    for (int tick = 0; tick < 40; tick++) {
        projectiles_update(&pool, &map, &splashes);
    }
    CHECK(projectiles_active(&pool, PROJECTILE_WATER) == 0);
    CHECK(projectiles_active(&pool, PROJECTILE_DUCK) == 0);
    CHECK(splashes_active(&splashes) == 1);
}

static void firing_spends_water_and_respects_cooldown(void)
{
    Player player;
    player_spawn(&player, vec2(1.5f, 1.5f), 0.0f);
    player.water = 2;
    CHECK(player_try_fire(&player));
    CHECK(player.water == 1);
    CHECK(!player_try_fire(&player));
    player.fire_cooldown = 0;
    CHECK(player_try_fire(&player));
    player.fire_cooldown = 0;
    CHECK(player.water == 0);
    CHECK(!player_try_fire(&player));
}

static int ticks_until_throw(Enemy *enemy, const Map *map, Vec2 target, int limit)
{
    Rng rng;
    rng_seed(&rng, 7);
    for (int tick = 1; tick <= limit; tick++) {
        if (enemy_update(enemy, map, target, &rng) == ENEMY_ACTION_THROW) {
            return tick;
        }
    }
    return -1;
}

static void goblin_throws_only_when_it_sees_the_player(void)
{
    Map map = pillar_map();
    Enemy hidden = goblin_at(vec2(5.5f, 2.5f));
    Enemy visible = goblin_at(vec2(5.5f, 1.5f));
    int limit = ENEMY_FIRST_THROW_MAX + ENEMY_WINDUP_TICKS + 1;
    CHECK(ticks_until_throw(&hidden, &map, vec2(1.5f, 2.5f), 300) == -1);
    CHECK(ticks_until_throw(&visible, &map, vec2(1.5f, 1.5f), limit) > 0);
}

static void water_interrupts_a_throw_windup(void)
{
    Map map = pillar_map();
    Enemy enemy = goblin_at(vec2(5.5f, 1.5f));
    Rng rng;
    rng_seed(&rng, 3);
    enemy.state = ENEMY_WINDUP;
    enemy.timer = 1;
    CHECK(!enemy_soak(&enemy, WATER_DAMAGE));
    CHECK(enemy.state == ENEMY_HURT);
    for (int tick = 0; tick < ENEMY_HURT_TICKS; tick++) {
        CHECK(enemy_update(&enemy, &map, vec2(1.5f, 1.5f), &rng) == ENEMY_ACTION_NONE);
    }
}

static void pickups_refill_up_to_the_cap_and_stay_when_full(void)
{
    Player player;
    player_spawn(&player, vec2(1.5f, 1.5f), 0.0f);
    Pickup water = { PICKUP_WATER, player.pos, true };
    Pickup health = { PICKUP_HEALTH, player.pos, true };
    player.water = PLAYER_MAX_WATER - 5;
    CHECK(pickup_try_collect(&water, &player));
    CHECK(player.water == PLAYER_MAX_WATER);
    CHECK(!water.active);
    CHECK(!pickup_try_collect(&health, &player));
    CHECK(health.active);
    player.health = 50;
    CHECK(pickup_try_collect(&health, &player));
    CHECK(player.health == 50 + PICKUP_HEALTH_AMOUNT);
}

static void pickups_out_of_reach_are_ignored(void)
{
    Player player;
    player_spawn(&player, vec2(1.5f, 1.5f), 0.0f);
    player.water = 0;
    Pickup water = { PICKUP_WATER, vec2(4.5f, 1.5f), true };
    CHECK(!pickup_try_collect(&water, &player));
    CHECK(player.water == 0);
}

void run_combat_tests(void)
{
    RUN_TEST(three_water_hits_soak_a_goblin);
    RUN_TEST(soaked_goblins_do_not_absorb_more_water);
    RUN_TEST(rubber_duck_hurts_the_player_once);
    RUN_TEST(water_splashes_on_walls_but_ducks_do_not);
    RUN_TEST(firing_spends_water_and_respects_cooldown);
    RUN_TEST(goblin_throws_only_when_it_sees_the_player);
    RUN_TEST(water_interrupts_a_throw_windup);
    RUN_TEST(pickups_refill_up_to_the_cap_and_stay_when_full);
    RUN_TEST(pickups_out_of_reach_are_ignored);
}
