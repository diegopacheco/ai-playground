#include "splash.h"

#include <string.h>

void splashes_clear(SplashPool *pool)
{
    memset(pool, 0, sizeof *pool);
}

bool splashes_spawn(SplashPool *pool, Vec2 pos)
{
    for (int i = 0; i < MAX_SPLASHES; i++) {
        if (pool->items[i].timer == 0) {
            pool->items[i].pos = pos;
            pool->items[i].timer = SPLASH_DURATION;
            return true;
        }
    }
    return false;
}

void splashes_update(SplashPool *pool)
{
    for (int i = 0; i < MAX_SPLASHES; i++) {
        if (pool->items[i].timer > 0) {
            pool->items[i].timer--;
        }
    }
}

int splashes_active(const SplashPool *pool)
{
    int count = 0;
    for (int i = 0; i < MAX_SPLASHES; i++) {
        count += pool->items[i].timer > 0;
    }
    return count;
}
