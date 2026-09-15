#ifndef SPLASH_H
#define SPLASH_H

#include <stdbool.h>

#include "vec.h"

#define MAX_SPLASHES 32
#define SPLASH_DURATION 10

typedef struct {
    Vec2 pos;
    int timer;
} Splash;

typedef struct {
    Splash items[MAX_SPLASHES];
} SplashPool;

void splashes_clear(SplashPool *pool);
bool splashes_spawn(SplashPool *pool, Vec2 pos);
void splashes_update(SplashPool *pool);
int splashes_active(const SplashPool *pool);

#endif
