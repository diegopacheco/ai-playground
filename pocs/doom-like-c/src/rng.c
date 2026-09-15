#include "rng.h"

void rng_seed(Rng *rng, uint32_t seed)
{
    rng->state = seed != 0 ? seed : 0x9E3779B9u;
}

uint32_t rng_next(Rng *rng)
{
    uint32_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

int rng_range(Rng *rng, int min, int max)
{
    if (max <= min) {
        return min;
    }
    return min + (int)(rng_next(rng) % (uint32_t)(max - min + 1));
}
