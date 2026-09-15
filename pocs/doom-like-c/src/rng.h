#ifndef RNG_H
#define RNG_H

#include <stdint.h>

typedef struct {
    uint32_t state;
} Rng;

void rng_seed(Rng *rng, uint32_t seed);
uint32_t rng_next(Rng *rng);
int rng_range(Rng *rng, int min, int max);

#endif
