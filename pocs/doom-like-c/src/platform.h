#ifndef PLATFORM_H
#define PLATFORM_H

#include <stdbool.h>
#include <stdint.h>

#include "input.h"

typedef struct Platform Platform;

Platform *platform_create(const char *title, int width, int height, int scale);
bool platform_poll(Platform *platform, Input *input);
void platform_present(Platform *platform, const uint32_t *argb);
uint64_t platform_ticks_ms(void);
void platform_destroy(Platform *platform);

#endif
