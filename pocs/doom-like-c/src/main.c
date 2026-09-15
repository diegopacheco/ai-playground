#include <stdio.h>
#include <time.h>

#include "config.h"
#include "framebuffer.h"
#include "game.h"
#include "palette.h"
#include "platform.h"
#include "render.h"
#include "texture.h"

#define MAX_FRAME_LAG_MS 250

static Game game;
static Framebuffer framebuffer;
static uint32_t argb[SCREEN_W * SCREEN_H];

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    palette_init();
    textures_init();
    if (!game_init(&game, (uint32_t)time(NULL))) {
        fprintf(stderr, "invalid level layout\n");
        return 1;
    }
    Platform *platform = platform_create(GAME_NAME, SCREEN_W, SCREEN_H, WINDOW_SCALE);
    if (platform == NULL) {
        return 1;
    }
    Input input = { 0 };
    const uint64_t tick_ms = 1000 / TICK_RATE;
    uint64_t previous = platform_ticks_ms();
    uint64_t lag = 0;
    while (platform_poll(platform, &input)) {
        uint64_t now = platform_ticks_ms();
        lag += now - previous;
        previous = now;
        if (lag > MAX_FRAME_LAG_MS) {
            lag = MAX_FRAME_LAG_MS;
        }
        while (lag >= tick_ms) {
            game_update(&game, &input);
            input.start = false;
            lag -= tick_ms;
        }
        render_frame(&framebuffer, &game);
        fb_to_argb(&framebuffer, argb);
        platform_present(platform, argb);
    }
    platform_destroy(platform);
    return 0;
}
