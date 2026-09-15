#include "platform.h"

#include <stdio.h>
#include <stdlib.h>

#include "SDL.h"

struct Platform {
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Texture *texture;
    int width;
};

Platform *platform_create(const char *title, int width, int height, int scale)
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return NULL;
    }
    Platform *platform = calloc(1, sizeof *platform);
    if (platform == NULL) {
        SDL_Quit();
        return NULL;
    }
    platform->width = width;
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0");
    platform->window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        width * scale, height * scale, SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    if (platform->window != NULL) {
        platform->renderer = SDL_CreateRenderer(platform->window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    }
    if (platform->renderer != NULL) {
        SDL_RenderSetLogicalSize(platform->renderer, width, height);
        platform->texture = SDL_CreateTexture(platform->renderer, SDL_PIXELFORMAT_ARGB8888,
            SDL_TEXTUREACCESS_STREAMING, width, height);
    }
    if (platform->texture == NULL) {
        fprintf(stderr, "SDL setup failed: %s\n", SDL_GetError());
        platform_destroy(platform);
        return NULL;
    }
    return platform;
}

static void read_keyboard(Input *input)
{
    const Uint8 *keys = SDL_GetKeyboardState(NULL);
    input->forward = keys[SDL_SCANCODE_W] || keys[SDL_SCANCODE_UP];
    input->back = keys[SDL_SCANCODE_S] || keys[SDL_SCANCODE_DOWN];
    input->strafe_left = keys[SDL_SCANCODE_A];
    input->strafe_right = keys[SDL_SCANCODE_D];
    input->turn_left = keys[SDL_SCANCODE_LEFT] || keys[SDL_SCANCODE_Q];
    input->turn_right = keys[SDL_SCANCODE_RIGHT] || keys[SDL_SCANCODE_E];
    input->fire = keys[SDL_SCANCODE_SPACE] || keys[SDL_SCANCODE_LCTRL];
}

bool platform_poll(Platform *platform, Input *input)
{
    (void)platform;
    bool running = true;
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            running = false;
        }
        if (event.type == SDL_KEYDOWN && !event.key.repeat) {
            SDL_Keycode key = event.key.keysym.sym;
            if (key == SDLK_ESCAPE) {
                running = false;
            }
            if (key == SDLK_RETURN || key == SDLK_KP_ENTER) {
                input->start = true;
            }
        }
    }
    read_keyboard(input);
    return running;
}

void platform_present(Platform *platform, const uint32_t *argb)
{
    SDL_UpdateTexture(platform->texture, NULL, argb, platform->width * 4);
    SDL_RenderClear(platform->renderer);
    SDL_RenderCopy(platform->renderer, platform->texture, NULL, NULL);
    SDL_RenderPresent(platform->renderer);
}

uint64_t platform_ticks_ms(void)
{
    return SDL_GetTicks64();
}

void platform_destroy(Platform *platform)
{
    if (platform->texture != NULL) {
        SDL_DestroyTexture(platform->texture);
    }
    if (platform->renderer != NULL) {
        SDL_DestroyRenderer(platform->renderer);
    }
    if (platform->window != NULL) {
        SDL_DestroyWindow(platform->window);
    }
    free(platform);
    SDL_Quit();
}
