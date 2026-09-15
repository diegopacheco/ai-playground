#ifndef RENDER_SCREENS_H
#define RENDER_SCREENS_H

#include "framebuffer.h"
#include "game.h"

void render_logo(Framebuffer *fb, int tick);
void render_title(Framebuffer *fb, int tick);
void render_end_overlay(Framebuffer *fb, const Game *game);

#endif
