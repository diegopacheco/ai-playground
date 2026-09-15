#ifndef RENDER_HUD_H
#define RENDER_HUD_H

#include "framebuffer.h"
#include "game.h"

void render_weapon(Framebuffer *fb, const Game *game);
void render_hud(Framebuffer *fb, const Game *game);

#endif
