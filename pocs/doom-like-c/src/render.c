#include "render.h"

#include "render_hud.h"
#include "render_screens.h"
#include "render_world.h"

void render_frame(Framebuffer *fb, const Game *game)
{
    if (game->mode == GAME_TITLE) {
        render_title(fb, game->tick);
        return;
    }
    render_world(fb, game);
    if (game->mode == GAME_PLAYING) {
        render_weapon(fb, game);
    }
    render_hud(fb, game);
    if (game->mode != GAME_PLAYING) {
        render_end_overlay(fb, game);
    }
}
