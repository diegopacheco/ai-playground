#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "combat.h"
#include "config.h"
#include "framebuffer.h"
#include "game.h"
#include "palette.h"
#include "render.h"
#include "render_screens.h"
#include "texture.h"

#define SHOT_SCALE 3
#define SHOT_SEED 20260915u
#define SHOT_TICK_LIMIT 6000

static Game game;
static Framebuffer framebuffer;
static const char *output_dir;

static void put_u32(uint8_t *out, uint32_t value)
{
    out[0] = (uint8_t)(value & 0xFF);
    out[1] = (uint8_t)((value >> 8) & 0xFF);
    out[2] = (uint8_t)((value >> 16) & 0xFF);
    out[3] = (uint8_t)((value >> 24) & 0xFF);
}

static bool write_bmp(const char *path, const Framebuffer *fb)
{
    int width = SCREEN_W * SHOT_SCALE;
    int height = SCREEN_H * SHOT_SCALE;
    int row_size = (width * 3 + 3) & ~3;
    uint32_t data_size = (uint32_t)(row_size * height);
    uint8_t header[54] = { 'B', 'M' };
    put_u32(header + 2, 54 + data_size);
    put_u32(header + 10, 54);
    put_u32(header + 14, 40);
    put_u32(header + 18, (uint32_t)width);
    put_u32(header + 22, (uint32_t)height);
    header[26] = 1;
    header[28] = 24;
    put_u32(header + 34, data_size);
    FILE *file = fopen(path, "wb");
    uint8_t *row = calloc((size_t)row_size, 1);
    if (file == NULL || row == NULL) {
        free(row);
        if (file != NULL) {
            fclose(file);
        }
        return false;
    }
    bool ok = fwrite(header, sizeof header, 1, file) == 1;
    for (int y = height - 1; y >= 0 && ok; y--) {
        for (int x = 0; x < width; x++) {
            uint32_t argb = palette_argb(fb_get(fb, x / SHOT_SCALE, y / SHOT_SCALE));
            row[x * 3] = (uint8_t)(argb & 0xFF);
            row[x * 3 + 1] = (uint8_t)((argb >> 8) & 0xFF);
            row[x * 3 + 2] = (uint8_t)((argb >> 16) & 0xFF);
        }
        ok = fwrite(row, (size_t)row_size, 1, file) == 1;
    }
    free(row);
    return fclose(file) == 0 && ok;
}

static bool save(const char *name)
{
    char path[512];
    snprintf(path, sizeof path, "%s/%s.bmp", output_dir, name);
    if (!write_bmp(path, &framebuffer)) {
        fprintf(stderr, "could not write %s\n", path);
        return false;
    }
    printf("wrote %s\n", path);
    return true;
}

static void start_at(Vec2 pos, float angle)
{
    game_init(&game, SHOT_SEED);
    game_start(&game);
    game.player.pos = pos;
    game.player.angle = angle;
}

static bool duck_in_front(void)
{
    Vec2 facing = player_direction(&game.player);
    for (int i = 0; i < MAX_PROJECTILES; i++) {
        const Projectile *projectile = &game.projectiles.items[i];
        if (!projectile->active || projectile->kind != PROJECTILE_DUCK) {
            continue;
        }
        Vec2 offset = vec2_sub(projectile->pos, game.player.pos);
        float distance = vec2_length(offset);
        if (distance > 1.2f && distance < 3.5f && (offset.x * facing.x + offset.y * facing.y) / distance > 0.93f) {
            return true;
        }
    }
    return false;
}

static bool battle_frame_ready(void)
{
    return game.kills == 0 && duck_in_front()
        && projectiles_active(&game.projectiles, PROJECTILE_WATER) >= 2;
}

static bool hurt_frame_ready(void)
{
    return game.player.hurt_timer == PLAYER_HURT_TICKS && game.player.health > DUCK_DAMAGE;
}

static bool run_until(bool (*ready)(void), const Input *input)
{
    for (int tick = 0; tick < SHOT_TICK_LIMIT && game.mode == GAME_PLAYING; tick++) {
        game_update(&game, input);
        if (ready()) {
            return true;
        }
    }
    return false;
}

static bool shot_logo(void)
{
    render_logo(&framebuffer, 6);
    return save("logo");
}

static bool shot_title(void)
{
    game_init(&game, SHOT_SEED);
    render_title(&framebuffer, 4);
    return save("title");
}

static bool shot_explore(void)
{
    Input idle = { 0 };
    start_at(vec2(1.5f, 15.5f), 0.12f);
    game_update(&game, &idle);
    render_frame(&framebuffer, &game);
    return save("explore");
}

static bool shot_battle(void)
{
    Input fire = { 0 };
    fire.fire = true;
    start_at(vec2(10.5f, 10.5f), 0.39f);
    if (!run_until(battle_frame_ready, &fire)) {
        fprintf(stderr, "battle frame never happened\n");
        return false;
    }
    render_frame(&framebuffer, &game);
    return save("battle");
}

static bool shot_duck_hit(void)
{
    Input idle = { 0 };
    start_at(vec2(9.5f, 12.5f), 0.0f);
    if (!run_until(hurt_frame_ready, &idle)) {
        fprintf(stderr, "duck hit frame never happened\n");
        return false;
    }
    render_frame(&framebuffer, &game);
    return save("duck-hit");
}

static bool game_over_ready(void)
{
    return false;
}

static bool shot_game_over(void)
{
    Input idle = { 0 };
    start_at(vec2(9.5f, 12.5f), 0.0f);
    run_until(game_over_ready, &idle);
    if (game.mode != GAME_OVER) {
        fprintf(stderr, "game over never happened\n");
        return false;
    }
    render_frame(&framebuffer, &game);
    return save("game-over");
}

static bool shot_victory(void)
{
    Input idle = { 0 };
    start_at(vec2(9.5f, 12.5f), 0.0f);
    for (int i = 0; i < game.enemy_count; i++) {
        while (!enemy_soak(&game.enemies[i], 1)) {
        }
        game.kills++;
    }
    game_update(&game, &idle);
    if (game.mode != GAME_WON) {
        fprintf(stderr, "victory never happened\n");
        return false;
    }
    render_frame(&framebuffer, &game);
    return save("victory");
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <output-dir>\n", argv[0]);
        return 1;
    }
    output_dir = argv[1];
    palette_init();
    textures_init();
    bool ok = shot_logo();
    ok = shot_title() && ok;
    ok = shot_explore() && ok;
    ok = shot_battle() && ok;
    ok = shot_duck_hit() && ok;
    ok = shot_game_over() && ok;
    ok = shot_victory() && ok;
    return ok ? 0 : 1;
}
