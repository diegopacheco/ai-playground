#include "check.h"

#include "palette.h"
#include "texture.h"

int check_count = 0;
int check_failures = 0;

int main(void)
{
    palette_init();
    textures_init();
    printf("level\n");
    run_level_tests();
    printf("world\n");
    run_world_tests();
    printf("combat\n");
    run_combat_tests();
    printf("game\n");
    run_game_tests();
    printf("render\n");
    run_render_tests();
    printf("%d checks, %d failures\n", check_count, check_failures);
    return check_failures == 0 ? 0 : 1;
}
