#ifndef CHECK_H
#define CHECK_H

#include <math.h>
#include <stdio.h>

extern int check_count;
extern int check_failures;

#define CHECK(condition) \
    do { \
        check_count++; \
        if (!(condition)) { \
            check_failures++; \
            fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        } \
    } while (0)

#define CHECK_NEAR(actual, expected, tolerance) CHECK(fabs((double)(actual) - (double)(expected)) <= (tolerance))

#define RUN_TEST(test) \
    do { \
        printf("  %s\n", #test); \
        test(); \
    } while (0)

void run_level_tests(void);
void run_world_tests(void);
void run_combat_tests(void);
void run_game_tests(void);
void run_render_tests(void);

#endif
