#ifndef VEC_H
#define VEC_H

#include <math.h>

typedef struct {
    float x;
    float y;
} Vec2;

static inline Vec2 vec2(float x, float y)
{
    Vec2 v = { x, y };
    return v;
}

static inline Vec2 vec2_add(Vec2 a, Vec2 b)
{
    return vec2(a.x + b.x, a.y + b.y);
}

static inline Vec2 vec2_sub(Vec2 a, Vec2 b)
{
    return vec2(a.x - b.x, a.y - b.y);
}

static inline Vec2 vec2_scale(Vec2 a, float s)
{
    return vec2(a.x * s, a.y * s);
}

static inline float vec2_length(Vec2 a)
{
    return sqrtf(a.x * a.x + a.y * a.y);
}

static inline float vec2_distance(Vec2 a, Vec2 b)
{
    return vec2_length(vec2_sub(a, b));
}

static inline Vec2 vec2_normalize(Vec2 a)
{
    float length = vec2_length(a);
    return length > 0.0f ? vec2_scale(a, 1.0f / length) : a;
}

static inline Vec2 vec2_from_angle(float angle)
{
    return vec2(cosf(angle), sinf(angle));
}

#endif
