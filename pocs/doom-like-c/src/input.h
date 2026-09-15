#ifndef INPUT_H
#define INPUT_H

#include <stdbool.h>

typedef struct {
    bool forward;
    bool back;
    bool strafe_left;
    bool strafe_right;
    bool turn_left;
    bool turn_right;
    bool fire;
    bool start;
} Input;

#endif
