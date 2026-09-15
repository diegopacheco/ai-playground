CC ?= cc
CFLAGS ?= -std=c99 -Wall -Wextra -Wpedantic -Wshadow -O2
BUILD := build
SDL_CFLAGS := $(shell sdl2-config --cflags)
SDL_LIBS := $(shell sdl2-config --libs)
HEADERS := $(wildcard src/*.h)
CORE_SRC := $(filter-out src/main.c src/platform.c,$(wildcard src/*.c))
TEST_SRC := $(wildcard tests/*.c)
SANITIZE := -g -fsanitize=address,undefined -fno-omit-frame-pointer

.PHONY: all run test shots clean

all: $(BUILD)/duckdoom

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/duckdoom: $(CORE_SRC) src/main.c src/platform.c $(HEADERS) | $(BUILD)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -Isrc $(CORE_SRC) src/main.c src/platform.c -o $@ $(SDL_LIBS) -lm

$(BUILD)/tests: $(CORE_SRC) $(TEST_SRC) $(HEADERS) tests/check.h | $(BUILD)
	$(CC) $(CFLAGS) $(SANITIZE) -Isrc -Itests $(CORE_SRC) $(TEST_SRC) -o $@ -lm

$(BUILD)/shots: $(CORE_SRC) tools/shots.c $(HEADERS) | $(BUILD)
	$(CC) $(CFLAGS) -Isrc $(CORE_SRC) tools/shots.c -o $@ -lm

run: $(BUILD)/duckdoom
	./$(BUILD)/duckdoom

test: $(BUILD)/tests
	./$(BUILD)/tests

shots: $(BUILD)/shots
	mkdir -p $(BUILD)/shots-out printscreens
	./$(BUILD)/shots $(BUILD)/shots-out
	for f in $(BUILD)/shots-out/*.bmp; do sips -s format png "$$f" --out "printscreens/$$(basename "$$f" .bmp).png" >/dev/null; done

clean:
	rm -rf $(BUILD)
