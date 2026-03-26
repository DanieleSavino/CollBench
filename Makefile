# ── Compiler ────────────────────────────────────────────────────────────────
CC      := mpicc

# ── Directories ──────────────────────────────────────────────────────────────
SRC_DIR   := src
INC_DIR   := include
BUILD_DIR := build
LIB_DIR   := $(BUILD_DIR)/lib
OBJ_DIR   := $(BUILD_DIR)/obj

# ── Target names ─────────────────────────────────────────────────────────────
LIB_NAME := libcollbench

# ── Sources ───────────────────────────────────────────────────────────────────
LIB_SRCS := $(wildcard $(SRC_DIR)/*.c)
LIB_OBJS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(LIB_SRCS))

# ── Flags ─────────────────────────────────────────────────────────────────────
COMMON_FLAGS := -Wall -Wextra -Wpedantic -I$(INC_DIR)

DEBUG_FLAGS   := $(COMMON_FLAGS) -O0 -g3 -DDEBUG
RELEASE_FLAGS := $(COMMON_FLAGS) -O3 -march=native -DNDEBUG

# ── Build mode (default: release) ─────────────────────────────────────────────
BUILD ?= release

ifeq ($(BUILD), debug)
    CFLAGS := $(DEBUG_FLAGS)
    SUFFIX := _debug
else ifeq ($(BUILD), release)
    CFLAGS := $(RELEASE_FLAGS)
    SUFFIX :=
else
    $(error Unknown BUILD mode: $(BUILD). Use BUILD=debug or BUILD=release)
endif

LIB_TARGET := $(LIB_DIR)/$(LIB_NAME)$(SUFFIX).a

# ── Phony targets ─────────────────────────────────────────────────────────────
.PHONY: all debug release clean help

all: $(LIB_TARGET)

debug:
	$(MAKE) BUILD=debug

release:
	$(MAKE) BUILD=release

# ── Compile objects ───────────────────────────────────────────────────────────
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# ── Static library ────────────────────────────────────────────────────────────
$(LIB_TARGET): $(LIB_OBJS) | $(LIB_DIR)
	$(AR) rcs $@ $^

# ── Directory creation ────────────────────────────────────────────────────────
$(OBJ_DIR) $(LIB_DIR):
	mkdir -p $@

# ── Compile commands (for clangd / LSP) ──────────────────────────────────────
compile_commands.json: $(wildcard $(SRC_DIR)/*.c)
	$(MAKE) clean
	bear -- $(MAKE) BUILD=debug

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD_DIR) compile_commands.json

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "Usage:"
	@echo "  make [BUILD=debug|release]   Build library (default: release)"
	@echo "  make debug                   Shorthand for BUILD=debug"
	@echo "  make release                 Shorthand for BUILD=release"
	@echo "  make compile_commands.json   Regenerate for clangd (requires bear)"
	@echo "  make clean                   Remove all build artifacts"
	@echo ""
	@echo "Outputs:"
	@echo "  $(LIB_DIR)/$(LIB_NAME)[_debug].a"
