# ── Compiler ────────────────────────────────────────────────────────────────
CC      := mpicc

# ── Directories ──────────────────────────────────────────────────────────────
SRC_DIR     := src
INC_DIR     := include
BUILD_DIR   := build
LIB_DIR     := $(BUILD_DIR)/lib
OBJ_DIR     := $(BUILD_DIR)/obj
BIN_DIR     := $(BUILD_DIR)/bin

# ── Target names ─────────────────────────────────────────────────────────────
LIB_NAME    := libcollbench
BINARY      := collbench

# ── Sources ───────────────────────────────────────────────────────────────────
# Library sources (everything except main.c)
LIB_SRCS    := $(filter-out $(SRC_DIR)/main.c, $(wildcard $(SRC_DIR)/*.c))
BIN_SRCS    := $(SRC_DIR)/main.c

LIB_OBJS    := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(LIB_SRCS))
BIN_OBJS    := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(BIN_SRCS))

# ── Flags ─────────────────────────────────────────────────────────────────────
COMMON_FLAGS := -Wall -Wextra -Wpedantic -I$(INC_DIR)

DEBUG_FLAGS   := $(COMMON_FLAGS) -O0 -g3 -DDEBUG
RELEASE_FLAGS := $(COMMON_FLAGS) -O3 -march=native -DNDEBUG

DEBUG_LDFLAGS   := -lm
RELEASE_LDFLAGS := -lm

# ── Build mode (default: release) ─────────────────────────────────────────────
BUILD ?= release

ifeq ($(BUILD), debug)
    CFLAGS  := $(DEBUG_FLAGS)
    LDFLAGS := $(DEBUG_LDFLAGS)
    SUFFIX  := _debug
else ifeq ($(BUILD), release)
    CFLAGS  := $(RELEASE_FLAGS)
    LDFLAGS := $(RELEASE_LDFLAGS)
    SUFFIX  :=
else
    $(error Unknown BUILD mode: $(BUILD). Use BUILD=debug or BUILD=release)
endif

LIB_TARGET := $(LIB_DIR)/$(LIB_NAME)$(SUFFIX).a
BIN_TARGET  := $(BIN_DIR)/$(BINARY)$(SUFFIX)

# ── Phony targets ─────────────────────────────────────────────────────────────
.PHONY: all lib bin debug release clean help

all: bin

debug:
	$(MAKE) BUILD=debug bin

release:
	$(MAKE) BUILD=release bin

lib: $(LIB_TARGET)

bin: $(BIN_TARGET)

# ── Compile objects ───────────────────────────────────────────────────────────
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# ── Static library ────────────────────────────────────────────────────────────
$(LIB_TARGET): $(LIB_OBJS) | $(LIB_DIR)
	$(AR) rcs $@ $^

# ── Binary ────────────────────────────────────────────────────────────────────
$(BIN_TARGET): $(BIN_OBJS) $(LIB_TARGET) | $(BIN_DIR)
	$(CC) $(LDFLAGS) $^ -o $@

# ── Directory creation ────────────────────────────────────────────────────────
$(OBJ_DIR) $(LIB_DIR) $(BIN_DIR):
	mkdir -p $@

# ── Compile commands (for clangd / LSP) ──────────────────────────────────────
compile_commands.json: $(wildcard $(SRC_DIR)/*.c)
	$(MAKE) clean
	bear -- $(MAKE) BUILD=debug all

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD_DIR) compile_commands.json

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "Usage:"
	@echo "  make [BUILD=debug|release]   Build binary (default: release)"
	@echo "  make debug                   Shorthand for BUILD=debug"
	@echo "  make release                 Shorthand for BUILD=release"
	@echo "  make lib [BUILD=...]         Build static library only"
	@echo "  make compile_commands.json   Regenerate for clangd (requires bear)"
	@echo "  make clean                   Remove all build artifacts"
	@echo ""
	@echo "Outputs:"
	@echo "  $(LIB_DIR)/$(LIB_NAME)[_debug].a"
	@echo "  $(BIN_DIR)/$(BINARY)[_debug]"
