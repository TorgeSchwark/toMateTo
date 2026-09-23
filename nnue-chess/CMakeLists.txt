cmake_minimum_required(VERSION 3.16)
project(nnue_chess CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# full_cycle uses std::thread to parallelize Stockfish calls across cores.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# --- AVX2 / AVX512 flags -----------------------------------------------
# AVX2 is the required baseline for the fast paths in include/nnue/simd.hpp;
# AVX512VNNI is opportunistic (only available on server-class / some newer
# consumer CPUs - notably NOT on 12th/13th/14th-gen Intel Core, which fuses
# AVX-512 off). Everything still builds and runs without either, falling
# back to the scalar reference implementation.
#
# Defaults to OFF: there is no runtime CPU-feature check anywhere in this
# codebase (simd.hpp picks its code path via compiler-defined macros at
# COMPILE time, not via CPUID at run time), so a binary built with this ON
# contains raw AVX-512 instructions unconditionally and will crash with
# "Illegal instruction" (SIGILL) the moment it runs on a CPU that doesn't
# actually have AVX-512 - which is most current consumer Intel CPUs,
# including every 12th/13th/14th-gen Core chip. Only turn this on if you
# know the *exact machine this binary will run on* supports AVX512VNNI
# (check with `grep avx512vnni /proc/cpuinfo` on Linux/WSL first).
option(NNUE_ENABLE_AVX512 "Also compile the AVX512VNNI fast path (only if your CPU actually has it - see above)" OFF)

if(MSVC)
  add_compile_options(/arch:AVX2 /O2)
  if(NNUE_ENABLE_AVX512)
    # MSVC doesn't expose a dedicated VNNI switch; /arch:AVX512 is the
    # closest fit and still requires the CPU to advertise AVX512VNNI at
    # runtime for the _mm512_dpbusd_epi32 path in simd.hpp to be reachable
    # (guarded there by the __AVX512VNNI__ macro).
    add_compile_options(/arch:AVX512)
  endif()
else()
  add_compile_options(-mavx2 -O3)
  if(NNUE_ENABLE_AVX512)
    add_compile_options(-mavx512f -mavx512bw -mavx512vnni)
  endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# --- Library (header-only, but a target makes IDE integration nicer) ---
add_library(nnue INTERFACE)
target_include_directories(nnue INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- Inference demo ------------------------------------------------------
add_executable(nnue_demo src/demo.cpp)
target_link_libraries(nnue_demo PRIVATE nnue)

# --- Training demo ---------------------------------------------------------
add_executable(nnue_train_demo training/train_demo.cpp)
target_link_libraries(nnue_train_demo PRIVATE nnue)

# --- SIMD correctness self-test (AVX2/AVX512 kernels vs. scalar reference) -
# simd_fast.cpp always gets the flags above; simd_scalar.cpp is compiled
# without them (and with NNUE_FORCE_SCALAR) regardless of the flags set
# globally, so the comparison is meaningful even when this whole project is
# built with AVX2/AVX512 enabled by default.
add_library(nnue_simd_scalar_ref OBJECT tests/simd_scalar.cpp)
target_compile_definitions(nnue_simd_scalar_ref PRIVATE NNUE_FORCE_SCALAR)
if(MSVC)
  target_compile_options(nnue_simd_scalar_ref PRIVATE /arch:SSE2)
else()
  target_compile_options(nnue_simd_scalar_ref PRIVATE -mno-avx -mno-avx2 -mno-avx512f)
endif()

add_executable(test_simd tests/test_simd.cpp tests/simd_fast.cpp $<TARGET_OBJECTS:nnue_simd_scalar_ref>)
target_link_libraries(test_simd PRIVATE nnue)

enable_testing()
add_test(NAME simd_correctness COMMAND test_simd)

# --- Full training cycle (Stockfish data generation + training) -----------
# Reuses the parent toMateTo engine's board representation and legal move
# generator (chess_board.cpp etc.) instead of reimplementing chess rules,
# and talks to Stockfish via fork+exec (see training/uci_engine.hpp), the
# same approach the parent repo's testing/stockfish_perft.cpp already uses -
# so this target only builds on POSIX and only if that engine is found one
# directory up (i.e. nnue-chess is checked out inside the toMateTo repo).
#
# Architecture is a compile-time choice (see full_cycle.cpp's header
# comment for why) - these four cache variables are it; change them with
# -DNNUE_ACC_SIZE=... on the cmake command line, or just use ../train.sh,
# which does that (and reconfigures/rebuilds) for you from a plain
# "--arch ACC,H1,H2,H3" flag.
set(NNUE_ACC_SIZE 256 CACHE STRING "Feature-transformer accumulator width (per perspective)")
set(NNUE_H1 128 CACHE STRING "Hidden layer 1 width")
set(NNUE_H2 32 CACHE STRING "Hidden layer 2 width")
set(NNUE_H3 32 CACHE STRING "Hidden layer 3 width")

get_filename_component(TOMATETO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

if(UNIX AND EXISTS "${TOMATETO_ROOT}/toMateTo_engine/move_generation/chess_board.cpp")
  add_executable(full_cycle
    training/full_cycle.cpp
    ${TOMATETO_ROOT}/toMateTo_engine/move_generation/chess_board.cpp
    ${TOMATETO_ROOT}/toMateTo_engine/move_generation/find_capture_moves.cpp
    ${TOMATETO_ROOT}/toMateTo_engine/move_generation/move_stack.cpp
    ${TOMATETO_ROOT}/toMateTo_engine/table_generation/knight_tables.cpp
    ${TOMATETO_ROOT}/toMateTo_engine/table_generation/magic_king_tables.cpp
    ${TOMATETO_ROOT}/toMateTo_engine/table_generation/magic_helper_functions.cpp
    ${TOMATETO_ROOT}/toMateTo_engine/table_generation/magic_gen.cpp
    ${TOMATETO_ROOT}/toMateTo_engine/table_generation/TT.cpp
  )
  target_include_directories(full_cycle PRIVATE ${TOMATETO_ROOT} ${TOMATETO_ROOT}/toMateTo_engine)
  target_compile_definitions(full_cycle PRIVATE
    NNUE_ACC_SIZE=${NNUE_ACC_SIZE}
    NNUE_H1=${NNUE_H1}
    NNUE_H2=${NNUE_H2}
    NNUE_H3=${NNUE_H3}
    TABLE_DIR_PATH="${TOMATETO_ROOT}/toMateTo_engine/table_generation/tables"
  )
  target_link_libraries(full_cycle PRIVATE nnue Threads::Threads)

  # --- Move-legality fuzzer: plays random games against the PARENT engine's
  # real move generator (read-only - same link approach as full_cycle above)
  # and independently re-verifies, after every move, that it didn't leave
  # the mover's own king in check - see tests/test_move_legality.cpp.
  add_executable(test_move_legality
    tests/test_move_legality.cpp
    ${TOMATETO_ROOT}/toMateTo_engine/move_generation/chess_board.cpp
    ${TOMATETO_ROOT}/toMateTo_engine/move_generation/find_capture_moves.cpp
    ${TOMATETO_ROOT}/toMateTo_engine/move_generation/move_stack.cpp
    ${TOMATETO_ROOT}/toMateTo_engine/table_generation/knight_tables.cpp
    ${TOMATETO_ROOT}/toMateTo_engine/table_generation/magic_king_tables.cpp
    ${TOMATETO_ROOT}/toMateTo_engine/table_generation/magic_helper_functions.cpp
    ${TOMATETO_ROOT}/toMateTo_engine/table_generation/magic_gen.cpp
  )
  target_include_directories(test_move_legality PRIVATE ${TOMATETO_ROOT} ${TOMATETO_ROOT}/toMateTo_engine)
  target_compile_definitions(test_move_legality PRIVATE
    TABLE_DIR_PATH="${TOMATETO_ROOT}/toMateTo_engine/table_generation/tables"
  )
  add_test(NAME move_legality COMMAND test_move_legality)
else()
  message(STATUS "full_cycle/test_move_legality targets skipped: parent toMateTo engine not found at "
                  "${TOMATETO_ROOT} (or not on a POSIX system) - it needs that repo's "
                  "move generator and Stockfish binary.")
endif()

# --- Self-play reinforcement learning (TD-bootstrapping, no Stockfish for
# training - see training/self_play.cpp's header comment) ------------------
# Unlike full_cycle, this links its OWN COPY of the move generator under
# engine/toMateTo_engine/ (see engine/README if present, or just diff
# against ../toMateTo_engine) - fully self-contained inside nnue-chess, so
# it only needs the copy to exist here, not the parent repo. Stockfish is
# still located (same "testing/stockfish_bin" search upward from cwd as
# full_cycle) but purely for the post-training validation step.
if(UNIX AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/engine/toMateTo_engine/move_generation/chess_board.cpp")
  add_executable(self_play
    training/self_play.cpp
    engine/toMateTo_engine/move_generation/chess_board.cpp
    engine/toMateTo_engine/move_generation/find_capture_moves.cpp
    engine/toMateTo_engine/move_generation/move_stack.cpp
    engine/toMateTo_engine/table_generation/knight_tables.cpp
    engine/toMateTo_engine/table_generation/magic_king_tables.cpp
    engine/toMateTo_engine/table_generation/magic_helper_functions.cpp
    engine/toMateTo_engine/table_generation/magic_gen.cpp
    engine/toMateTo_engine/table_generation/TT.cpp
  )
  target_include_directories(self_play PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/engine
                                                ${CMAKE_CURRENT_SOURCE_DIR}/engine/toMateTo_engine)
  target_compile_definitions(self_play PRIVATE
    NNUE_ACC_SIZE=${NNUE_ACC_SIZE}
    NNUE_H1=${NNUE_H1}
    NNUE_H2=${NNUE_H2}
    NNUE_H3=${NNUE_H3}
    TABLE_DIR_PATH="${CMAKE_CURRENT_SOURCE_DIR}/engine/toMateTo_engine/table_generation/tables"
  )
  target_link_libraries(self_play PRIVATE nnue)
else()
  message(STATUS "self_play target skipped: engine/toMateTo_engine/ copy not found under "
                  "${CMAKE_CURRENT_SOURCE_DIR} (or not on a POSIX system).")
endif()

# --- TestEngine: real search (TT, null-move pruning, PVS, delta-pruned
# quiescence - see engine/toMateTo_engine/toMateTo/toMateTo_nnue.{h,cpp}'s
# header comment) evaluated by an NNUE net kept incrementally updated across
# the search tree, not recomputed per node. Shares the engine/ copy with
# self_play (same reasoning: fully self-contained inside nnue-chess/,
# ../toMateTo_engine/ is never touched) plus that copy's toMateTo/ subdir
# (eval.cpp for delta-pruning's game-phase helper, toMateTo_nnue.cpp for the
# search itself).
option(NNUE_DEBUG_LEGALITY
  "Debug build: re-verify (independently, no magic bitboards) after every move in TestEngine's search that it didn't leave the mover's own king in check - see toMateTo_nnue.cpp. Off by default: it's an extra full board rescan per move, not free."
  OFF)

set(TESTENGINE_SOURCES
  engine/toMateTo_engine/move_generation/chess_board.cpp
  engine/toMateTo_engine/move_generation/find_capture_moves.cpp
  engine/toMateTo_engine/move_generation/move_stack.cpp
  engine/toMateTo_engine/table_generation/knight_tables.cpp
  engine/toMateTo_engine/table_generation/magic_king_tables.cpp
  engine/toMateTo_engine/table_generation/magic_helper_functions.cpp
  engine/toMateTo_engine/table_generation/magic_gen.cpp
  engine/toMateTo_engine/table_generation/TT.cpp
  engine/toMateTo_engine/toMateTo/eval.cpp
  engine/toMateTo_engine/toMateTo/toMateTo_nnue.cpp
)

if(UNIX AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/engine/toMateTo_engine/toMateTo/toMateTo_nnue.cpp")
  add_executable(test_nnue_accumulator tests/test_nnue_accumulator.cpp ${TESTENGINE_SOURCES})
  target_include_directories(test_nnue_accumulator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/engine
                                                            ${CMAKE_CURRENT_SOURCE_DIR}/engine/toMateTo_engine
                                                            ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_definitions(test_nnue_accumulator PRIVATE
    NNUE_ACC_SIZE=${NNUE_ACC_SIZE}
    NNUE_H1=${NNUE_H1}
    NNUE_H2=${NNUE_H2}
    NNUE_H3=${NNUE_H3}
    TABLE_DIR_PATH="${CMAKE_CURRENT_SOURCE_DIR}/engine/toMateTo_engine/table_generation/tables"
  )
  target_link_libraries(test_nnue_accumulator PRIVATE nnue)
  add_test(NAME nnue_accumulator_correctness COMMAND test_nnue_accumulator)

  # --- strength_match: plays TestEngine vs. Elo-capped Stockfish, --------
  # parallel across --threads games - see testing/strength_match.cpp.
  add_executable(strength_match testing/strength_match.cpp ${TESTENGINE_SOURCES})
  target_include_directories(strength_match PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/engine
                                                     ${CMAKE_CURRENT_SOURCE_DIR}/engine/toMateTo_engine
                                                     ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                     ${CMAKE_CURRENT_SOURCE_DIR}/training)
  target_compile_definitions(strength_match PRIVATE
    NNUE_ACC_SIZE=${NNUE_ACC_SIZE}
    NNUE_H1=${NNUE_H1}
    NNUE_H2=${NNUE_H2}
    NNUE_H3=${NNUE_H3}
    TABLE_DIR_PATH="${CMAKE_CURRENT_SOURCE_DIR}/engine/toMateTo_engine/table_generation/tables"
  )
  target_link_libraries(strength_match PRIVATE nnue Threads::Threads)
  if(NNUE_DEBUG_LEGALITY)
    target_compile_definitions(strength_match PRIVATE NNUE_DEBUG_LEGALITY)
    target_compile_definitions(test_nnue_accumulator PRIVATE NNUE_DEBUG_LEGALITY)
  endif()
else()
  message(STATUS "test_nnue_accumulator/strength_match targets skipped: engine/toMateTo_engine/toMateTo/ not found.")
endif()
