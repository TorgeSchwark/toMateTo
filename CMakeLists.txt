cmake_minimum_required(VERSION 3.16)
project(toMateTo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- define project folders here (only change here if folder name or location is changed!) ----
set(GUI_DIR chess_gui)
set(ENGINE_DIR toMateTo_engine)

add_executable(toMateTo
    main.cpp
    ${GUI_DIR}/gui.cpp
    ${ENGINE_DIR}/move_generation/chess_board.cpp
    ${ENGINE_DIR}/table_generation/knight_tables.cpp
    ${ENGINE_DIR}/table_generation/magic_king_tables.cpp
    ${ENGINE_DIR}/table_generation/magic_helper_functions.cpp
    ${ENGINE_DIR}/table_generation/magic_gen.cpp
    ${ENGINE_DIR}/move_generation/move_stack.cpp
)

target_include_directories(toMateTo
    PRIVATE
        .
        ${GUI_DIR}
        ${ENGINE_DIR}
)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

target_link_libraries(toMateTo
    PRIVATE
        sfml-graphics
        sfml-window
        sfml-system
)

target_compile_definitions(toMateTo PRIVATE
    TABLE_DIR_PATH="${CMAKE_SOURCE_DIR}/toMateTo_engine/table_generation/tables"
    ASSET_DIR_PATH="${CMAKE_SOURCE_DIR}/chess_gui/chess_images/"
)

