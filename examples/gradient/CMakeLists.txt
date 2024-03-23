cmake_minimum_required(VERSION 3.14)
project(mnfs-gradient LANGUAGES CXX)

include("../../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    find_package(mnfs REQUIRED)
endif()

set(sources main.cpp)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

add_executable(mnfs-gradient)
target_sources(mnfs-gradient PRIVATE ${sources})
target_link_libraries(mnfs-gradient PRIVATE mnfs::mnfs fmt)

if(NOT is_top_level)
    win_copy_deps_to_target_dir(mnfs-gradient mnfs::mnfs)
endif()
