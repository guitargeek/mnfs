cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 20)
project(mnfs
    VERSION 1.0.0
    DESCRIPTION "Template for C++ library built with CMake"
    LANGUAGES CXX)

#----------------------------------------------------------------------------------------------------------------------
# general settings and options
#----------------------------------------------------------------------------------------------------------------------

include(cmake/utils.cmake)
include(GNUInstallDirs)

string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

# MYLIB_SHARED_LIBS option (undefined by default) can be used to force shared/static build
option(MYLIB_BUILD_TESTS "Build mnfs tests" OFF)
option(MYLIB_BUILD_EXAMPLES "Build mnfs examples" OFF)
option(MYLIB_BUILD_DOCS "Build mnfs documentation" OFF)
option(MYLIB_INSTALL "Generate target for installing mnfs" ${is_top_level})
set_if_undefined(MYLIB_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/mnfs" CACHE STRING
    "Install path for mnfs package-related CMake files")

if(DEFINED MYLIB_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ${MYLIB_SHARED_LIBS})
endif()

if(NOT DEFINED CMAKE_BUILD_TYPE AND NOT DEFINED CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set_if_undefined(CMAKE_CXX_VISIBILITY_PRESET hidden)
set_if_undefined(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

add_library(mnfs) # initialized below
add_library(mnfs::mnfs ALIAS mnfs)

#----------------------------------------------------------------------------------------------------------------------
# mnfs dependencies
#----------------------------------------------------------------------------------------------------------------------

# Search for your dependencies here

#----------------------------------------------------------------------------------------------------------------------
# mnfs sources
#----------------------------------------------------------------------------------------------------------------------

include(GenerateExportHeader)
set(export_file_name "export_shared.h")

if(NOT BUILD_SHARED_LIBS)
    set(export_file_name "export_static.h")
endif()

generate_export_header(mnfs EXPORT_FILE_NAME include/mnfs/${export_file_name})

set(sources
    include/mnfs/derivative.h
    include/mnfs/export.h
    src/derivative.cpp)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

#----------------------------------------------------------------------------------------------------------------------
# mnfs target
#----------------------------------------------------------------------------------------------------------------------

include(CMakePackageConfigHelpers)

target_sources(mnfs PRIVATE ${sources})
target_compile_definitions(mnfs PUBLIC "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:MYLIB_STATIC_DEFINE>")

target_include_directories(mnfs
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>")

set_target_properties(mnfs PROPERTIES
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION})

if(MYLIB_INSTALL AND NOT CMAKE_SKIP_INSTALL_RULES)
    configure_package_config_file(cmake/mnfs-config.cmake.in mnfs-config.cmake
        INSTALL_DESTINATION "${MYLIB_INSTALL_CMAKEDIR}")

    write_basic_package_version_file(mnfs-config-version.cmake
        COMPATIBILITY SameMajorVersion)

    install(TARGETS mnfs EXPORT mnfs_export
        RUNTIME COMPONENT mnfs
        LIBRARY COMPONENT mnfs NAMELINK_COMPONENT mnfs-dev
        ARCHIVE COMPONENT mnfs-dev
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    install(DIRECTORY include/
        TYPE INCLUDE
        COMPONENT mnfs-dev)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/mnfs/${export_file_name}"
        COMPONENT mnfs-dev
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mnfs")

    set(targets_file "mnfs-shared-targets.cmake")

    if(NOT BUILD_SHARED_LIBS)
        set(targets_file "mnfs-static-targets.cmake")
    endif()

    install(EXPORT mnfs_export
        COMPONENT mnfs-dev
        FILE "${targets_file}"
        DESTINATION "${MYLIB_INSTALL_CMAKEDIR}"
        NAMESPACE mnfs::)

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/mnfs-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/mnfs-config-version.cmake"
        COMPONENT mnfs-dev
        DESTINATION "${MYLIB_INSTALL_CMAKEDIR}")

    if(MSVC)
        set(pdb_file "")
        set(pdb_file_destination "")

        if(BUILD_SHARED_LIBS)
            set(pdb_file "$<TARGET_PDB_FILE:mnfs>")
            set(pdb_file_destination "${CMAKE_INSTALL_BINDIR}")
        else()
            # TARGET_PDB_FILE does not work for pdb file generated for static library build, determining it manually
            set(pdb_file "$<TARGET_FILE_DIR:mnfs>/$<TARGET_FILE_PREFIX:mnfs>$<TARGET_FILE_BASE_NAME:mnfs>.pdb")
            set(pdb_file_destination "${CMAKE_INSTALL_LIBDIR}")
        endif()

        install(FILES "${pdb_file}"
            COMPONENT mnfs-dev
            CONFIGURATIONS Debug RelWithDebInfo
            DESTINATION "${pdb_file_destination}"
            OPTIONAL)
    endif()
endif()

#----------------------------------------------------------------------------------------------------------------------
# other targets
#----------------------------------------------------------------------------------------------------------------------

if(MYLIB_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(MYLIB_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(MYLIB_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    doxygen_add_docs(docs include)
endif()
