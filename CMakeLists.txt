cmake_minimum_required(VERSION 3.10)

project(multi-tool)

find_package(Boost REQUIRED COMPONENTS program_options)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-Wall -Wextra -Werror -Wno-unused-parameter -Wno-ignored-qualifiers -Wno-sign-compare)
set(CMAKE_CXX_FLAGS_DEBUG "-g -Og")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(${Boost_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

# Make a single library containing all tools:
file(GLOB TOOL_HEADERS ${PROJECT_SOURCE_DIR}/src/tools/*.hpp)
file(GLOB TOOL_SRC ${PROJECT_SOURCE_DIR}/src/tools/*.cpp)
add_library(tools SHARED ${TOOL_HEADERS} ${TOOL_SRC})
target_link_libraries(tools -lpoplar -lpopops -lgcl_ct -lpopart)

# Auto detect the available tools at configuration time:
set(GENERATED_TOOLS_FILE ${CMAKE_SOURCE_DIR}/include/cmake_discovered_tools.hpp)
file(WRITE ${GENERATED_TOOLS_FILE} "// This file is auto generated.\n#pragma once\n\n")
file(APPEND ${GENERATED_TOOLS_FILE} "#include \<tool_registry.hpp\>\n\n")
foreach(FILE_PATH ${TOOL_HEADERS})
    get_filename_component(F ${FILE_PATH} NAME)
    file(APPEND ${GENERATED_TOOLS_FILE} "#include \<tools/${F}\>\n")
endforeach()
file(APPEND ${GENERATED_TOOLS_FILE} "\nToolFactoryRegistry globalTools = {\n")
foreach(FILE_PATH ${TOOL_HEADERS})
    get_filename_component(F ${FILE_PATH} NAME_WE)
    list(APPEND TOOL_NAMES ${F})
    file(APPEND ${GENERATED_TOOLS_FILE} "  REGISTER_TOOL(${F}),\n")
endforeach()
file(APPEND ${GENERATED_TOOLS_FILE} "};\n\n")
message(STATUS "Discovered the following tools: ${TOOL_NAMES}")

# Manually add libraries with specific functionality (potentially
# used across mutliple tools).
file(GLOB FFT_HEADERS ${PROJECT_SOURCE_DIR}/src/fft/*.hpp)
file(GLOB FFT_SRC ${PROJECT_SOURCE_DIR}/src/fft/*.cpp)
add_library(fft ${FFT_HEADERS} ${FFT_SRC})

set(PROG_SRC
    ${CMAKE_SOURCE_DIR}/src/multi_tool.cpp
)
add_executable(multi-tool ${PROG_SRC})
target_link_libraries(multi-tool Boost::program_options fft tools -lpoplin -lpopops -lpoputil -lpthread -lpvti)
