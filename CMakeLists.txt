cmake_minimum_required(VERSION 3.10)

project(multi-tool)

find_package(Boost REQUIRED COMPONENTS program_options)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-Wall -Wextra -Werror -Wno-unused-parameter -Wno-ignored-qualifiers -Wno-sign-compare -Wno-error=deprecated-declarations)
set(CMAKE_CXX_FLAGS_DEBUG "-g -Og")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

execute_process(COMMAND bash "-c" "popc --version | cut -d ' ' -f3 | head -1" OUTPUT_VARIABLE POPLAR_VERSION)
string(REPLACE "." ";" VERSION_LIST ${POPLAR_VERSION})
list(GET VERSION_LIST 0 POPLAR_VERSION_MAJOR)
list(GET VERSION_LIST 1 POPLAR_VERSION_MINOR)
list(GET VERSION_LIST 2 POPLAR_VERSION_PATCH)
message(STATUS "Detected Poplar version ${POPLAR_VERSION_MAJOR}.${POPLAR_VERSION_MINOR}.${POPLAR_VERSION_PATCH}")

# Poplar 2.x and older had a different gcl library name:
set(GCL_LIB_NAME "gcl")
if(${POPLAR_VERSION_MAJOR} LESS 3)
  message(WARNING "Poplar version older than 3 detected. Consider updating.")
  set(GCL_LIB_NAME "gcl_ct")
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/glm/.git")
  message(WARNING "The GLM submodule has not been initialised.")
endif()

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
)

# popc won't accept relative path from within Poplar so we define this absolute
# path to give us a prefix to start from:
add_compile_definitions(DEFAULT_CODELET_DIR="${PROJECT_SOURCE_DIR}/src/codelets")

# Make a single library containing all tools:
file(GLOB TOOL_HEADERS ${PROJECT_SOURCE_DIR}/src/tools/*.hpp)
file(GLOB TOOL_SRC ${PROJECT_SOURCE_DIR}/src/tools/*.cpp)
add_library(tools SHARED ${TOOL_HEADERS} ${TOOL_SRC})
target_link_libraries(tools -lpoplar -lpopops -l${GCL_LIB_NAME})

# Auto detect the available tools at configuration time:
set(GENERATED_TOOLS_FILE ${PROJECT_SOURCE_DIR}/include/cmake_discovered_tools.hpp)
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

# Generate a bash completion script: if you source this script
# bash will auto complete names of discovered tools:
set(COMPLETION_FILE ${PROJECT_SOURCE_DIR}/completion.sh)
file(WRITE ${COMPLETION_FILE} "\#/usr/bin/env bash\n\# This file is auto generated during CMake configuration.\n")
file(APPEND ${COMPLETION_FILE} "_poplar_multi_tool_completions()\n{\n  if \[ \"\${\#COMP_WORDS\[\@\]}\" != \"2\" \]; then\n    return\n  fi\n  TOOLS=\"")
foreach(T ${TOOL_NAMES})
    file(APPEND ${COMPLETION_FILE} "${T} ")
endforeach()
file(APPEND ${COMPLETION_FILE} "\"\n  COMPREPLY=($(compgen -W \"\$\{TOOLS\}\" \"\$\{COMP_WORDS\[1\]\}\"))\n}\n")
file(APPEND ${COMPLETION_FILE} "\ncomplete -o default -F _poplar_multi_tool_completions multi-tool\n")

# Manually add libraries with specific functionality (potentially used across mutliple tools):
add_subdirectory(${PROJECT_SOURCE_DIR}/src/fft)

set(PROG_SRC
    ${PROJECT_SOURCE_DIR}/src/multi_tool.cpp
)
add_executable(multi-tool ${PROG_SRC})
target_link_libraries(multi-tool Boost::program_options fft tools -lpopnn -lpoplin -lpopops -lpoputil -lpthread -lpvti)
set_target_properties(multi-tool PROPERTIES BUILD_RPATH_USE_ORIGIN TRUE)
