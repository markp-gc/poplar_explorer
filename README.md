# Poplar Exporer Tool

The purpose of this tool is to enable rapid prototyping of Poplar/Poplibs programs by providing
a framework which takes care of all boiler plate setup code and provides access to multiple
tools with a unified command line interface and features (e.g. graph executable save/load, compilation
progress and consistent logging). Compatible tools are auto discovered at build configuration time
to make adding new programs as easy as possible.

## Configuration and Building

1) Prepare the environment.

Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system.

2) Install apt dependencies, configure, and build.

```
sudo apt install cmake ninja-build libspdlog-dev
mkdir build
cd build
cmake ../ -G Ninja
ninja -j10
```

You can then run `./multi-tool` to list the available tools e.g.:

```
Usage: ./multi-tool tool-name [--help]

Please choose a tool to run from the following:
    EmptyTool GroupedMatmulBenchmark MatmulBenchmark RemoteBufferBenchmark 
```

To get help for a specific tool add the tool name first on the command line followed by `--help`. E.g.:
```
./multi-tool MatmulBenchmark --help
```

## Adding a new tool

A new tool is created by defining a C++ class that inherits from the abstract base classes `ipu_utils::BuilderInterface` and `ToolInterface`.

### Implement the Required Interfaces

`BuilderInterface` gives an interface for
construction and execution of Poplar graphs, and `ToolInterface` ensures a consistent command
line interface and enables autoamtic tool discovery. You can use `src/tools/EmptyTools` as a
template to make a new tool: it contains all the methods you need to override and a brief
description of what each method should do. More detailed descriptions of the interfaces can
be found in `src/ipu_utils.hpp` (where `BuilderInterface` is defined).

Note: `BuilderInterface` can stand alone and be used independently of the `ToolInterface` but
the converse is not true (`ToolInterface` expects the tool class to also inherit from
`BuilderInterface`).

### Auto discovery.

Tool classes that are declared in C++ header files with `.hpp` extension in the `src\tools` directory
will be auto discovered at configuration time. The name of the tool is inferred form the header file
so the object name must match. E.g. a new tool `class MyTool ...` must be declared in a header file
`src\tools\MyTool.hpp`. (Note: You need to manually re-run CMake to detect a newly added tool).
You can check that your tool was detected correctly during configuration by checking it appears in
the generated file: `cmake_discovered_tools.hpp`. The tool will then be auto discovered and listed as
an available tool when you run `./multi-tool` with no arguments.
