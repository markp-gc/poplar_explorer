// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <cstdlib>
#include <vector>
#include <limits>
#include <chrono>
#include <memory>
#include <sstream>

#include <boost/program_options.hpp>
#include "io_utils.hpp"
#include "tool_registry.hpp"

#include "../include/cmake_discovered_tools.hpp"

/// Parse the tool name and return the tool name and a
/// factory function that will create the tool specified
/// in the command line.
std::pair<std::string, ToolFactoryFunction> parseToolName(int argc, char** argv) {
  namespace po = boost::program_options;

  // We only want to get the tool name here:
  po::options_description desc("Tool Selection Options");
  desc.add_options()
  ("list-tools", po::value<std::string>(),
   "Print a list of available tools and exit."
  )
  ("tool-name", po::value<std::string>(),
   "Choose the tool to be executed."
  )
  ("misc-positional", po::value<std::vector<std::string>>(),
   "Not a real option: mops up excess positional args."
  )
  ;

  // Allow arbitrary number of positional arguments otherwise
  // command line must use = to set all other arguments:
  po::positional_options_description p;
  p.add("tool-name", 1);
  p.add("misc-positional", -1);

  boost::program_options::variables_map args;
  auto parsed = po::command_line_parser(argc, argv).options(desc).positional(p).allow_unregistered().run();
  po::store(parsed, args);

  if (args.count("tool-name") == 0) {
    std::cout << "Usage: " << argv[0] << " tool-name [--help]\n\n";
    std::cerr << "Please choose a tool to run from the following:\n"
              << enumerateToolNames(globalTools) << "\n\n";
    throw std::runtime_error("No tool specified.");
  }

  auto toolName = args.at("tool-name").as<std::string>();

  if (globalTools.count(toolName) == 0) {
    std::cerr << "Unrecognised tool: '" << toolName << "'\n\n";
    std::cerr << "Please choose a tool to run from the following:\n"
              << enumerateToolNames(globalTools) << "\n\n";
    throw std::runtime_error("Unrecognised tool name.");
  }

  return std::make_pair(toolName, globalTools.at(toolName));
}

/// Parse the general options and options for the selected tool in one go:
void parseOptions(int argc, char** argv,
                  boost::program_options::options_description& toolOptionsDesc,
                  boost::program_options::variables_map& args) {
  namespace po = boost::program_options;
  po::options_description desc("General Options");

  desc.add_options()
  ("help", "Show help for the specified tool."
  )
  ("model",
   po::bool_switch()->default_value(false),
   "If set then use IPU model instead of hardware."
  )
  ("ipus",
   po::value<std::size_t>()->default_value(1),
   "Number of IPUs to use."
  )
  ("replicas",
   po::value<std::size_t>()->default_value(1),
   "Number of replicas."
  )
  ("save-exe",
   po::value<std::string>()->default_value(""),
   "Save the Poplar graph executable after compilation using this name (prefix)."
  )
  ("load-exe",
   po::value<std::string>()->default_value(""),
   "Load a previously saved executable with this name (prefix) and skip graph and program construction. "
  )
  ("compile-only", po::bool_switch()->default_value(false),
   "If set and save-exe is also set then exit after compiling and saving the graph."
  )
  ("attach-immediately", po::bool_switch()->default_value(false),
  "If false (default) then the device is not acquired until the program is ready to run, if true then the device is acquired before compilation but this does not currently work on IPUOF systems (program will abort)."
  )
  ("log-level", po::value<std::string>()->default_value("debug"),
  "Set the log level to one of the following: 'trace', 'debug', 'info', 'warn', 'err', 'critical', 'off'.")
  ;

  po::options_description all("All Options");
  all.add(desc).add(toolOptionsDesc);

  auto parser = po::command_line_parser(argc, argv).options(all);
  po::store(parser.run(), args);
  if (args.count("help")) {
    std::cout << all << "\n";
    std::exit(0);
  }

  auto saveExe = !args.at("save-exe").as<std::string>().empty();
  auto loadExe = !args.at("load-exe").as<std::string>().empty();
  if (saveExe && loadExe) {
    throw std::logic_error("You can not set both save-exe and load-exe.");
  }

}

ipu_utils::RuntimeConfig configFromOptions(const boost::program_options::variables_map& args) {
  auto exeName = args.at("save-exe").as<std::string>();
  if (exeName.empty()) { exeName = args.at("load-exe").as<std::string>(); }

  return ipu_utils::RuntimeConfig{
    args.at("ipus").as<std::size_t>(),
    args.at("replicas").as<std::size_t>(),
    exeName,
    args.at("model").as<bool>(),
    !args.at("save-exe").as<std::string>().empty(),
    !args.at("load-exe").as<std::string>().empty(),
    args.at("compile-only").as<bool>(),
    args.at("compile-only").as<bool>() || !args.at("attach-immediately").as<bool>()
  };
}

void setupLogging(const boost::program_options::variables_map& args) {
  std::map<std::string, spdlog::level::level_enum> levelFromStr = {
    {"trace", spdlog::level::trace},
    {"debug", spdlog::level::debug},
    {"info", spdlog::level::info},
    {"warn", spdlog::level::warn},
    {"err", spdlog::level::err},
    {"critical", spdlog::level::critical},
    {"off", spdlog::level::off}
  };

  const auto levelStr = args["log-level"].as<std::string>();
  try {
    spdlog::set_level(levelFromStr.at(levelStr));
  } catch (const std::exception& e) {
    std::stringstream ss;
    ss << "Invalid log-level: '" << levelStr << "'";
    throw std::runtime_error(ss.str());
  }
  spdlog::set_pattern("[%H:%M:%S.%f] [%L] [%t] %v");
}

inline std::string makeArgsFileName(const std::string& name) {
    return name + ".poplar.cmd";
}

// Very simple serialisation of command line:
void serialiseCommandLine(std::ostream& os, int argc, char** argv) {
  for (auto i = 0u; i < argc; ++i) {
    os << argv[i] << "\n";
  }
}

// Note: there is no formatting check of the command args file.
void deserialiseAndParseCommandLine(std::istream& is,
                                    boost::program_options::options_description& desc,
                                    boost::program_options::variables_map& result) {
  if (!is.good()) {
    throw std::runtime_error("Bad input file stream");
  }
  std::stringstream ss;
  ss << is.rdbuf();
  auto args = ss.str();
  auto argc = std::count(args.begin(), args.end(), '\n');
  ipu_utils::logger()->trace("Loaded {} args:\n{}", argc, args);
  std::replace(args.begin(), args.end(), '\n', '\0');
  char* subStrPtrs[argc];
  std::size_t p = 0u;
  for (auto i = 0u; i < argc; ++i) {
    subStrPtrs[i] = &args[p];
    auto s = std::string(subStrPtrs[i]);
    p += s.length() + 1;
  }
  parseOptions(argc, &subStrPtrs[0], desc, result);
}

int main(int argc, char** argv) {
  std::string toolName;
  ToolFactoryFunction factoryFunc;
  std::tie(toolName, factoryFunc) = parseToolName(argc, argv);
  std::unique_ptr<ToolInterface> tool = factoryFunc();

  boost::program_options::options_description desc(toolName + " Options");
  tool->addToolOptions(desc);

  boost::program_options::variables_map allOpts;
  parseOptions(argc, argv, desc, allOpts);

  setupLogging(allOpts);
  ipu_utils::logger()->info("Selected tool {}", toolName);

  auto cfg = configFromOptions(allOpts);

  // If executable saving is requested we need to save the command arguments also:
  if (cfg.saveExe) {
    auto fn = makeArgsFileName(cfg.exeName);
    ipu_utils::logger()->info("Exe save requested: saving command args to '{}'", fn);
    std::ofstream fs(fn);
    serialiseCommandLine(fs, argc, argv);
  } else if (cfg.loadExe) {
    auto fn = makeArgsFileName(cfg.exeName);
    ipu_utils::logger()->info("Exe load requested: re-parsing command args from '{}'", fn);
    try {
      std::ifstream fs(fn);
      deserialiseAndParseCommandLine(fs, desc, allOpts);
    } catch (const std::exception& e) {
      ipu_utils::logger()->warn("Error loading command args: '{}'. Continuing but your "
                                "program may give incorrect results or crash if the arguments affect execution.", e.what());
    }
  }

  boost::program_options::notify(allOpts);

  tool->setRuntimeConfig(cfg);
  tool->init(allOpts);

  return ipu_utils::GraphManager().run(tool->getGraphBuilder());
}
