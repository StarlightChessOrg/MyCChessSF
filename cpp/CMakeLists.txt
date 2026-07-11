cmake_minimum_required(VERSION 3.18)
project(xqwlight_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

message(STATUS "xqwlight_core: using Python_EXECUTABLE=${Python_EXECUTABLE}")

option(
  MYCCHESSSF_FETCH_PYBIND11
  "If pybind11 is missing, download release tarball from GitHub (needs HTTPS; can be slow)"
  OFF)

set(_pybind11_search_paths
    "${CMAKE_CURRENT_SOURCE_DIR}/pybind11-master"
    "${CMAKE_CURRENT_SOURCE_DIR}/../MyCChessRL/cpp/pybind11-master"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../MyCChessRL/cpp/pybind11-master")

set(_pybind11_from_subdir FALSE)
foreach(_root IN LISTS _pybind11_search_paths)
  if(EXISTS "${_root}/CMakeLists.txt")
    message(STATUS "xqwlight_core: using local pybind11 at ${_root}")
    add_subdirectory("${_root}" "${CMAKE_CURRENT_BINARY_DIR}/pybind11-build")
    set(_pybind11_from_subdir TRUE)
    break()
  endif()
endforeach()

if(NOT _pybind11_from_subdir)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _p11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _p11_err
    RESULT_VARIABLE _p11_rc)
  if(_p11_rc EQUAL "0" AND EXISTS "${_p11_cmake_dir}/pybind11Config.cmake")
    message(STATUS "xqwlight_core: using pip pybind11 (pybind11_DIR=${_p11_cmake_dir})")
    set(pybind11_DIR "${_p11_cmake_dir}")
    find_package(pybind11 CONFIG REQUIRED)
  elseif(MYCCHESSSF_FETCH_PYBIND11)
    include(FetchContent)
    FetchContent_Declare(
      pybind11 URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz)
    FetchContent_MakeAvailable(pybind11)
  else()
    message(
      FATAL_ERROR
      "xqwlight_core: 未找到 pybind11。\n"
      "  pip install pybind11\n"
      "  或复制 MyCChessRL/cpp/pybind11-master 到本目录 cpp/\n"
      "  或 cmake -DMYCCHESSSF_FETCH_PYBIND11=ON\n"
      "  import 错误: ${_p11_err}")
  endif()
endif()

pybind11_add_module(xqwlight_core MODULE bindings.cpp)

target_include_directories(xqwlight_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_compile_options(xqwlight_core PRIVATE -Wall -Wextra -Wno-sign-compare -fPIC)

install(TARGETS xqwlight_core LIBRARY DESTINATION .)
