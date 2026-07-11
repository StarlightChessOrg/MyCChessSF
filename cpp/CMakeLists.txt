cmake_minimum_required(VERSION 3.18)
project(xqwlight_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

message(STATUS "xqwlight_core: Python_EXECUTABLE=${Python_EXECUTABLE}")

option(
  MYCCHESSSF_FETCH_PYBIND11
  "Download pybind11 v2.13.6 from GitHub if not found locally"
  OFF)

set(_pybind11_search_paths
    "${CMAKE_CURRENT_SOURCE_DIR}/pybind11-master"
    "${CMAKE_CURRENT_SOURCE_DIR}/../MyCChessRL/cpp/pybind11-master"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../MyCChessRL/cpp/pybind11-master")

set(_pybind11_from_subdir FALSE)
foreach(_root IN LISTS _pybind11_search_paths)
  if(EXISTS "${_root}/CMakeLists.txt")
    message(STATUS "xqwlight_core: pybind11 at ${_root}")
    add_subdirectory("${_root}" "${CMAKE_CURRENT_BINARY_DIR}/pybind11-build")
    set(_pybind11_from_subdir TRUE)
    break()
  endif()
endforeach()

if(NOT _pybind11_from_subdir)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE _p11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _p11_cli_rc)
  if(NOT _p11_cli_rc EQUAL "0" OR NOT EXISTS "${_p11_cmake_dir}/pybind11Config.cmake")
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
      OUTPUT_VARIABLE _p11_cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_VARIABLE _p11_err
      RESULT_VARIABLE _p11_rc)
  else()
    set(_p11_rc 0)
  endif()
  if(_p11_rc EQUAL "0" AND EXISTS "${_p11_cmake_dir}/pybind11Config.cmake")
    message(STATUS "xqwlight_core: pip pybind11 (${_p11_cmake_dir})")
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
      "xqwlight_core: pybind11 not found for ${Python_EXECUTABLE}.\n"
      "  ${Python_EXECUTABLE} -m pip install pybind11\n"
      "  or bash scripts/build_linux.sh (installs pybind11 automatically)\n"
      "  or vendor cpp/pybind11-master\n"
      "  or cmake -DMYCCHESSSF_FETCH_PYBIND11=ON\n"
      "  (${_p11_err})")
  endif()
endif()

set(_xqnnue_simd_flags "")
set(_xqnnue_vnni_sources "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64|armv8)")
  message(STATUS "xqwlight_core: ARM target, NEON dispatch when available")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i686|x86)")
  set(_xqnnue_simd_flags -mavx2 -mfma)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mavx512f" _has_avx512f)
  check_cxx_compiler_flag("-mavx512vnni" _has_avx512vnni)
  check_cxx_compiler_flag("-mavx512vl" _has_avx512vl)
  if(_has_avx512f AND _has_avx512vnni AND _has_avx512vl)
    list(APPEND _xqnnue_vnni_sources xqwl_nnue_simd_vnni.cpp)
    message(STATUS "xqwlight_core: x86 AVX2 + optional AVX512-VNNI TU")
  else()
    message(STATUS "xqwlight_core: x86 AVX2 (no AVX512-VNNI TU)")
  endif()
else()
  message(STATUS "xqwlight_core: CPU '${CMAKE_SYSTEM_PROCESSOR}', scalar NNUE fallback")
endif()

set(_xqwl_include_dirs
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/inc"
    "${CMAKE_CURRENT_SOURCE_DIR}")

pybind11_add_module(xqwlight_core MODULE bindings.cpp ${_xqnnue_vnni_sources})

target_include_directories(xqwlight_core PRIVATE ${_xqwl_include_dirs})
target_compile_options(xqwlight_core PRIVATE -Wall -Wextra -Wno-sign-compare -fPIC ${_xqnnue_simd_flags})
if(_xqnnue_vnni_sources)
  set_source_files_properties(xqwl_nnue_simd_vnni.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512vnni -mavx512vl")
  target_compile_definitions(xqwlight_core PRIVATE MYCCHESSSF_HAS_VNNI=1)
endif()

add_executable(xqwl_gen_nnue gen_nnue_main.cpp ${_xqnnue_vnni_sources})
target_include_directories(xqwl_gen_nnue PRIVATE ${_xqwl_include_dirs})
target_compile_options(xqwl_gen_nnue PRIVATE -Wall -Wextra -Wno-sign-compare ${_xqnnue_simd_flags})
if(_xqnnue_vnni_sources)
  set_source_files_properties(xqwl_nnue_simd_vnni.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512vnni -mavx512vl")
  target_compile_definitions(xqwl_gen_nnue PRIVATE MYCCHESSSF_HAS_VNNI=1)
endif()

install(TARGETS xqwlight_core LIBRARY DESTINATION .)
install(TARGETS xqwl_gen_nnue RUNTIME DESTINATION .)
