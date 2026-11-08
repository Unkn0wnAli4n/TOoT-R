# Offline tools, run once per install by tools/build_game.py:
#   oot-extract  .nup -> flat module images (main/sdk/subsdk0.img)
#   recomp       flat image -> C++ (dynarmic's A64 frontend used as the instruction decoder)
cmake_minimum_required(VERSION 3.25)
project(tootr_tools C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD REQUIRED IMPORTED_TARGET libzstd)
pkg_check_modules(LZ4 REQUIRED IMPORTED_TARGET liblz4)
find_package(fmt CONFIG REQUIRED)

add_executable(oot-extract "${ROOT}/tools/extract.cpp" "${ROOT}/tools/zbic.cpp")
target_include_directories(oot-extract PRIVATE "${ROOT}/third_party/zbic")
set_source_files_properties("${ROOT}/tools/zbic.cpp" PROPERTIES COMPILE_OPTIONS "-w")
target_link_libraries(oot-extract PRIVATE PkgConfig::ZSTD PkgConfig::LZ4)

# Minimal dynarmic frontend (see third_party/dynarmic/PROVENANCE.md). The A64 translator includes
# its impl/*.cpp files itself.
set(DYN "${ROOT}/third_party/dynarmic/src/dynarmic")
file(GLOB_RECURSE DYNARMIC_SOURCES "${DYN}/*.cpp")
list(FILTER DYNARMIC_SOURCES EXCLUDE REGEX "/frontend/A64/translate/impl/")
add_library(dynarmic_frontend STATIC ${DYNARMIC_SOURCES})
target_include_directories(dynarmic_frontend PUBLIC "${ROOT}/third_party/dynarmic/src" "${DYN}"
                           "${ROOT}/third_party/dynarmic/compat")
target_compile_definitions(dynarmic_frontend PUBLIC ARCHITECTURE_x86_64=1 MCL_IGNORE_ASSERTS=1)
target_compile_options(dynarmic_frontend PRIVATE -w)
target_link_libraries(dynarmic_frontend PUBLIC fmt::fmt)

add_executable(recomp recomp.cpp dynarmic_hooks.cpp)
target_link_libraries(recomp PRIVATE dynarmic_frontend)
