cmake_minimum_required(VERSION 3.25)
project(tootr CXX)

# The game code is C++ generated from the user's own copy of the game by tools/build_game.py; it is
# never part of the repository. Either build it from GEN_DIR, or link a game_code archive built
# earlier (OOT_GAME_LIB) so runtime-only updates do not recompile it.
set(GEN_DIR "" CACHE PATH "Generated code directory (main/, sdk/, subsdk0/)")
set(OOT_GAME_LIB "" CACHE FILEPATH "Prebuilt game_code static archive")
option(OOT_GAME_ONLY "Only build the game_code archive" OFF)

option(OOT_DEBUG_INFO "Debug info in the runtime (tools/dev.sh turns it on)" OFF)
if(OOT_DEBUG_INFO)
    set(OOT_G -g)
endif()

include(cmake/GameCode.cmake)
if(OOT_GAME_ONLY)
    return()
endif()

if(WIN32)
    # (MSYS2 CLANG64) 64-bit file offsets; lean Windows headers; Winsock.
    add_compile_definitions(_FILE_OFFSET_BITS=64 NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL3 REQUIRED IMPORTED_TARGET sdl3)
pkg_check_modules(OPUS REQUIRED IMPORTED_TARGET opus)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavcodec libavutil libswscale) # (movies: nn::codec on FFmpeg/dav1d)
find_package(Vulkan REQUIRED)
find_package(fmt CONFIG REQUIRED)
# Shader compiler (Eden/yuzu shader_recompiler with this project's NVN2/Ampere changes), vendored under third_party/shader with sirit, the SPIR-V headers
# and Hookshot's minimal common/boost shims (see third_party/shader/PROVENANCE.md). GPL-3.0-or-later.
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/shader")
file(GLOB SIRIT_SOURCES "${SHADER_DIR}/sirit/src/*.cpp" "${SHADER_DIR}/sirit/src/instructions/*.cpp")
add_library(sirit STATIC ${SIRIT_SOURCES})
target_include_directories(sirit PUBLIC "${SHADER_DIR}/sirit/include" "${SHADER_DIR}/spirv-headers/include"
                           PRIVATE "${SHADER_DIR}/sirit/src")
target_compile_options(sirit PRIVATE -O2 -w)
file(GLOB_RECURSE SHADER_RECOMPILER_SOURCES "${SHADER_DIR}/src/shader_recompiler/*.cpp")
list(FILTER SHADER_RECOMPILER_SOURCES EXCLUDE REGEX "/backend/(glsl|glasm)/")
add_library(shader_recompiler STATIC ${SHADER_RECOMPILER_SOURCES}
    "${SHADER_DIR}/src/video_core/textures/astc.cpp")
target_include_directories(shader_recompiler PUBLIC "${SHADER_DIR}/src" "${SHADER_DIR}/compat")
target_compile_options(shader_recompiler PRIVATE -O2 ${OOT_G} -w -fno-strict-aliasing -fbracket-depth=1024
    -include "${SHADER_DIR}/compat/fmt_enum_compat.h")
target_compile_definitions(shader_recompiler PUBLIC ARCHITECTURE_x86_64=1 NDEBUG)
target_link_libraries(shader_recompiler PUBLIC sirit fmt::fmt)
add_library(shader_bridge STATIC runtime/native/nvn/shader_compiler.cpp)
target_compile_options(shader_bridge PRIVATE -O1 ${OOT_G} -fno-strict-aliasing)
target_link_libraries(shader_bridge PUBLIC shader_recompiler fmt::fmt pthread lz4 zstd)

# Dear ImGui (third_party/imgui, MIT): the graphics settings window (Insert).
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui")
add_library(imgui STATIC "${IMGUI_DIR}/imgui.cpp" "${IMGUI_DIR}/imgui_draw.cpp" "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp" "${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp" "${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp")
target_include_directories(imgui PUBLIC "${IMGUI_DIR}" "${IMGUI_DIR}/backends")
target_compile_options(imgui PRIVATE -O2 -w)
target_link_libraries(imgui PUBLIC PkgConfig::SDL3 Vulkan::Vulkan)

file(GLOB_RECURSE RUNTIME_SOURCES CONFIGURE_DEPENDS runtime/*.cpp)
list(REMOVE_ITEM RUNTIME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/runtime/native/nvn/shader_compiler.cpp")
add_executable(tootr ${RUNTIME_SOURCES})
target_include_directories(tootr PRIVATE runtime)
target_compile_options(tootr PRIVATE -O2 ${OOT_G} ${ARCH_FLAGS} -Wall -Wno-unused-function -fno-omit-frame-pointer)
target_link_libraries(tootr PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,game_code>" PkgConfig::SDL3 PkgConfig::OPUS PkgConfig::FFMPEG Vulkan::Vulkan shader_bridge imgui pthread $<$<NOT:$<BOOL:${WIN32}>>:atomic>)
# The renderer allocates many small objects per draw; jemalloc's thread caches make that much
# cheaper than glibc malloc (measured ~15-20% more frames per second).
if(NOT WIN32)
    find_library(JEMALLOC_LIBRARY NAMES jemalloc)
endif()
if(JEMALLOC_LIBRARY)
    target_link_libraries(tootr PRIVATE ${JEMALLOC_LIBRARY})
endif()
target_link_options(tootr PRIVATE -fuse-ld=lld)
if(WIN32)
    # Guest threads run recompiled code whose frames live on the host stack: give host threads the
    # 8+ MiB a Linux thread gets (address space only).
    target_link_options(tootr PRIVATE -Wl,--stack,0x1000000)
    target_link_libraries(tootr PRIVATE ws2_32 psapi)
else()
    target_link_options(tootr PRIVATE -Wl,--no-relax)
endif()

install(TARGETS tootr RUNTIME DESTINATION .)
