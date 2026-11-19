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
# The default is Longshot's fork of that translator (third_party/shader-ls, GPL-3.0-or-later, see its PROVENANCE files)
# with its own bridge (shader_compiler_ls.cpp): its mesh programs' depth matches their depth pre-pass (the white holes
# in doors, Link's hat, vines) and it carries the task/mesh compute fallback for GPUs without mesh shaders.
# -DOOT_TRANSLATOR=eden builds the earlier one.
set(OOT_TRANSLATOR "longshot" CACHE STRING "Shader translator: longshot (third_party/shader-ls) or eden (third_party/shader)")
if(OOT_TRANSLATOR STREQUAL "longshot")
    set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/shader-ls")
    set(SHADER_COMPAT "${SHADER_DIR}/compat/vk" "${SHADER_DIR}/compat/gpu")
    set(SHADER_BRIDGE runtime/native/nvn/shader_compiler_ls.cpp)
    set(SHADER_FLAGS "")
else()
    set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/shader")
    set(SHADER_COMPAT "${SHADER_DIR}/compat")
    set(SHADER_BRIDGE runtime/native/nvn/shader_compiler.cpp)
    set(SHADER_FLAGS -include "${SHADER_DIR}/compat/fmt_enum_compat.h")
endif()
file(GLOB SIRIT_SOURCES "${SHADER_DIR}/sirit/src/*.cpp" "${SHADER_DIR}/sirit/src/instructions/*.cpp")
add_library(sirit STATIC ${SIRIT_SOURCES})
target_include_directories(sirit PUBLIC "${SHADER_DIR}/sirit/include" "${SHADER_DIR}/spirv-headers/include"
                           PRIVATE "${SHADER_DIR}/sirit/src")
target_compile_options(sirit PRIVATE -O2 -w)
file(GLOB_RECURSE SHADER_RECOMPILER_SOURCES "${SHADER_DIR}/src/shader_recompiler/*.cpp")
list(FILTER SHADER_RECOMPILER_SOURCES EXCLUDE REGEX "/backend/(glsl|glasm)/")
add_library(shader_recompiler STATIC ${SHADER_RECOMPILER_SOURCES}
    "${SHADER_DIR}/src/video_core/textures/astc.cpp")
target_include_directories(shader_recompiler PUBLIC "${SHADER_DIR}/src" ${SHADER_COMPAT})
target_compile_options(shader_recompiler PRIVATE -O2 ${OOT_G} -w -fno-strict-aliasing -fbracket-depth=1024 ${SHADER_FLAGS})
target_compile_definitions(shader_recompiler PUBLIC ARCHITECTURE_x86_64=1 NDEBUG)
target_link_libraries(shader_recompiler PUBLIC sirit fmt::fmt)
add_library(shader_bridge STATIC ${SHADER_BRIDGE})
target_include_directories(shader_bridge PRIVATE runtime/native/nvn)
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
list(REMOVE_ITEM RUNTIME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/runtime/native/nvn/shader_compiler_ls.cpp")
# Longshot's temporal upscalers (NVIDIA DLSS, AMD FSR 3.1 and its frame interpolation), vendored unchanged under
# third_party/longshot-upscale (GPL-3.0-or-later, see its PROVENANCE.md); compat/ stands in for the headers of
# Longshot's renderer, and runtime/native/nvn/longshot_upscale.cpp defines what they declare.
# NVIDIA's DLSS SDK is not part of this project (NVIDIA's licence): -DOOT_NGX_SDK=<folder with include/ and lib/>
# builds DLSS in (tools/ngx_sdk.py fetches it); without it DLSS is left out.
set(LS_UPSCALE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/longshot-upscale")
set(OOT_NGX_SDK "" CACHE PATH "NVIDIA's DLSS SDK (include/, lib/): builds NVIDIA DLSS in")
add_library(longshot_upscale STATIC ${LS_UPSCALE_DIR}/vk_temporal.cpp ${LS_UPSCALE_DIR}/vk_dlss.cpp ${LS_UPSCALE_DIR}/vk_fsr3.cpp
            ${LS_UPSCALE_DIR}/vk_fsr3_inputs.cpp ${LS_UPSCALE_DIR}/vk_fsr3fi.cpp ${LS_UPSCALE_DIR}/vk_fsr4.cpp)
# (AMD FSR 4: only AMD's MIT API headers are here, third_party/fidelityfx; AMD's library is loaded from the user's PC when it is there)
target_include_directories(longshot_upscale PUBLIC ${LS_UPSCALE_DIR}/compat ${LS_UPSCALE_DIR})
target_include_directories(longshot_upscale PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fidelityfx)
target_compile_definitions(longshot_upscale PRIVATE LONGSHOT_FSR=1)
target_compile_options(longshot_upscale PRIVATE -O2 ${OOT_G} -w -fdeclspec)
target_link_libraries(longshot_upscale PUBLIC Vulkan::Vulkan)
if(OOT_NGX_SDK AND EXISTS "${OOT_NGX_SDK}/include/nvsdk_ngx_vk.h")
    target_compile_definitions(longshot_upscale PUBLIC LONGSHOT_NGX=1)
    target_include_directories(longshot_upscale PRIVATE "${OOT_NGX_SDK}/include")
    if(WIN32)
        # (NVIDIA's library is made with Microsoft's compiler: it names three of Microsoft's runtime libraries, which
        # this toolchain does not have, and calls a few of their functions. Empty libraries of those names satisfy the
        # linker, and runtime/native/nvn/ngx_msvc_shim.cpp gives the functions.)
        set(NGX_STUBS "${CMAKE_BINARY_DIR}/ngx-stubs")
        # (made by the toolchain's own archiver: written as text, "!<arch>" and a newline came out as a file the linker of
        # MSYS2's native tools called "unknown file type" on a player's PC)
        file(MAKE_DIRECTORY "${NGX_STUBS}")
        foreach(stub libcpmt LIBCMT OLDNAMES)
            file(REMOVE "${NGX_STUBS}/lib${stub}.a")
            execute_process(COMMAND "${CMAKE_AR}" rc "${NGX_STUBS}/lib${stub}.a" RESULT_VARIABLE made)
            if(NOT made EQUAL 0 OR NOT EXISTS "${NGX_STUBS}/lib${stub}.a")
                file(WRITE "${NGX_STUBS}/lib${stub}.a" "!<arch>\n")
            endif()
        endforeach()
        target_link_directories(longshot_upscale PUBLIC "${NGX_STUBS}")
        target_link_libraries(longshot_upscale PUBLIC "${OOT_NGX_SDK}/lib/nvsdk_ngx_s.lib")
    else()
        target_link_libraries(longshot_upscale PUBLIC "${OOT_NGX_SDK}/lib/libnvsdk_ngx.a" dl)
    endif()
    message(STATUS "NVIDIA DLSS: built in (SDK at ${OOT_NGX_SDK})")
else()
    message(STATUS "NVIDIA DLSS: not built in (no OOT_NGX_SDK)")
endif()
add_executable(tootr ${RUNTIME_SOURCES})
target_include_directories(tootr PRIVATE runtime)
# NVIDIA DLSS 5 neural rendering (runtime/native/nvn/neural_render.cpp): its SDK is not public yet. When it is, like DLSS's
# it is downloaded onto the player's PC under NVIDIA's licence and never part of this project: -DOOT_DLSS_NR_SDK=<folder>.
set(OOT_DLSS_NR_SDK "" CACHE PATH "NVIDIA's DLSS 5 neural rendering SDK: builds its stage in")
if(OOT_DLSS_NR_SDK AND EXISTS "${OOT_DLSS_NR_SDK}/include")
    target_compile_definitions(tootr PRIVATE OOT_DLSS_NR=1)
    target_include_directories(tootr PRIVATE "${OOT_DLSS_NR_SDK}/include")
    message(STATUS "NVIDIA DLSS 5 neural rendering: stage built in (SDK at ${OOT_DLSS_NR_SDK})")
endif()
target_link_libraries(tootr PRIVATE longshot_upscale)
if(OOT_TRANSLATOR STREQUAL "longshot")
    target_compile_definitions(tootr PRIVATE OOT_SHADER_LONGSHOT=1)
endif()
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
