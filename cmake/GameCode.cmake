# Flags and rules for the game code (part of the game_code archive cache key in tools/build_game.py:
# editing this file rebuilds the archive).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ARCH_FLAGS -msse4.2 -maes -mpclmul -mf16c -mcx16)
include(CheckCXXSourceRuns)
if(NOT CMAKE_CROSSCOMPILING)
    check_cxx_source_runs("int main(){__builtin_cpu_init();return __builtin_cpu_supports(\"fma\")?0:1;}" OOT_HOST_HAS_FMA)
endif()
option(OOT_USE_HOST_FMA "Use hardware FMA on a compatible host CPU" ${OOT_HOST_HAS_FMA})
if(OOT_USE_HOST_FMA)
    list(APPEND ARCH_FLAGS -mfma)
endif()
# Preserve separate guest multiply/add rounding; explicit std::fma remains fused.
list(APPEND ARCH_FLAGS -ffp-contract=off)

if(OOT_GAME_LIB)
    add_library(game_code STATIC IMPORTED)
    set_target_properties(game_code PROPERTIES IMPORTED_LOCATION "${OOT_GAME_LIB}")
else()
    if(NOT GEN_DIR)
        message(FATAL_ERROR "Set GEN_DIR (generated code) or OOT_GAME_LIB; tools/build_game.py does this.")
    endif()
    file(GLOB GAME_SOURCES CONFIGURE_DEPENDS
         "${GEN_DIR}/main/*.cpp" "${GEN_DIR}/sdk/*.cpp" "${GEN_DIR}/subsdk0/*.cpp")
    add_library(game_code STATIC ${GAME_SOURCES})
    target_include_directories(game_code PRIVATE runtime)
    # -fno-slp-vectorize: generated code keeps guest registers in Cpu; merging two 8-byte register
    # loads into one 16-byte load right after the caller's 8-byte stores (x29/x30 in every prologue)
    # defeats store forwarding.
    target_compile_options(game_code PRIVATE -O2 -w ${ARCH_FLAGS} -fno-strict-aliasing -fno-slp-vectorize)
    # Single huge functions: optimization time grows superlinearly, so build them at -O1.
    foreach(src ${GAME_SOURCES})
        if(src MATCHES "_big_[0-9]+\\.cpp$")
            set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "-O1")
        endif()
    endforeach()
endif()
