cmake_minimum_required(VERSION 3.20)
project(gpu_profiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE glslc REQUIRED)

set(SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})
file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/shaders/*.comp)

set(SHADER_BINARIES "")
foreach(SHADER_SOURCE IN LISTS SHADER_SOURCES)
  get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
  set(SHADER_BINARY ${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv)
  add_custom_command(
    OUTPUT ${SHADER_BINARY}
    COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.2 -Werror -o ${SHADER_BINARY} ${SHADER_SOURCE}
    DEPENDS ${SHADER_SOURCE}
    COMMENT "Compiling shader ${SHADER_NAME}"
    VERBATIM)
  list(APPEND SHADER_BINARIES ${SHADER_BINARY})
endforeach()
add_custom_target(shaders ALL DEPENDS ${SHADER_BINARIES})

add_executable(gpu_profiler src/main.cpp src/vk_context.cpp src/buffers.cpp src/simulation.cpp
                            src/report.cpp)
add_dependencies(gpu_profiler shaders)
target_compile_options(gpu_profiler PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(gpu_profiler PRIVATE Vulkan::Vulkan)
# Lets the program find compiled shaders whatever folder it is run from.
target_compile_definitions(gpu_profiler PRIVATE GPU_PROFILER_SHADER_DIR="${SHADER_OUTPUT_DIR}")

# Homebrew's validation layer manifest names its library without a folder, so the
# program must search Homebrew's library folder when the loader opens the layer.
if(APPLE)
  get_filename_component(VULKAN_LIBRARY_DIR ${Vulkan_LIBRARY} DIRECTORY)
  set_target_properties(gpu_profiler PROPERTIES BUILD_RPATH ${VULKAN_LIBRARY_DIR})
endif()

enable_testing()
add_test(NAME info COMMAND gpu_profiler --info)
foreach(TEST_NAME IN ITEMS clean_shutdown validation_live validation_leak vector_add
                           force_naive force_tiled momentum validation_canary timestamps
                           stats batch_equivalence validation_modes)
  add_test(NAME ${TEST_NAME} COMMAND gpu_profiler --test ${TEST_NAME})
endforeach()
