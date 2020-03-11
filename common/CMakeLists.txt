set(COMMON_SOURCES 
    defs.h
    render.h
    render.cu 
)

add_library(common STATIC ${COMMON_SOURCES})
set_target_properties(common PROPERTIES
LINKER_LANGUAGE CUDA
CUDA_STANDARD 11
CUDA_RESOLVE_DEVICE_SYMBOLS ON
CMAKE_CUDA_SEPARABLE_COMPILATION ON
POSITION_INDEPENDENT_CODE OFF
        OUTPUT_NAME "common"
        FOLDER "COMMON")
target_include_directories(common PUBLIC 
    ${PONOS_INCLUDES} 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    )