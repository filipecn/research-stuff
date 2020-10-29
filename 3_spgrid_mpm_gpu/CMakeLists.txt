enable_language("CUDA")

set(SPGRID_SOURCES 
    SPGrid/SPGrid_Mask.h
    SPGrid/SPGrid_Utilities.h
    SPGrid/SPGrid_Set.h
    SPGrid/SPGrid_Array.h
    SPGrid/std_array.h
    SPGrid/SPGrid_Geometry.h
    SPGrid/SPGrid_Allocator.h
    SPGrid/SPGrid_Utilities.cpp
    SPGrid/SPGrid_Geometry.cpp
)
add_library(spgrid STATIC ${SPGRID_SOURCES})
set(SPGRID_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/SPGrid)
target_include_directories(spgrid PUBLIC ${SPGRID_INCLUDE})

add_executable(hello_spgrid 
    main.cu 
    grid_domain.h
    grid_domain.cpp
    )
add_dependencies(hello_spgrid ponos spgrid)
target_include_directories(hello_spgrid PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SPGRID_INCLUDE}
    ${PONOS_INCLUDES}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(hello_spgrid 
    ${PONOS_LIBRARIES} gcov cudart
    spgrid
    cudadevrt
    pthread stdc++)
target_compile_options(hello_spgrid PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        -arch=sm_50
        -Xcompiler -pg 
        --relocatable-device-code=true
        #    -–extended-lambda
    >)

set_target_properties(hello_spgrid PROPERTIES
    LINKER_LANGUAGE CUDA
    CMAKE_CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 11
    CMAKE_CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
)
