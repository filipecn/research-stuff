enable_language("CUDA")

set(HEADERS
    blas/pcg2.h
    colliders/collider2.h
    colliders/box_collider2.h
    colliders/sphere_collider2.h
    integrators/integrator2.h
    integrators/semi_lagrangian_integrator2.h
    integrators/maccormack_integrator2.h
    injectors/smoke_injector2.h
    smoke_solver2.h
    smoke_solver2_steps.h
)

set(SOURCES
    blas/pcg2.cu
    integrators/semi_lagrangian_integrator2.cu
    integrators/maccormack_integrator2.cu
    scene2.h
    injectors/smoke_injector2.cu
    smoke_solver2.cu
    smoke_solver2_steps.cu
)

# 2d
add_library(cuSmoke2d STATIC ${SOURCES} ${HEADERS})

add_dependencies(cuSmoke2d ponos)

target_compile_options(cuSmoke2d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
--generate-line-info
--use_fast_math
-arch=sm_50
-Xcompiler -pg 
--relocatable-device-code=true
#    -–extended-lambda
>)

set_target_properties(cuSmoke2d PROPERTIES
LINKER_LANGUAGE CUDA
CMAKE_CUDA_SEPARABLE_COMPILATION ON
CUDA_RESOLVE_DEVICE_SYMBOLS ON
CUDA_STANDARD 11
CMAKE_CUDA_STANDARD_REQUIRED ON
CXX_STANDARD 11
CXX_STANDARD_REQUIRED ON
)
target_include_directories(cuSmoke2d PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${COMMON_INCLUDES}
    ${PONOS_INCLUDES}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
target_link_libraries(cuSmoke2d PRIVATE common 
 ${PONOS_LIBRARIES} 
 gcov
cufft)



# APP
add_executable(cuSmoke2d_app main2.cu
${SOURCES} 
${HEADERS}
${COMMON_SOURCES}
)
add_dependencies(cuSmoke2d_app ponos)

target_compile_options(cuSmoke2d_app PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
--generate-line-info
--use_fast_math
-arch=sm_50
-Xcompiler -pg 
--relocatable-device-code=true
#    -–extended-lambda
>)

set_target_properties(cuSmoke2d_app PROPERTIES
LINKER_LANGUAGE CUDA
CMAKE_CUDA_SEPARABLE_COMPILATION ON
CUDA_RESOLVE_DEVICE_SYMBOLS ON
CUDA_STANDARD 11
CMAKE_CUDA_STANDARD_REQUIRED ON
CXX_STANDARD 11
CXX_STANDARD_REQUIRED ON
)
target_include_directories(cuSmoke2d_app PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${COMMON_INCLUDES}
    ${PONOS_INCLUDES}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
target_link_libraries(cuSmoke2d_app PRIVATE common 
 ${PONOS_LIBRARIES} 
 X11
 gcov
cufft)