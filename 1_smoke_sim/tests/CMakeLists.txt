set(SOURCES 
    main.cu
    blas_tests.cu
    simulation_steps_tests.cu
)

add_executable(smoke_sim_tests ${SOURCES})
target_include_directories(smoke_sim_tests PUBLIC 
${COMMON_INCLUDES}
${PONOS_INCLUDES}
${CATCH2_INCLUDES} 
${SMOKE_SIM_SOURCE_DIR})
add_dependencies(smoke_sim_tests ponos cuSmoke2d)

target_compile_options(smoke_sim_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
--generate-line-info
--use_fast_math
-arch=sm_50
-Xcompiler -pg 
--relocatable-device-code=true
#    -–extended-lambda
>)

set_target_properties(smoke_sim_tests PROPERTIES
LINKER_LANGUAGE CUDA
CMAKE_CUDA_SEPARABLE_COMPILATION ON
CUDA_RESOLVE_DEVICE_SYMBOLS ON
CUDA_STANDARD 11
CMAKE_CUDA_STANDARD_REQUIRED ON
CXX_STANDARD 11
CXX_STANDARD_REQUIRED ON
)
                       
if(UNIX)

target_link_libraries(smoke_sim_tests ${PONOS_LIBRARIES} cuSmoke2d gcov)
                       
add_custom_target(test_smoke_sim
        COMMAND lcov --directory  . --zerocounters
        COMMAND smoke_sim_tests DEPENDS smoke_sim_tests
        COMMAND lcov --directory . --capture -output-file cov.info
        COMMAND lcov --remove cov.info '/usr/*' --output-file cov.info
        COMMAND lcov --remove cov.info '*/ext/*' --output-file cov.info
        COMMAND lcov --remove cov.info '*/tests/*' --output-file cov.info
        COMMAND genhtml cov.info --output-directory out
        )

else(UNIX)

target_link_libraries(smoke_sim_tests ponos smoke_sim)

endif(UNIX)
