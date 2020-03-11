set(SOURCES 
    main.cpp
    openvdb_tests.cpp
    vdb_io_tests.cpp
    vdb_particle_system_tests.cpp
    vdb_velocity_grid_tests.cpp
)

add_executable(openvdb_flip_tests ${SOURCES})
target_link_options(openvdb_flip_tests INTERFACE --coverage)

target_include_directories(openvdb_flip_tests PUBLIC 
${PONOS_INCLUDES}
${CATCH2_INCLUDES} 
${VDBLIB_INCLUDE_DIR}
)
add_dependencies(openvdb_flip_tests ponos vdblib)

if(UNIX)

target_link_libraries(openvdb_flip_tests ${PONOS_LIBRARIES} vdblib gcov)
                       
add_custom_target(test_openvdb_flip
        COMMAND lcov --directory  .. --zerocounters
        COMMAND openvdb_flip_tests DEPENDS openvdb_flip_tests
        COMMAND lcov --directory .. --capture -output-file cov.info
        COMMAND lcov --remove cov.info '/usr/*' --output-file cov.info
        COMMAND lcov --remove cov.info '*/ext/*' --output-file cov.info
        COMMAND lcov --remove cov.info '*/tests/*' --output-file cov.info
        COMMAND genhtml cov.info --output-directory out
        )

else(UNIX)

target_link_libraries(openvdb_flip_tests ponos vdblib)

endif(UNIX)
