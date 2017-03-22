set(TEST_NAME tenncortest)
set(TEST_MAIN "" CACHE STRING "path to test main")
enable_testing()

add_definitions(${MSVC_COMPILER_DEFS})
include_directories(${GTEST_INCLUDE_DIRS} ${TENNCOR_INCLUDE} include)
file(GLOB_RECURSE TEST_SRC src/*.cpp)

add_executable(${TEST_NAME} ${TEST_SRC} ${TEST_MAIN})
add_dependencies(${TEST_NAME} googletest ${TENNCOR_LIB_TEST})

# link googletest depending on  available libraries
if(NOT WIN32 OR MINGW)
    target_link_libraries(${TEST_NAME}
        ${GTEST_LIBS_DIR}/libgtest.a
        ${GTEST_LIBS_DIR}/libgtest_main.a )
else()
    target_link_libraries(${TEST_NAME}
        debug ${GTEST_LIBS_DIR}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${CMAKE_FIND_LIBRARY_SUFFIXES}
        optimized ${GTEST_LIBS_DIR}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${CMAKE_FIND_LIBRARY_SUFFIXES} )
    target_link_libraries(${TEST_NAME}
        debug ${GTEST_LIBS_DIR}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${CMAKE_FIND_LIBRARY_SUFFIXES}
        optimized ${GTEST_LIBS_DIR}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${CMAKE_FIND_LIBRARY_SUFFIXES} )
endif()

target_link_libraries(${TEST_NAME}
        pthread gtest gmock
        ${CMAKE_THREAD_LIBS_INIT}
        ${TENNCOR_LIB_TEST})
add_test(test ${TEST_NAME})
