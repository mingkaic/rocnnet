cmake_minimum_required(VERSION 2.8)
project(tenncor C CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)
option(RunTest "Test" OFF)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-Wno-reorder -Wall -ansi -Wno-deprecated -pthread)
endif()

if (MSVC)
    #vc 2012 fix for vararg templates
    set(MSVC_COMPILER_DEFS "-D_VARIADIC_MAX=10")
endif()

#>>>>>>>>>>>>>>>>>>>> TENNCOR SPECIFIC <<<<<<<<<<<<<<<<<<<<
# TODO: make option
#ADD_DEFINITIONS(-DEDGE_RCD="op-profile.csv")

set(TENNCOR_INCLUDE include)
FILE(GLOB_RECURSE TENNCOR_SRC src/*.cpp)

# set tenncor include path
include_directories(${TENNCOR_INCLUDE})

set(TENNCOR_LIB tenncor)
add_library(${TENNCOR_LIB} STATIC ${TENNCOR_SRC})

#>>>>>>>>>>>>>>>>>>>> GOOGLE TEST <<<<<<<<<<<<<<<<<<<<
if (RunTest)
    set(TEST_NAME tenncortest)
    set(TEST_MAIN "" CACHE STRING "path to test main")

    add_subdirectory(ext/gtest)
    enable_testing()

    add_definitions(${MSVC_COMPILER_DEFS})
    include_directories(${GTEST_INCLUDE_DIRS} ${TENNCOR_INCLUDE} tests/include)
    file(GLOB_RECURSE TEST_SRC tests/src/*.cpp)

    add_executable(${TEST_NAME} ${TEST_SRC} ${TEST_MAIN})
    add_dependencies(${TEST_NAME} googletest ${TENNCOR_LIB})

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
        ${TENNCOR_LIB})
    add_test(test ${TEST_NAME})
endif()
