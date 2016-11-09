cmake_minimum_required(VERSION 3.6)
set(PROJECT_NAME_STR cnnet)
project(${PROJECT_NAME_STR} C CXX)

find_package(Threads REQUIRED)

set(CMAKE_BINARY_DIR ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#-------------------
# Other flags
#-------------------
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-Wall -ansi -Wno-deprecated -pthread)
endif()

if (MSVC)
    #vc 2012 fix for vararg templates
    set(MSVC_COMPILER_DEFS "-D_VARIADIC_MAX=10")
endif()

#-------------------
# set compilation flag
#-------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

#-------------------
# Build TENNCOR Library
#-------------------
include(${PROJECT_SOURCE_DIR}/tenncor/CMakeLists.txt)

#-------------------
# set common include folder for module
#-------------------
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include)
set(EXT_PROJECTS_DIR ${PROJECT_SOURCE_DIR}/ext)

#-------------------
# Module source
#-------------------
include_directories(${COMMON_INCLUDES} ${TENNCOR_INCLUDE_DIR})

add_subdirectory(${EXT_PROJECTS_DIR}/gtest)
#add_subdirectory(${EXT_PROJECTS_DIR}/gbenchmark)

#-------------------
# Google Test Setup
#-------------------
add_definitions(${MSVC_COMPILER_DEFS})
enable_testing()

file(GLOB ROC_SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)

function(GoogleTest PROJECT_TEST_NAME TEST_DIR)
    include_directories(${GTEST_INCLUDE_DIRS} ${COMMON_INCLUDES})

    file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/tests/${TEST_DIR}/*.cpp)

    add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES} ${ROC_SRC})

    # Test dependencies
    add_dependencies(${PROJECT_TEST_NAME}
        googletest)
        #googlebenchmark)

    if(NOT WIN32 OR MINGW)
        target_link_libraries(${PROJECT_TEST_NAME}
                ${GTEST_LIBS_DIR}/libgtest.a
                ${GTEST_LIBS_DIR}/libgtest_main.a
                )
    else()
        target_link_libraries(${PROJECT_TEST_NAME}
                debug ${GTEST_LIBS_DIR}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${CMAKE_FIND_LIBRARY_SUFFIXES}
                optimized ${GTEST_LIBS_DIR}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${CMAKE_FIND_LIBRARY_SUFFIXES}
                )
        target_link_libraries(${PROJECT_TEST_NAME}
                debug ${GTEST_LIBS_DIR}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${CMAKE_FIND_LIBRARY_SUFFIXES}
                optimized ${GTEST_LIBS_DIR}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${CMAKE_FIND_LIBRARY_SUFFIXES}
                )
    endif()

    # Link
    target_link_libraries(${PROJECT_TEST_NAME} ${PROJECT_LIB_NAME} ${CMAKE_THREAD_LIBS_INIT})

    add_test(test ${PROJECT_TEST_NAME})
endfunction(GoogleTest)

GoogleTest(tenncor_test tenncor)
GoogleTest(rocnnet_test rocnnet)
GoogleTest(dqn_test rocnnet_dqn)
GoogleTest(unit_test unit_test)