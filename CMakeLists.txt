#>>>>>>>>>>>>>>>>>>>>>>>> GENERAL SETUP COMMENCE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
cmake_minimum_required(VERSION 3.6)
set(PROJECT_NAME_STR rocnnet)
project(${PROJECT_NAME_STR} C CXX)

find_package(Threads REQUIRED)

set(CMAKE_BINARY_DIR ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(UNIT_MAIN "" CACHE STRING "path to unit-test main")
set(PERFORM_MAIN "" CACHE STRING "path to performance-test main")

#-------------------
# Options
#-------------------
OPTION (OFFLINE "Building from offline?" OFF)
OPTION (BUILD_DOC "Build doxygen documents?" OFF)

#-------------------
# Other flags
#-------------------
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-Wno-reorder -Wall -ansi -Wno-deprecated -pthread)
endif()

if (MSVC)
    #vc 2012 fix for vararg templates
    set(MSVC_COMPILER_DEFS "-D_VARIADIC_MAX=10")
endif()

#-------------------
# set compilation flag
#-------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set (CMAKE_CXX_STANDARD 14)

#-------------------
# set common include folder for module
#-------------------
set(EXT_PROJECTS_DIR ${PROJECT_SOURCE_DIR}/ext)
#>>>>>>>>>>>>>>>>>>>>>>>> GENERAL SETUP COMPLETE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>> EXTERNAL SETUP COMMENCE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
if (OFFLINE)
    # find local externals
    #-------------------
    # adding googletest
    #-------------------
    find_package(GTest REQUIRED)
    if(BUILD_DOC)
        #-------------------
        # adding doxygen
        #-------------------
        find_package(Doxygen QUIET)
    endif(BUILD_DOC)
else(OFFLINE)
    # setup externals using in ext directory cmakelists
    #-------------------
    # adding googletest
    #-------------------
    add_subdirectory(${EXT_PROJECTS_DIR}/gtest)
    if(BUILD_DOC)
        #-------------------
        # adding doxygen
        #-------------------
        add_subdirectory(${EXT_PROJECTS_DIR}/doxygen)
    endif(BUILD_DOC)
endif(OFFLINE)
#>>>>>>>>>>>>>>>>>>>>>>>> EXTERNAL SETUP COMPLETE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>> DOXYGEN SETUP COMMENCE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
if(BUILD_DOC)
    if(DOXYGEN_FOUND)
        # set input and output files
        set(DOXYGEN_IN ${PROJECT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${PROJECT_SOURCE_DIR}/src/Doxyfile)

        message("Doxygen build started")
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        add_customer_target(doc_doxygen ALL
                COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Generating API Documentation with Doxygen"
                VERBATIM)
    else (DOXYGEN_FOUND)
        message(WARNING "Doxygen need to be installed to generate the doxygen documetation")
    endif(DOXYGEN_FOUND)
endif(BUILD_DOC)
#>>>>>>>>>>>>>>>>>>>>>>>> DOXYGEN SETUP COMPLETE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>> BUILD TENNCOR LIBRARY <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#-------------------
# set tenncor paths
#-------------------
set(TENNCOR_DIR ${PROJECT_SOURCE_DIR}/tenncor)
set(TENNCOR_INCLUDE ${TENNCOR_DIR}/include)
FILE(GLOB_RECURSE TENNCOR_SRC
    ${TENNCOR_DIR}/src/*.cpp)

#-------------------
# set tenncor include path
#-------------------
include_directories(${TENNCOR_INCLUDE})

set(TENNCOR_LIB tenncor)
add_library(${TENNCOR_LIB} STATIC ${TENNCOR_SRC})
#>>>>>>>>>>>>>>>>>>>>>>>> DONE BUILDING TENNCOR <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BUILD ROCNNET DLL <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#-------------------
# set rocnnet paths
#-------------------
set(ROCNNET_DIR ${PROJECT_SOURCE_DIR}/rocnnet)
set(ROCNNET_INCLUDE ${ROCNNET_DIR}/include)
FILE(GLOB_RECURSE ROCNNET_SRC
    ${ROCNNET_DIR}/src/*.cpp)

#-------------------
# set tenncor include path
#-------------------
include_directories(${ROCNNET_INCLUDE})

set(ROCNNET_DLL rocnnet)
add_library(${ROCNNET_DLL} SHARED ${ROCNNET_SRC})
target_link_libraries (${ROCNNET_DLL} ${TENNCOR_LIB})
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DONE ROCNNET APP <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BUILD IO LIBRARY <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
set(IO_DIR ${PROJECT_SOURCE_DIR}/tenncor_io)
set(TENN_IO_INCLUDE ${IO_DIR}/include)
FILE(GLOB_RECURSE TENN_IO_SRC
        ${IO_DIR}/src/*.cpp)

include_directories(${TENN_IO_INCLUDE})

set(TENN_IO_DLL tenncor_io)
add_library(${TENN_IO_DLL} SHARED ${TENN_IO_SRC})
target_link_libraries (${TENN_IO_DLL} ${TENNCOR_LIB})
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DONE IO LIBRARY <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BUILD DEMO1 AGENT <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
set(DEMO_DIR ${PROJECT_SOURCE_DIR}/demo1)

include_directories(
    ${TENNCOR_INCLUDE}
    ${ROCNNET_INCLUDE})

add_executable (Demo ${DEMO_DIR}/agent.cpp)
target_link_libraries (Demo ${TENNCOR_LIB} ${ROCNNET_DLL})
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DONE DEMO1 AGENT <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>> UNIT TESTING USING GOOGLETEST <<<<<<<<<<<<<<<<<<<<<<<<<
set(UNIT_TEST_NAME unit_test)
set(UNIT_TEST_PATH ${PROJECT_SOURCE_DIR}/tests/unit_tests)
set(MOCK_INCLUDE ${UNIT_TEST_PATH}/mock)

#-------------------
# finalize googletest setup
#-------------------
add_definitions(${MSVC_COMPILER_DEFS})
enable_testing()

include_directories(${GTEST_INCLUDE_DIRS} ${TENNCOR_INCLUDE} ${MOCK_INCLUDE})

file(GLOB_RECURSE UNIT_TEST_SRC
    ${UNIT_TEST_PATH}/*.cpp)

add_executable(${UNIT_TEST_NAME} ${UNIT_TEST_SRC} ${UNIT_MAIN})

#-------------------
# Test dependencies
#-------------------
add_dependencies(${UNIT_TEST_NAME}
    googletest
    tenncor)

#-------------------
# link googletest depending on  available libraries
#-------------------
if(NOT WIN32 OR MINGW)
    target_link_libraries(${UNIT_TEST_NAME}
            ${GTEST_LIBS_DIR}/libgtest.a
            ${GTEST_LIBS_DIR}/libgtest_main.a
            )
else()
    target_link_libraries(${UNIT_TEST_NAME}
            debug ${GTEST_LIBS_DIR}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${CMAKE_FIND_LIBRARY_SUFFIXES}
            optimized ${GTEST_LIBS_DIR}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${CMAKE_FIND_LIBRARY_SUFFIXES}
            )
    target_link_libraries(${UNIT_TEST_NAME}
            debug ${GTEST_LIBS_DIR}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${CMAKE_FIND_LIBRARY_SUFFIXES}
            optimized ${GTEST_LIBS_DIR}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${CMAKE_FIND_LIBRARY_SUFFIXES}
            )
endif()

# Link
target_link_libraries(${UNIT_TEST_NAME}
    pthread
    gmock
    gtest
    ${CMAKE_THREAD_LIBS_INIT}
    ${TENNCOR_LIB})
add_test(unit ${UNIT_TEST_NAME})
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DONE UNIT TESTING <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>> PERFORMANCE TESTING USING GOOGLETEST <<<<<<<<<<<<<<<<<<<<<<<<<
set(PERFORM_TEST_NAME performance_test)
set(PERFORM_TEST_PATH ${PROJECT_SOURCE_DIR}/tests/performance_tests)

#-------------------
# finalize googletest setup
#-------------------
add_definitions(${MSVC_COMPILER_DEFS})
enable_testing()

include_directories(${GTEST_INCLUDE_DIRS} ${ROCNNET_INCLUDE})

file(GLOB_RECURSE PERFORM_TEST_SRC
        ${PERFORM_TEST_PATH}/*.cpp)

add_executable(${PERFORM_TEST_NAME} ${PERFORM_TEST_SRC} ${PERFORM_MAIN})

#-------------------
# Test dependencies
#-------------------
add_dependencies(${PERFORMANCE_TEST_NAME}
        googletest
        tenncor
        rocnnet)

#-------------------
# link googletest depending on available libraries
#-------------------
if(NOT WIN32 OR MINGW)
    target_link_libraries(${UNIT_TEST_NAME}
            ${GTEST_LIBS_DIR}/libgtest.a
            ${GTEST_LIBS_DIR}/libgtest_main.a
            )
else()
    target_link_libraries(${UNIT_TEST_NAME}
            debug ${GTEST_LIBS_DIR}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${CMAKE_FIND_LIBRARY_SUFFIXES}
            optimized ${GTEST_LIBS_DIR}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${CMAKE_FIND_LIBRARY_SUFFIXES}
            )
    target_link_libraries(${UNIT_TEST_NAME}
            debug ${GTEST_LIBS_DIR}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${CMAKE_FIND_LIBRARY_SUFFIXES}
            optimized ${GTEST_LIBS_DIR}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${CMAKE_FIND_LIBRARY_SUFFIXES}
            )
endif()

# Link
target_link_libraries(${PERFORM_TEST_NAME} 
    pthread
    gtest
    ${CMAKE_THREAD_LIBS_INIT}
    ${ROCNNET_DLL}
    ${TENNCOR_LIB})
add_test(perform ${PERFORM_TEST_NAME})
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DONE PERFORMANCE TESTING <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
