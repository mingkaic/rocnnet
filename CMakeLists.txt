#>>>>>>>>>>>>>>>>>>>>>>>> GENERAL SETUP COMMENCE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
cmake_minimum_required(VERSION 3.6)
set(PROJECT_NAME_STR rocnnet)
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
# set common include folder for module
#-------------------
set(EXT_PROJECTS_DIR ${PROJECT_SOURCE_DIR}/ext)

#-------------------
# setup googletest in external directory
#-------------------
add_subdirectory(${EXT_PROJECTS_DIR}/gtest)
#>>>>>>>>>>>>>>>>>>>>>>>> GENERAL SETUP COMPLETE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BUILD DEMO1 AGENT <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
set(DEMO_DIR ${PROJECT_SOURCE_DIR}/demo1)

include_directories(${TENNCOR_INCLUDE})
include_directories(${ROCNNET_INCLUDE})

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

add_executable(${UNIT_TEST_NAME} ${UNIT_TEST_SRC})

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

add_executable(${PERFORM_TEST_NAME} ${PERFORM_TEST_SRC})

#-------------------
# Test dependencies
#-------------------
add_dependencies(${PERFORMANCE_TEST_NAME}
		googletest
		tenncor
		rocnnet)

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
target_link_libraries(${PERFORMANCE_TEST_NAME}
		${CMAKE_THREAD_LIBS_INIT}
        ${ROCNNET_DLL})
add_test(perform ${PERFORM_TEST_NAME})
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DONE UNIT TESTING <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<