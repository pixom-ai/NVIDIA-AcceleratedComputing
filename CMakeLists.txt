cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH TRUE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


project(teachingkit-labs)

set(BUILD_LIBWB_LIBRARY ON CACHE BOOL "Turn on build of libwb library")
set(BUILD_SOLUTION ON CACHE BOOL "Turn on build of code solution")
# set(BUILD_DESCRIPTION OFF CACHE BOOL "Turn on build of lab description")
set(BUILD_DATASET_GENERATOR ON CACHE BOOL "Turn on building of dataset generator")
set(BUILD_TEMPLATE OFF CACHE BOOL "Turn on build of code templates")

list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/support/cmake/modules)

if (BUILD_SOLUTION OR BUILD_TEMPLATE OR BUILD_LIBWB_LIBRARY)
  find_package(CUDA REQUIRED)
else()
  find_package(CUDA)
endif ()

if (NOT WIN32)
    find_package(EnableCxx11 REQUIRED)
endif (NOT WIN32)

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CUDA_VERBOSE_BUILD OFF)

# allow C99 for cmake versions less than 3.1
set (CMAKE_C_STANDARD 99)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9.3")
    if(CUDA_VERSION VERSION_LESS "8.0")
      message(STATUS "Found gcc >=5 and CUDA <= 7.5, adding workaround C++ flags")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
    endif(CUDA_VERSION VERSION_LESS "8.0")
  endif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9.3")
endif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")


set(CUDA_PROPAGATE_HOST_FLAGS TRUE)

# For some MPs it may be necessary to specify the compute capability of your
# NVIDIA GPU. In that case, simply uncomment the following two lines that start
# with 'set' and change the value of COMPUTE_CAPABILITY to one of the following
# hardware architectures: Tesla-class '11', '12', or '13'; Fermi-class '20'
# or '21'; Kepler-class '30', '32', '35'; or Maxwell-class '50' or '52'
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.0 3.5+PTX 5.0 Maxwell 6.1)
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

# Set warning levels for host compilation
if (UNIX)
    add_compile_options(-Wno-unused-function)
    add_compile_options(-Wno-unused-parameter)
    add_compile_options(-Wextra)
    add_compile_options(-Wno-unused-variable)
endif ()


add_definitions(-DWB_USE_CUDA)

# Select the correct host compiler on OS X
if (APPLE)
    if (${CMAKE_SYSTEM_VERSION} LESS 12.0)
        # OS X 10.7 or less
        find_program(GCC_EXECUTABLE NAMES g++ PATHS ENV PATH)
        if (NOT GCC_EXECUTABLE)
            message(FATAL_ERROR "Could not find host compiler 'g++'")
        endif(NOT GCC_EXECUTABLE)
        set(CUDA_HOST_COMPILER ${GCC_EXECUTABLE})
    else (${CMAKE_SYSTEM_VERSION} LESS 12.0)
        # OS X 10.8+
        find_program(CLANG_EXECUTABLE NAMES clang++-3.6 clang++ PATHS ENV PATH)
        if (NOT CLANG_EXECUTABLE)
            message(FATAL_ERROR "Could not find host compiler 'clang++'")
        endif ()
        set(CUDA_HOST_COMPILER ${CLANG_EXECUTABLE})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CUDA_INCLUDE_DIRS}")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++" )
    endif (${CMAKE_SYSTEM_VERSION} LESS 12.0)
endif (APPLE)

# Set the correct linker options on UNIX platforms
if (UNIX)
    set(LINK_LIBRARIES "-lm -lcuda")
    if (APPLE)
        find_library(CUDA_LIBRARY NAMES CUDA)
        get_filename_component(FRAMEWORK_DIRECTORY ${CUDA_LIBRARY} DIRECTORY)
        set(LINK_LIBRARIES "${LINK_LIBRARIES} -F${FRAMEWORK_DIRECTORY} -framework CUDA")
    else (APPLE)
        include(CheckLibraryExists)
        check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME)
        if (HAVE_CLOCK_GETTIME)
            set(LINK_LIBRARIES "${LINK_LIBRARIES} -lrt")
        else (HAVE_CLOCK_GETTIME)
            message(FATAL_ERROR "clock_gettime() not found")
        endif (HAVE_CLOCK_GETTIME)
    endif (APPLE)
endif (UNIX)

macro(add_lab_helper NAME SOURCES)
	cuda_add_executable(${NAME} ${SOURCES})
	target_link_libraries(${NAME} ${WBLIB} ${LINK_LIBRARIES})
    target_link_libraries(${NAME} ${CUDA_cudart_static_LIBRARY})
    target_link_libraries(${NAME} ${CUDA_LIBRARIES})
endmacro(add_lab_helper)

macro(add_lab_solution NAME SOURCES)
	if (BUILD_SOLUTION)
		add_lab_helper("${NAME}_Solution" ${SOURCES})
	endif(BUILD_SOLUTION)
endmacro(add_lab_solution)


macro(add_lab_template NAME SOURCES)
    # template code is not meant to be compilable,
    # so we do not copile it. Enable the BUILD_TEMPLATE
    # flag during cmake generation to override this
    # behavior
	if (BUILD_TEMPLATE)
		add_lab_helper("${NAME}_Template" ${SOURCES})
	endif(BUILD_TEMPLATE)
endmacro(add_lab_template)

macro(add_generator NAME SOURCES)
  if (BUILD_DATASET_GENERATOR)
	cuda_add_executable("${NAME}_DatasetGenerator" ${SOURCES})
    target_link_libraries("${NAME}_DatasetGenerator" ${WBLIB} ${LINK_LIBRARIES})
    target_link_libraries("${NAME}_DatasetGenerator" ${CUDA_cudart_static_LIBRARY})
    target_link_libraries("${NAME}_DatasetGenerator" ${CUDA_LIBRARIES})
  endif (BUILD_DATASET_GENERATOR)
endmacro(add_generator)

macro(add_lab NAME)
	if (BUILD_DESCRIPTION)
		find_program(PANDOC pandoc)
		set(MAKE_LAB_HANDOUT_PY "${PROJECT_SOURCE_DIR}/support/make_lab_handout.py")
		add_custom_target(
			"Build${NAME}Description"
			COMMAND python ${MAKE_LAB_HANDOUT_PY}
							${CMAKE_CURRENT_LIST_DIR}
							${PROJECT_SOURCE_DIR}/support
							${CMAKE_CURRENT_BINARY_DIR}
		)
		add_custom_target("${NAME}Description" ALL echo DEPENDS "Build${NAME}Description")
		set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${NAME}.pdf)
		set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${NAME}.docx)
	endif (BUILD_DESCRIPTION)
endmacro(add_lab)

include_directories(${PROJECT_SOURCE_DIR}/libwb)
CUDA_INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/libwb)

if (BUILD_LIBWB_LIBRARY)
    include(libwb/sources.cmake)
    cuda_add_library(${WBLIB} ${LIBWB_SOURCE_FILES} STATIC)
    if (WIN32)
        cuda_find_library_local_first(CUDA_cuda_LIBRARY cuda "\"cuda\" library")
        target_link_libraries(${WBLIB} ${CUDA_cuda_LIBRARY})
    endif (WIN32)
    target_link_libraries(${WBLIB} ${LINK_LIBRARIES})
    target_link_libraries(${WBLIB} ${CUDA_cudart_static_LIBRARY})
    target_link_libraries(${WBLIB} ${CUDA_LIBRARIES})
endif (BUILD_LIBWB_LIBRARY)


include(Module2/sources.cmake)
include(Module3/sources.cmake)
include(Module4/sources.cmake)
include(Module7/sources.cmake)
include(Module8/sources.cmake)
include(Module9/sources.cmake)
include(Module10/sources.cmake)
# include(Module11/sources.cmake)
include(Module12/sources.cmake)
include(Module14/sources.cmake)

# include(Module24/sources.cmake)

# These are the OpenCL and OpenACC
# labs and are not supported by the
# current cmake configuration
# include(Module21/sources.cmake)
# include(Module22/sources.cmake)

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()
