cmake_minimum_required(VERSION 3.14)

project(3D_Reconstruction LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt: moc/uic/rcc
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# If QTDIR is set in env
if(DEFINED ENV{QTDIR})
  list(APPEND CMAKE_PREFIX_PATH $ENV{QTDIR})
endif()

# Packages
find_package(Qt5 5.12 REQUIRED COMPONENTS Widgets)  # Widgets pociągnie Gui i Core
find_package(OpenCV REQUIRED)
find_package(PCL 1.3 REQUIRED COMPONENTS common io filters visualization)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found")
endif()

# Source paths
set(FORMS_DIR   "${CMAKE_SOURCE_DIR}/forms")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(SOURCE_DIR  "${CMAKE_SOURCE_DIR}/src")

set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_AUTOUIC_SEARCH_PATHS} ${FORMS_DIR})

file(GLOB_RECURSE SOURCES
  "${FORMS_DIR}/*.ui"
  "${FORMS_DIR}/*.qrc"
  "${INCLUDE_DIR}/*.h"
  "${SOURCE_DIR}/*.cpp"
)

# GUI for Windows
add_executable(3D_Reconstruction WIN32 ${SOURCES})

# Include’y tylko dla targetu
target_include_directories(3D_Reconstruction PRIVATE
  ${INCLUDE_DIR}
  ${SOURCE_DIR}
  ${FORMS_DIR}
  ${PCL_INCLUDE_DIRS}
)

# PCL
add_definitions(${PCL_DEFINITIONS})

# Linkowanie — bez twardych ścieżek do .lib
target_link_libraries(3D_Reconstruction PRIVATE
  Qt5::Widgets
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

# OpenMP
if(OpenMP_CXX_FOUND)
  target_link_libraries(3D_Reconstruction PRIVATE OpenMP::OpenMP_CXX)
endif()
