cmake_minimum_required(VERSION 3.14)

project(3D_Reconstruction LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt: moc/uic/rcc
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Jeśli QTDIR jest ustawione w środowisku, dodaj do prefix path (pomocne na Windows)
if(DEFINED ENV{QTDIR})
  list(APPEND CMAKE_PREFIX_PATH $ENV{QTDIR})
endif()

# Pakiety
find_package(Qt5 5.12 REQUIRED COMPONENTS Widgets)  # Widgets pociągnie Gui i Core
find_package(OpenCV REQUIRED)
find_package(PCL 1.3 REQUIRED COMPONENTS common io filters visualization)

# Opcjonalnie OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found")
endif()

# Ścieżki źródeł
set(FORMS_DIR   "${CMAKE_SOURCE_DIR}/forms")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(SOURCE_DIR  "${CMAKE_SOURCE_DIR}/src")

# AutoUIC będzie przeszukiwał forms/
set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_AUTOUIC_SEARCH_PATHS} ${FORMS_DIR})

# Zbiór źródeł (ok do tego projektu; świadomość, że GLOB nie wykrywa automatycznie nowych plików w niektórych IDE)
file(GLOB_RECURSE SOURCES
  "${FORMS_DIR}/*.ui"
  "${FORMS_DIR}/*.qrc"
  "${INCLUDE_DIR}/*.h"
  "${SOURCE_DIR}/*.cpp"
)

# Aplikacja GUI na Windows (WIN32).
add_executable(3D_Reconstruction WIN32 ${SOURCES})

# Include’y tylko dla targetu
target_include_directories(3D_Reconstruction PRIVATE
  ${INCLUDE_DIR}
  ${SOURCE_DIR}
  ${FORMS_DIR}
  ${PCL_INCLUDE_DIRS}
)

# Definicje z PCL (np. VTK, itp.)
add_definitions(${PCL_DEFINITIONS})

# Linkowanie — bez twardych ścieżek do .lib
target_link_libraries(3D_Reconstruction PRIVATE
  Qt5::Widgets
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

# OpenMP (jeśli jest)
if(OpenMP_CXX_FOUND)
  target_link_libraries(3D_Reconstruction PRIVATE OpenMP::OpenMP_CXX)
endif()

# Przy MSVC możesz chcieć większe obj-y (duże template’y PCL)
if(MSVC)
  target_compile_options(3D_Reconstruction PRIVATE /bigobj)
endif()
