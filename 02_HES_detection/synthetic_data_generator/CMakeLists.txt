cmake_minimum_required(VERSION 3.16)
project(SyntheticDataGenerator LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Toggle to allow configuring without OpenCV
option(ENABLE_OPENCV "Build with OpenCV" ON)

# Headers first (so they show up in VS solution)
set(SDG_HEADERS
    include/ConfigLoader.h
    include/DatasetGenerator.h
    include/makeJson.h
)

# Sources
set(SDG_SOURCES
    src/main.cpp
    src/ConfigLoader.cpp
    src/DatasetGenerator.cpp
    src/makeJson.cpp
)

# OpenCV (optional)
if(ENABLE_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc highgui calib3d)
  if(NOT OpenCV_FOUND)
    message(WARNING "OpenCV not found. Set -DOpenCV_DIR=... or configure with -DENABLE_OPENCV=OFF to generate a stub target.")
    set(ENABLE_OPENCV OFF)
  endif()
endif()

if(ENABLE_OPENCV)
  add_executable(${PROJECT_NAME} ${SDG_SOURCES} ${SDG_HEADERS})

  target_include_directories(${PROJECT_NAME}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party
      ${OpenCV_INCLUDE_DIRS}
  )

  target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

  # Copy config.json next to the binary
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/config.json
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/config.json)
else()
  # Stub so configuration still succeeds without OpenCV
  add_custom_target(${PROJECT_NAME}
    ALL
    COMMENT "OpenCV disabled or not found: no build will be produced.")
endif()

# Visual Studio filters
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src     PREFIX "Source Files"  FILES ${SDG_SOURCES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/include PREFIX "Header Files"  FILES ${SDG_HEADERS})
