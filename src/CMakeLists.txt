
find_package( OpenCV REQUIRED )

add_subdirectory(evaluate)

include_directories(${OpenCV_INCLUDE_DIRS} )
include_directories(evaluate)


add_library(feature SHARED "feature.cpp")
add_library(bucket SHARED "bucket.cpp")
add_library(utils SHARED "utils.cpp")
add_library(visualOdometry SHARED "visualOdometry.cpp")

find_package(Sophus REQUIRED)
find_package( G2O REQUIRED )
find_package(Pangolin REQUIRED)
include_directories( ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS})


add_library(Converter SHARED "Converter.cpp")
target_link_libraries(Converter ${OpenCV_LIBS} g2o_core g2o_stuff g2o_types_sba)

add_library(MapDrawer SHARED "MapDrawer.cpp")
target_link_libraries(MapDrawer ${Sophus_LIBRARIES}
        ${Pangolin_LIBRARIES} ${OpenCV_LIBS})
add_library(Frame SHARED "Frame.cpp")


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
message(${PROJECT_SOURCE_DIR})

add_executable( run main.cpp )
target_link_libraries( bucket ${OpenCV_LIBS} )
target_link_libraries( feature ${OpenCV_LIBS} bucket)
target_link_libraries( utils evaluate_odometry feature ${OpenCV_LIBS} )
target_link_libraries( visualOdometry utils bucket feature ${OpenCV_LIBS})


target_link_libraries( Frame ${OpenCV_LIBS} )
target_link_libraries( run ${OpenCV_LIBS} bucket feature utils visualOdometry Frame Converter MapDrawer)
