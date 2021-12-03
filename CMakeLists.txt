cmake_minimum_required(VERSION 2.8)

project( HelloWorld_Proj )

find_package ( OpenCV REQUIRED )

add_executable ( HelloWorld_Exec src/main.cpp )
target_link_libraries ( HelloWorld_Exec ${OpenCV_LIBS} )