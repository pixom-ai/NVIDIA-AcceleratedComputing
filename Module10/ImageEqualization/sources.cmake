XXXDISABLEXadd_labXXX("ImageEqualization")
add_lab_template("ImageEqualization" ${CMAKE_CURRENT_LIST_DIR}/template.cu)
add_lab_solution("ImageEqualization" ${CMAKE_CURRENT_LIST_DIR}/solution.cu)
add_generator("ImageEqualization" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
