add_lab("ImageBlur")
add_lab_template("ImageBlur" ${CMAKE_CURRENT_LIST_DIR}/template.cu)
add_lab_solution("ImageBlur" ${CMAKE_CURRENT_LIST_DIR}/solution.cu)
add_generator("ImageBlur" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
