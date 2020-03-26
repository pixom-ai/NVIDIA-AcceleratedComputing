add_lab("Convolution")
add_lab_template("Convolution" ${CMAKE_CURRENT_LIST_DIR}/template.cu)
add_lab_solution("Convolution" ${CMAKE_CURRENT_LIST_DIR}/solution.cu)
add_generator("Convolution" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
