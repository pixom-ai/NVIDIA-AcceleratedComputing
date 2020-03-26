add_lab("Reduction")
add_lab_template("Reduction" ${CMAKE_CURRENT_LIST_DIR}/template.cu)
add_lab_solution("Reduction" ${CMAKE_CURRENT_LIST_DIR}/solution.cu)
add_generator("Reduction" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)

