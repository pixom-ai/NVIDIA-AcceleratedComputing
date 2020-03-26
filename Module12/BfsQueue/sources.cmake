add_lab("BfsQueue")
add_lab_template("BfsfQueue" ${CMAKE_CURRENT_LIST_DIR}/template.cu)
add_lab_solution("BfsQueue" ${CMAKE_CURRENT_LIST_DIR}/solution.cu)
add_generator("BfsQueue" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
