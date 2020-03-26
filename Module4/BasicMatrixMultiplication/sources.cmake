
add_lab("BasicMatrixMultiplication")
add_lab_template("BasicMatrixMultiplication" ${CMAKE_CURRENT_LIST_DIR}/template.cu)
add_lab_solution("BasicMatrixMultiplication" ${CMAKE_CURRENT_LIST_DIR}/solution.cu)
add_generator("BasicMatrixMultiplication" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
