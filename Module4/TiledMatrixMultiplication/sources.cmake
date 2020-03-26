
add_lab("TiledMatrixMultiplication")
add_lab_template("TiledMatrixMultiplication" ${CMAKE_CURRENT_LIST_DIR}/template.cu)
add_lab_solution("TiledMatrixMultiplication" ${CMAKE_CURRENT_LIST_DIR}/solution.cu)
add_generator("TiledMatrixMultiplication" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
