add_lab("VectorAdd")
add_lab_template("VectorAdd" ${CMAKE_CURRENT_LIST_DIR}/template.cu)
add_lab_solution("VectorAdd" ${CMAKE_CURRENT_LIST_DIR}/solution.cu)
add_generator("VectorAdd" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
