add_lab("VectorAdd_Stream")
add_lab_template("VectorAdd_Stream" ${CMAKE_CURRENT_LIST_DIR}/template.cu)
add_lab_solution("VectorAdd_Stream" ${CMAKE_CURRENT_LIST_DIR}/solution.cu)
add_generator("VectorAdd_Stream" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
