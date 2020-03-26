
add_lab("OpenACCVectorAdd")
add_lab_template("OpenACCVectorAdd" ${CMAKE_CURRENT_LIST_DIR}/template.cpp)
add_lab_solution("OpenACCVectorAdd" ${CMAKE_CURRENT_LIST_DIR}/solution.cpp)
add_generator("OpenACCVectorAdd" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
