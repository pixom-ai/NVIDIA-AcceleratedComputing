add_lab("Stencil")
add_lab_template("Stencil" ${CMAKE_CURRENT_LIST_DIR}/template.cu)
add_lab_solution("Stencil" ${CMAKE_CURRENT_LIST_DIR}/solution.cu)
add_generator("Stencil" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
