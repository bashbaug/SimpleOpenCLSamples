set(SAMPLE_NAME enumopencl)
set(SAMPLE_NUMBER 00)

project(${SAMPLE_NAME})
include_directories( ${OpenCL_INCLUDE_DIR} )
add_executable( ${SAMPLE_NAME} main.cpp )
target_link_libraries( ${SAMPLE_NAME} OpenCL )
set_target_properties( ${SAMPLE_NAME} PROPERTIES FOLDER "Samples/${SAMPLE_NUMBER}_${SAMPLE_NAME}" )

install(TARGETS ${SAMPLE_NAME} CONFIGURATIONS Debug   DESTINATION ${SAMPLES_PLATFORM_NAME}/Debug   )
install(TARGETS ${SAMPLE_NAME} CONFIGURATIONS Release DESTINATION ${SAMPLES_PLATFORM_NAME}/Release )
