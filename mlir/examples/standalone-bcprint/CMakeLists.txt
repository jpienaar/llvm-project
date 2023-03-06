# Instance of using parser
add_executable(MlirBytecodeStandalonePrintParse PrintParse.c)
if(NOT MSVC)
	target_link_libraries(MlirBytecodeStandalonePrintParse PUBLIC m)
endif()
