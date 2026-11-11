# CMake generated Testfile for 
# Source directory: /home/ska/Desktop/programming/code/tests/compilation
# Build directory: /home/ska/Desktop/programming/code/build-tests/tests/compilation
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_compilation "/usr/bin/cmake" "-DBUILD_DIR=/home/ska/Desktop/programming/code/build-tests" "-DTARGET=test_compilation" "-DCONFIG=" "-P" "/home/ska/Desktop/programming/code/tests/cmake/BuildAndRun.cmake")
set_tests_properties(test_compilation PROPERTIES  LABELS "interface" RESOURCE_LOCK "dnn_build" _BACKTRACE_TRIPLES "/home/ska/Desktop/programming/code/tests/compilation/CMakeLists.txt;38;add_test;/home/ska/Desktop/programming/code/tests/compilation/CMakeLists.txt;0;")
