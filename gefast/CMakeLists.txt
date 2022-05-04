set(LIBRARY_HEADER_FILES
        math/cayley.h
        math/rotation.h
        types.h
        solver/generalized_eigensolver.h
        )

set(LIBRARY_SOURCE_FILES
        math/cayley.cpp
        math/rotation.cpp
        solver/generalized_eigensolver.cpp
        )

include(${PROJECT_SOURCE_DIR}/cmake/LibraryConfig.cmake)
target_link_libraries(${LIBRARY_NAME} Eigen3::Eigen)