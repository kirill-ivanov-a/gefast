# Target
add_library(${LIBRARY_NAME}
        ${LIBRARY_SOURCE_FILES}
        ${LIBRARY_HEADER_FILES}
        )

# Alias:
add_library(${PROJECT_NAME}::${LIBRARY_NAME} ALIAS ${LIBRARY_NAME})

# C++17
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_17)

# Add definitions for targets
# Values:
#   - Debug  : -DGEFAST_DEBUG=1
#   - Release: -DGEFAST_DEBUG=0
#   - others : -DGEFAST_DEBUG=0
target_compile_definitions(${LIBRARY_NAME} PUBLIC
        "${PROJECT_NAME_UPPERCASE}_DEBUG=$<CONFIG:Debug>")

target_include_directories(
        ${LIBRARY_NAME} PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${GENERATED_HEADERS_DIR}>"
        "$<INSTALL_INTERFACE:.>"
)

# Targets:
install(
        TARGETS "${LIBRARY_NAME}"
        EXPORT "${TARGETS_EXPORT_NAME}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# Headers:
foreach (file ${LIBRARY_HEADER_FILES})
    get_filename_component(dir ${file} DIRECTORY)
    install(FILES ${file} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${LIBRARY_FOLDER}/${dir})
endforeach ()


# Headers:
install(
        FILES "${GENERATED_HEADERS_DIR}/${LIBRARY_FOLDER}/version.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${LIBRARY_FOLDER}"
)

# Config
install(
        FILES "${PROJECT_CONFIG_FILE}"
        "${VERSION_CONFIG_FILE}"
        DESTINATION "${CONFIG_INSTALL_DIR}"
)

# Config
install(
        EXPORT "${TARGETS_EXPORT_NAME}"
        FILE "${PROJECT_NAME}Targets.cmake"
        DESTINATION "${CONFIG_INSTALL_DIR}"
        NAMESPACE "${PROJECT_NAME}::"
)
