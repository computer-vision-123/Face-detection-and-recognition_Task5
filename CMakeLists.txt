cmake_minimum_required(VERSION 3.15)
project(FaceVision VERSION 1.1 LANGUAGES CXX)
cmake_policy(SET CMP0148 NEW)

# ── C++ standard ────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Dependency paths  (override via -D flags or environment) ────────
set(OpenCV_DIR   "C:/opencv/build"
    CACHE PATH "Path to OpenCV build directory")
set(pybind11_DIR "C:/Users/Sarah/AppData/Local/Programs/Python/Python312/Lib/site-packages/pybind11/share/cmake/pybind11"
    CACHE PATH "Path to pybind11 cmake directory")

# ── pybind11 / Python ────────────────────────────────────────────────
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

# ── OpenCV ───────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs objdetect)

# ── cv_backend Python extension ─────────────────────────────────────
pybind11_add_module(cv_backend
    ${CMAKE_SOURCE_DIR}/backend/src/cv_backend_module.cpp   # pybind11 bindings
    ${CMAKE_SOURCE_DIR}/backend/src/pca_engine.cpp          # PCA / Eigenfaces
    ${CMAKE_SOURCE_DIR}/backend/src/face_detector.cpp       # Haar cascade detection
    ${CMAKE_SOURCE_DIR}/backend/src/image_utils.cpp         # I/O and channel helpers
    ${CMAKE_SOURCE_DIR}/backend/src/roc_plotter.cpp         # ROC curve computation
)

target_include_directories(cv_backend PRIVATE
    ${CMAKE_SOURCE_DIR}/backend/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(cv_backend PRIVATE
    ${OpenCV_LIBS}
)

# ── Output: place .pyd/.so next to frontend/main.py ─────────────────
foreach(CONFIG "" _RELEASE _DEBUG _RELWITHDEBINFO _MINSIZEREL)
    set_target_properties(cv_backend PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY${CONFIG} "${CMAKE_SOURCE_DIR}/frontend"
        RUNTIME_OUTPUT_DIRECTORY${CONFIG} "${CMAKE_SOURCE_DIR}/frontend"
    )
endforeach()

install(TARGETS cv_backend
    LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/frontend"
    RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/frontend"
)

# ── Haar cascade XML ─────────────────────────────────────────────────
# Source: committed to repo under data/
set(CASCADE_SRC
    "${CMAKE_SOURCE_DIR}/data/haarcascade_frontalface_default.xml")
set(CASCADE_DST_DIR "${CMAKE_SOURCE_DIR}/frontend/data")

if(NOT EXISTS "${CASCADE_SRC}")
    message(WARNING
        "Haar cascade not found at ${CASCADE_SRC}. "
        "Copy haarcascade_frontalface_default.xml from your OpenCV "
        "installation into <project_root>/data/ before building.")
endif()

# Copy next to the extension after every successful build
add_custom_command(TARGET cv_backend POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CASCADE_DST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CASCADE_SRC}"
        "${CASCADE_DST_DIR}/haarcascade_frontalface_default.xml"
    COMMENT "Copying Haar cascade XML → frontend/data/"
)

# Also honour `cmake --install`
install(FILES "${CASCADE_SRC}"
    DESTINATION "${CMAKE_SOURCE_DIR}/frontend/data"
)

# ── Status messages ──────────────────────────────────────────────────
message(STATUS "---------------------------------------------------")
message(STATUS "  Project      : ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "  C++ standard : ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV       : ${OpenCV_VERSION}")
message(STATUS "  Python       : ${Python_VERSION}")
message(STATUS "  Output dir   : ${CMAKE_SOURCE_DIR}/frontend")
message(STATUS "  Cascade XML  : ${CASCADE_SRC}")
message(STATUS "---------------------------------------------------")