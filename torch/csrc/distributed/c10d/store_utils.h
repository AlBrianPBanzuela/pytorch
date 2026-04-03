#pragma once

#include <c10/macros/Export.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/python_headers.h>

// Extract c10::intrusive_ptr<c10d::Store> from a Python Store object.
// Defined in init.cpp (libtorch_python.so). For use by third-party backend
// extensions that need native Store access across pybind11 module boundaries.
//
// On success, assigns a new strong reference to *out and returns 0.
// On failure, sets a Python error (if applicable), leaves *out unchanged, and
// returns -1. The out-param signature keeps the ABI trivially C-compatible
// (only pointers cross the boundary) while preserving stable symbol linkage
// across shared libraries.
extern "C" C10_EXPORT int THPStore_Unwrap(
    PyObject* obj,
    c10::intrusive_ptr<c10d::Store>* out);
