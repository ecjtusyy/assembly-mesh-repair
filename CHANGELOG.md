# Changelog

## Python + CGAL bridge prototype

- replaced the old main pipeline with:
  - Python vertex welding / cleanup
  - CGAL triangle soup checker
  - CGAL `autorefine_triangle_soup()`
  - Python cleanup
  - CGAL post-check
- added `cgal_bridge/` with CMake build files and C++ bridge executables
- rewrote `ops/stitch.py` to do deterministic vertex welding, degenerate removal,
  duplicate-face removal, and vertex compaction
- added `ops/cgal_refine.py` for subprocess execution with timeout, stdout/stderr capture,
  and structured parsing
- rewrote `ops/pipeline_impl.py` to remove the old approximate narrow-phase from the main path
- disabled geometry-moving smoothing in the production path
- added minimal regression assets and Python cleanup tests
- wired CMake to prefer a vendored official CGAL 6.1.1 release under `third_party/CGAL-6.1.1`
- validated CGAL bridge build end-to-end against CGAL 6.1.1 in this environment
- patched `autorefine_obj.cpp` to pass integer snap-rounding named parameters, avoiding a GCC 14 template deduction failure in CGAL 6.1.1 headers
- expanded `tests/run_minimal_regression.sh` to cover the full five-case acceptance sweep (`clean_tri`, `dup_vertex`, `tri_cross`, `shared_point_multi_intersection`, `mixed_case`)
