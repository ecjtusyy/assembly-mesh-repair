#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

cmake -S cgal_bridge -B build/cgal
cmake --build build/cgal -j

python3 -m unittest tests.test_python_cleanup -v

mkdir -p tests/out/minimal_suite

./build/cgal/check_self_intersections tests/data/clean_tri.obj --list_pairs
./build/cgal/check_self_intersections tests/data/dup_vertex.obj --list_pairs
./build/cgal/check_self_intersections tests/data/tri_cross.obj --list_pairs
./build/cgal/check_self_intersections tests/data/shared_point_multi_intersection.obj --list_pairs
./build/cgal/check_self_intersections tests/data/mixed_case.obj --list_pairs

python3 pipeline.py \
  --input \
    tests/data/clean_tri.obj \
    tests/data/dup_vertex.obj \
    tests/data/tri_cross.obj \
    tests/data/shared_point_multi_intersection.obj \
    tests/data/mixed_case.obj \
  --output_dir tests/out/minimal_suite \
  --report_json tests/out/minimal_suite/report.json

./build/cgal/check_self_intersections tests/out/minimal_suite/clean_tri_repaired.obj --list_pairs
./build/cgal/check_self_intersections tests/out/minimal_suite/dup_vertex_repaired.obj --list_pairs
./build/cgal/check_self_intersections tests/out/minimal_suite/tri_cross_repaired.obj --list_pairs
./build/cgal/check_self_intersections tests/out/minimal_suite/shared_point_multi_intersection_repaired.obj --list_pairs
./build/cgal/check_self_intersections tests/out/minimal_suite/mixed_case_repaired.obj --list_pairs
