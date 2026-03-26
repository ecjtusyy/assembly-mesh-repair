# Actual Test Records

Environment: Debian 13 (trixie), GCC 14.2.0, CMake 3.31.6, vendored CGAL 6.1.1 plus system GMP/MPFR/Boost.

## Dependency installation
```bash
apt-get update
apt-get install -y build-essential cmake libcgal-dev libmpfr-dev libgmp-dev libboost-program-options-dev libboost-system-dev libboost-thread-dev zlib1g-dev python3 python3-pip
```

Actual note: `apt-get` was pointed at the internal Artifactory Debian mirrors because direct DNS resolution to `deb.debian.org` was unavailable in this container.

## Build
```bash
cmake -S cgal_bridge -B build/cgal
cmake --build build/cgal -j
```

Result: both `build/cgal/check_self_intersections` and `build/cgal/autorefine_obj` were produced successfully.

## clean_tri.obj

```bash
./build/cgal/check_self_intersections tests/data/clean_tri.obj --list_pairs
python3 pipeline.py --input tests/data/clean_tri.obj --output_dir tests/out/minimal_suite
./build/cgal/check_self_intersections tests/out/minimal_suite/clean_tri_repaired.obj --list_pairs
```

- Raw input checker actual: self_intersect=0, count=0
- Pipeline pre-check after Python cleanup: self_intersect=0, count=0
- Output checker actual: self_intersect=0, count=0
- Vertex count: raw input 3 -> after pre-cleanup 3 -> final 3
- Face count: raw input 1 -> after pre-cleanup 1 -> final 1
- Pre-cleanup merged_vertices=0, duplicate_removed=0, degenerate_removed=0
- Autorefined: False
- Output degenerate faces: False
- Output duplicate faces: False

## dup_vertex.obj

```bash
./build/cgal/check_self_intersections tests/data/dup_vertex.obj --list_pairs
python3 pipeline.py --input tests/data/dup_vertex.obj --output_dir tests/out/minimal_suite
./build/cgal/check_self_intersections tests/out/minimal_suite/dup_vertex_repaired.obj --list_pairs
```

- Raw input checker actual: self_intersect=1, count=1
- Pipeline pre-check after Python cleanup: self_intersect=0, count=0
- Output checker actual: self_intersect=0, count=0
- Vertex count: raw input 4 -> after pre-cleanup 3 -> final 3
- Face count: raw input 2 -> after pre-cleanup 1 -> final 1
- Pre-cleanup merged_vertices=1, duplicate_removed=1, degenerate_removed=0
- Autorefined: False
- Output degenerate faces: False
- Output duplicate faces: False

## tri_cross.obj

```bash
./build/cgal/check_self_intersections tests/data/tri_cross.obj --list_pairs
python3 pipeline.py --input tests/data/tri_cross.obj --output_dir tests/out/minimal_suite
./build/cgal/check_self_intersections tests/out/minimal_suite/tri_cross_repaired.obj --list_pairs
```

- Raw input checker actual: self_intersect=1, count=1
- Pipeline pre-check after Python cleanup: self_intersect=1, count=1
- Output checker actual: self_intersect=0, count=0
- Vertex count: raw input 6 -> after pre-cleanup 6 -> final 7
- Face count: raw input 2 -> after pre-cleanup 2 -> final 7
- Pre-cleanup merged_vertices=0, duplicate_removed=0, degenerate_removed=0
- Autorefined: True
- Output degenerate faces: False
- Output duplicate faces: False

## shared_point_multi_intersection.obj

```bash
./build/cgal/check_self_intersections tests/data/shared_point_multi_intersection.obj --list_pairs
python3 pipeline.py --input tests/data/shared_point_multi_intersection.obj --output_dir tests/out/minimal_suite
./build/cgal/check_self_intersections tests/out/minimal_suite/shared_point_multi_intersection_repaired.obj --list_pairs
```

- Raw input checker actual: self_intersect=1, count=3
- Pipeline pre-check after Python cleanup: self_intersect=1, count=3
- Output checker actual: self_intersect=0, count=0
- Vertex count: raw input 9 -> after pre-cleanup 9 -> final 16
- Face count: raw input 3 -> after pre-cleanup 3 -> final 24
- Pre-cleanup merged_vertices=0, duplicate_removed=0, degenerate_removed=0
- Autorefined: True
- Output degenerate faces: False
- Output duplicate faces: False

## mixed_case.obj

```bash
./build/cgal/check_self_intersections tests/data/mixed_case.obj --list_pairs
python3 pipeline.py --input tests/data/mixed_case.obj --output_dir tests/out/minimal_suite
./build/cgal/check_self_intersections tests/out/minimal_suite/mixed_case_repaired.obj --list_pairs
```

- Raw input checker actual: self_intersect=1, count=3
- Pipeline pre-check after Python cleanup: self_intersect=1, count=1
- Output checker actual: self_intersect=0, count=0
- Vertex count: raw input 7 -> after pre-cleanup 6 -> final 7
- Face count: raw input 3 -> after pre-cleanup 2 -> final 7
- Pre-cleanup merged_vertices=1, duplicate_removed=1, degenerate_removed=0
- Autorefined: True
- Output degenerate faces: False
- Output duplicate faces: False
