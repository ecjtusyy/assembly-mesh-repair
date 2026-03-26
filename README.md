# assembly-mesh-repair (Python + CGAL bridge prototype)

This branch migrates the project away from the original teaching-style approximate
narrow-phase (`tri_tri_intersection` / Python edge splitting / T-junction propagation /
Laplacian smoothing) and onto the new main path:

```text
OBJ -> Python vertex welding / cleanup -> CGAL checker
    -> CGAL autorefine_triangle_soup (if needed)
    -> Python cleanup -> CGAL checker -> OBJ
```

## What this prototype is for

Two problem classes are in scope:

1. **Topological errors caused by duplicate / near-duplicate vertices**
   - weld vertices within `eps_v`
   - remap all face indices
   - delete degenerate triangles
   - delete duplicate triangles
   - delete isolated vertices and compact indices

2. **Geometric self-intersections in triangle soups**
   - detect with CGAL triangle soup self-intersection APIs
   - repair with `CGAL::Polygon_mesh_processing::autorefine_triangle_soup()`
   - certify the output again with the CGAL checker

## First-version scope and limitations

- Input: OBJ
- Output: OBJ
- Guaranteed records: `v` and `f`
- `vt`, `vn`, `g`, `o`, `usemtl`, `mtllib`, smoothing groups and custom attributes are
  ignored in the first version
- Polygon faces are triangulated by fan triangulation during OBJ import
- The target guarantee is **triangle-soup self-intersection free according to the CGAL
  checker**, not full CAD semantic healing
- Multiple `--input` files are processed independently in the current prototype; cross-file
  repair is not attempted yet
- Geometry-moving post-processes such as Laplacian smoothing are intentionally disabled

## Repository layout

```text
cgal_bridge/
  CMakeLists.txt
  obj_triangle_soup_io.h
  check_self_intersections.cpp
  autorefine_obj.cpp
mesh/
  io_obj.py
  mesh.py
ops/
  stitch.py          # welding / cleanup / duplicate + degenerate removal
  cgal_refine.py     # subprocess bridge
  pipeline_impl.py   # new main pipeline
geom/
  intersection.py    # deprecated teaching stub
  retriangle.py      # deprecated teaching stub
tests/
  data/*.obj
  test_python_cleanup.py
  run_minimal_regression.sh
```

## Build the CGAL bridge

This repository now vendors the official CGAL 6.1.1 library release under `third_party/CGAL-6.1.1`, and the bridge CMakeLists tries that copy first. On Debian/Ubuntu-like systems you still need the numeric/toolchain dependencies (GMP/MPFR/Boost/CMake/compiler); if `third_party/CGAL-6.1.1` is removed, a system CGAL installation can also satisfy `find_package(CGAL)`.

Recommended dependency install on Debian/Ubuntu-like systems:

```bash
apt-get update
apt-get install -y \
  build-essential \
  cmake \
  libmpfr-dev \
  libgmp-dev \
  libboost-program-options-dev \
  libboost-system-dev \
  libboost-thread-dev \
  zlib1g-dev \
  python3 \
  python3-pip
```

Then build the bridge:

```bash
cmake -S cgal_bridge -B build/cgal
cmake --build build/cgal -j
```

This produces two executables:

- `build/cgal/check_self_intersections`
- `build/cgal/autorefine_obj`

## Checker usage

```bash
./build/cgal/check_self_intersections tests/data/tri_cross.obj --list_pairs
```

Expected output format:

```text
self_intersect=1
count=1
pair=0,1
```

## Autorefine usage

```bash
./build/cgal/autorefine_obj tests/data/tri_cross.obj tests/out/tri_cross_refined.obj
./build/cgal/check_self_intersections tests/out/tri_cross_refined.obj --list_pairs
```

`autorefine_obj` always enables `apply_iterative_snap_rounding(true)`.

## Python CLI usage

```bash
python pipeline.py \
  --input tests/data/tri_cross.obj \
  --output_dir tests/out/tri_cross \
  --eps_v 1e-9 \
  --eps_mode relative_bbox \
  --build_dir build/cgal
```

Outputs:

- repaired OBJ: `tests/out/tri_cross/tri_cross_repaired.obj`
- intermediate work files: `tests/out/tri_cross/tri_cross_work/`

## Minimal regression set

The minimal regression assets requested for this prototype are included:

- `tests/data/clean_tri.obj`
- `tests/data/dup_vertex.obj`
- `tests/data/tri_cross.obj`
- `tests/data/shared_point_multi_intersection.obj`
- `tests/data/mixed_case.obj`

Python-side cleanup tests:

```bash
python -m unittest tests.test_python_cleanup
```

Full bridge regression (builds the bridge, runs Python cleanup tests, then runs the full five-case acceptance sweep):

```bash
bash tests/run_minimal_regression.sh
```

## Deprecated modules kept only for contrast

The following modules remain in the tree only as explicit deprecated stubs and are not used
by the production path anymore:

- `geom/intersection.py`
- `geom/retriangle.py`
- `ops/t_junction.py`
- `ops/quality.py`

If they are called directly they raise an error explaining that the CGAL bridge must be used.
