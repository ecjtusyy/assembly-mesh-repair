# Environment blocker logs captured during implementation

## 1. Git clone attempt

```text
git clone https://github.com/ecjtusyy/assembly-mesh-repair.git
fatal: unable to access 'https://github.com/ecjtusyy/assembly-mesh-repair.git/': Could not resolve host: github.com
```

## 2. apt update attempt

```text
apt-get update
Err:1 http://deb.debian.org/debian trixie InRelease
  Temporary failure resolving 'deb.debian.org'
```

## 3. apt install attempt

```text
apt-get install -y libcgal-dev libmpfr-dev
E: Unable to locate package libcgal-dev
E: Package 'libmpfr-dev' has no installation candidate
```

## 4. CMake configure attempt

```text
cmake -S cgal_bridge -B build/cgal
CMake Error at CMakeLists.txt:10 (message):
  CGAL was not found.  Install libcgal-dev (and GMP/MPFR development packages) before configuring this bridge.
```

## 5. CMake build attempt after failed configure

```text
cmake --build build/cgal -j
gmake: Makefile: No such file or directory
gmake: *** No rule to make target 'Makefile'.  Stop.
```

## 6. End-to-end CLI attempt without built bridge

```text
python pipeline.py --input tests/data/tri_cross.obj --output_dir tests/out_cli
[CLI][ERROR] processing failed for tests/data/tri_cross.obj: CGAL bridge executable 'check_self_intersections' was not found under build/cgal. Build the bridge first with `cmake -S cgal_bridge -B build/cgal && cmake --build build/cgal -j`.
```
