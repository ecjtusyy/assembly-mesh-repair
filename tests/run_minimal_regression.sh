#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q

python pipeline.py \
  --input \
  "tests/data/土块加底土（相互穿透、存在体积重叠）.obj" \
  "tests/data/整体元素土块底土（存在共享接触面）.obj" \
  "tests/data/基坑1.0（存在多部分贴合和局部重叠）.obj" \
  "tests/data/基坑单元格未合并（存在多部分贴合和局部重叠）.obj" \
  --output_dir tests/out/solid_regression \
  --mode solid \
  --report_json tests/out/solid_regression/report.json
