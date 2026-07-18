# assembly-mesh-repair

装配体三角网格修复与单区域四面体生成工具。主程序使用 Python，精确实体布尔运算由 Manifold3D 的 Python wheel 完成，不需要配置 CGAL 和 CMake。Gmsh 用于可选的表面均匀细分和真实四面体体网格生成。

## 三种模式

| 模式 | 用途 | 处理方式 |
|---|---|---|
| `assembly` | 接触分析、保留多个零件 | 分零件清理，不跨零件焊点 |
| `surface` | 开放曲面、薄片模型 | 清理退化面、统一绕序、拆分非流形面扇，可补小孔 |
| `solid` | 3D 打印、统一实体外壳 | 先检查每个零件，再做精确布尔并集 |

程序读取 OBJ 时保留 `o`、`g` 和 `usemtl`。零件先按共享边拆分，坐标相同但拓扑独立的接触体不会被提前焊在一起。

## 安装

```bash
python -m pip install -r requirements.txt
```

只有需要近似重建时才安装：

```bash
python -m pip install -r requirements-approx.txt
```

需要 Gmsh 均匀细分或四面体生成时再安装：

```bash
python -m pip install -r requirements-gmsh.txt
```

Ubuntu 或 Codespaces 如果提示缺少 `libGLU.so.1` 或 `libXft.so.2`：

```bash
sudo apt-get update
sudo apt-get install -y libglu1-mesa libxft2
```

## 使用

保留装配体：

```bash
python pipeline.py \
  --input "tests/data/基坑1.0（存在多部分贴合和局部重叠）.obj" \
  --output_dir tests/out \
  --mode assembly \
  --report_json tests/out/report.json
```

合并成一个实体：

```bash
python pipeline.py \
  --input "tests/data/基坑1.0（存在多部分贴合和局部重叠）.obj" \
  --output_dir tests/out \
  --mode solid \
  --report_json tests/out/report.json
```

合并后使用 Gmsh 做一级均匀细分：

```bash
python pipeline.py \
  --input "tests/data/基坑1.0（存在多部分贴合和局部重叠）.obj" \
  --output_dir tests/out \
  --mode solid \
  --uniform_refine_levels 1
```

一级细分把每个三角形拆成 4 个，二级细分拆成 16 个。程序在细分前后都会重新验收，确保细分没有引入非流形边、非流形点、孔洞和体积错误。

从修复后的闭合表面生成并验收四面体体网格：

```bash
python pipeline.py \
  --input "tests/data/整体元素土块底土（存在共享接触面）.obj" \
  --output_dir tests/out \
  --mode solid \
  --tetrahedralize \
  --target_size 0.5 \
  --min_tet_quality 0.05 \
  --max_geometry_deviation_rel 1e-6 \
  --max_volume_error_rel 1e-6 \
  --report_json tests/out/report.json
```

`target_size=0` 时使用包围盒对角线的 `1/8`。输出包括：

- `*_solid_repaired.obj`：体网格使用的已验收边界；
- `*_solid_volume.msh`：带 `domain` 和 `boundary` 物理组的一阶四面体；
- `*_solid_quality.vtk`：每个四面体的 `mean_ratio`，可用 ParaView 查看；
- JSON 报告：硬有效性、质量阈值、几何偏差和体积一致性证据。

第一阶段故意只支持单区域。输入有多个非空 `usemtl` 时程序会拒绝生成体网格，因为把材料界面直接并入一个 `domain` 会改变有限元问题。

体网格前处理会从相对包围盒 `1e-7` 开始，自适应降低近点焊接容差，选择仍然闭合、流形且无自交的最大安全值。若几何中存在无法在允许偏差内消除的超薄区域，程序会保留 `.msh` 和质量 `.vtk`，但在 JSON 中标记 `tetra_quality_below_threshold`，不会把低质量网格写成成功。

修复开放表面并填补三角形、四边形小孔：

```bash
python pipeline.py \
  --input tests/data/mixed_case.obj \
  --output_dir tests/out \
  --mode surface \
  --fill_holes
```

精确实体合并失败，并且允许改变几何时，才能开启近似重建：

```bash
python pipeline.py \
  --input tests/data/tri_cross.obj \
  --output_dir tests/out \
  --mode solid \
  --approximate_rebuild \
  --rebuild_resolution 50000
```

## 修复顺序

```text
读取 OBJ 和零件身份
→ 按共享边拆分零件
→ 每个零件内部清理
→ 根据 mode 保留、拆分或合并
→ 检查非流形边、非流形顶点、绕序、闭合性和正体积
→ 验收失败就报错
```

开启 `--tetrahedralize` 后继续执行：

```text
闭合表面自相交检查
→ Gmsh 离散曲面分类和几何重建
→ 单区域四面体生成
→ Netgen 质量优化
→ 翻转、零体积、重复单元和边界一致性检查
→ mean-ratio、双向表面偏差和总体积误差验收
→ 验收通过才标记 success
```

四面体的 `mean-ratio` 在 `0` 到 `1` 之间，正四面体为 `1`。程序把两类结论分开：

- 硬错误：翻转、零体积、重复四面体、体边界不一致；
- 可配置阈值：最低 `mean-ratio` 和最大相对几何偏差。

OBJ 的闭合三角表面可以交给 Gmsh 生成四面体，但“Gmsh 能生成”不等于“网格可用于可信有限元”。本项目因此保留生成前后的独立验收；它只能降低由离散网格本身引起的错误，不能证明材料参数、载荷、边界条件、本构模型和求解算法正确。

`solid` 模式不会把失败伪装成成功。输入零件不是闭合正体积时，精确布尔运算会停止；只有显式开启 `--approximate_rebuild` 才允许 PCU 重新生成近似外壳。

`surface` 模式处理组合拓扑，不执行开放曲面的精确三角形求交切分。严重自交三角汤需要明确选择近似重建。

闭合实体输出会检查非相邻三角面的自相交。扫描先做包围盒宽相过滤，再做三角形求交；数值上接近的合法拼接点按相对包围盒容差视为邻接，避免把 Manifold3D 的浮点接缝误报成穿透。

## 当前范围

可以自动处理：

- 重复或极近顶点；
- 退化面、重复面、孤立顶点；
- 面绕序不一致；
- 非流形边和蝴蝶结顶点；
- 同一开放曲面内的 T 型接头；
- 三角形、四边形小孔；
- 多个闭合零件的接触、穿透和体积重叠；
- 严重三角汤的近似闭合重建。

下面几类输入没有唯一精确答案：

- 开放薄片要求自动猜测实体内外；
- 大面积缺失表面；
- 莫比乌斯结构；
- 需要恢复 CAD 曲面和材料语义的模型。

程序会保留开放表面、拒绝精确实体合并，或者由用户明确选择近似重建。

## 测试

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

回归测试包含人工构造的非流形边、面扇、孔洞、近似重建，以及仓库中的四个真实装配体模型。

包含 Gmsh 的完整测试：

```bash
python -m pip install -r requirements-dev.txt -r requirements-gmsh.txt
python -m pytest -q
```

完整测试还包含真实 Gmsh 四面体生成，并验证 `.msh`、质量 `.vtk`、单元方向、边界一致性、几何偏差和体积误差。

## 可视化验证

四个真实模型的修复前后对比图、修复后 OBJ 和逐项验收结果保存在
`docs/validation`。重新生成：

```bash
python -m pip install -r requirements-visual.txt
python tools/render_validation.py
```

脚本检查退化面、重复面、非流形边、非流形点、闭合性、绕序、正体积、
输出组件数和 Manifold3D 回读，并计算修复前零件体积之和与布尔并集体积之差。
任何一项失败都会直接报错，不会生成成功结论。
