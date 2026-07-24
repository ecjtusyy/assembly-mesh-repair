# assembly-mesh-repair

装配体三角网格修复与单区域四面体生成工具。主程序使用 Python，精确实体布尔运算由 Manifold3D 的 Python wheel 完成，不需要配置 CGAL 和 CMake。TetGen 用于默认的边界锁定四面体生成，Gmsh 用于可选的表面细分和允许容差修复的体网格生成。

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

需要边界锁定四面体生成时安装：

```bash
python -m pip install -r requirements-tetgen.txt
```

需要表面质量重剖分、均匀细分或宽松四面体生成时再安装：

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

### 保持几何的表面质量重剖分

均匀细分只会复制原三角形的角度，不能改善狭长三角形。对分片平面的机械模型应使用：

```bash
python pipeline.py \
  --input model.obj \
  --output_dir out \
  --mode solid \
  --quality_surface_remesh \
  --min_surface_angle 15 \
  --min_surface_mean_ratio 0.2 \
  --max_surface_condition 10 \
  --report_json out/report.json
```

该流程先把同标签、同平面的三角形合并成平面片，删除片内原有对角线，再由 Gmsh Frontal-Delaunay 在平面内部布点。以下内容是硬约束：

- 原直线特征边上的节点只能沿原线段增加；
- 新节点不能离开所属原平面；
- 每个平面片面积和闭合表面体积不能超过给定误差；
- 输出仍须通过闭合、绕序、非流形和自相交验收；
- 最小角、`mean-ratio` 或形状条件数任一不合格即失败。

`surface_target_size=0` 会根据平面片面积和周长估计尺寸，并限制片区之间的尺寸跳变。JSON 同时记录重剖分前后质量分位数、坏面数量和几何误差。

这个选项会改变三角形连接关系，但不改变分片平面的几何外形。若原几何本身含有尖角或极短特征边，严格保留几何与“所有三角形均满足最小角”可能互相冲突；程序会报告 `surface_quality_below_threshold`，不会删除特征或移动外轮廓来伪造成功。

从已经合法的闭合表面生成边界锁定四面体体网格：

```bash
python pipeline.py \
  --input examples/closed_tetra.obj \
  --output_dir out \
  --mode solid \
  --tetrahedralize \
  --tetra_mode strict \
  --min_tet_quality 0.05 \
  --report_json out/report.json
```

`strict` 是默认模式。它禁止焊点、补洞、均匀细分和近似重建，并使用 TetGen 的边界保护选项。若先显式开启 `--quality_surface_remesh`，TetGen 会逐点逐面锁定已经通过几何与质量验收的新表面。以下比较针对“进入 TetGen 的表面”；未开启质量重剖分时它就是原始 OBJ：

- 输入 TetGen 的顶点坐标逐字节相等；
- 输入 TetGen 的三角面顶点编号相同，允许输出调整面顺序和朝向；
- 边界上没有新增 Steiner 点；
- OBJ 的 `g` 边界分组写入 `.msh` 物理组；没有有效 `g` 时使用 `o`；
- JSON 中输入、输出边界的 SHA-256 必须一致。

因此严格模式只会在实体内部增加节点。输入边界不合法或固定边界限制了四面体质量时，程序会失败并报告原因，不会改成另一个模型。

严格模式中 `target_size=0` 表示不设置单元尺寸硬阈值。若显式指定尺寸，程序会同时检查实际最大四面体体积；固定边界使目标无法达到时，报告 `tetra_size_above_target`。宽松模式中 `target_size=0` 使用包围盒对角线的 `1/8`。输出包括：

- `*_solid_repaired.obj`：体网格使用的已验收边界；
- `*_solid_volume.msh`：带体区域和原始边界分组的一阶四面体；
- `*_solid_quality.vtk`：每个四面体的 `mean_ratio`，可用 ParaView 查看；
- JSON 报告：硬有效性、质量阈值、几何偏差和体积一致性证据。

第一阶段故意只支持单区域。输入有多个非空 `usemtl` 时程序会拒绝生成体网格，因为把材料界面直接并入一个 `domain` 会改变有限元问题。

如果原始 OBJ 不是合法闭合实体，必须先单独修复并由用户确认修复后的几何。只有用户明确允许表面在给定容差内变化时，才能选择 Gmsh 宽松模式：

```bash
python pipeline.py \
  --input model.obj \
  --output_dir out \
  --mode solid \
  --tetrahedralize \
  --tetra_mode relaxed \
  --max_geometry_deviation_rel 1e-6
```

宽松模式会进行近点处理和 Gmsh 离散曲面重建，所以不能宣称外边界完全不变。严格和宽松结果都要经过翻转、退化、重复单元、体边界、体积和质量验收。

若多零件装配体使用了约 \(10^{-6}\) 的建模偏移来制造相交，布尔并集后可能
残留极薄的数值伪特征。只有确认这些偏移不是实际结构尺寸时，才可在并集前
显式统一近重合坐标面。基坑回归模型的已验证命令为：

```bash
python pipeline.py \
  --input "tests/data/基坑1.0（存在多部分贴合和局部重叠）.obj" \
  --output_dir out \
  --mode solid \
  --pre_union_snap_rel 3e-8 \
  --tetrahedralize \
  --tetra_mode relaxed \
  --target_size 2 \
  --min_tet_quality 0.05 \
  --max_geometry_deviation_rel 1e-6 \
  --max_volume_error_rel 1e-6 \
  --report_json out/report.json
```

`--pre_union_snap_rel` 默认关闭。它只在各零件仍然独立时规范化坐标，保留
整个装配体包围盒，然后重新检查每个零件并执行实体并集。报告会记录坐标簇、
移动顶点数和最大位移。容差可能删除小于阈值的真实薄结构，因此不能根据
“网格更好看”自动开启。

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
原始闭合边界验收
→ TetGen 边界保护四面体生成
→ 输入/输出边界逐点逐面和 SHA-256 对比
→ 翻转、零体积、重复单元和边界一致性检查
→ mean-ratio 和总体积误差验收
→ 验收通过才标记 success
```

四面体的 `mean-ratio` 在 `0` 到 `1` 之间，正四面体为 `1`。程序把两类结论分开：

- 硬错误：翻转、零体积、重复四面体、体边界不一致；
- 可配置阈值：最低 `mean-ratio` 和最大相对体积误差。

`--tetra_mode relaxed` 才会改用 Gmsh，并额外检查双向表面偏差。

OBJ 的闭合三角表面可以交给 TetGen 或 Gmsh 生成四面体，但“能够生成”不等于“网格可用于可信有限元”。本项目因此保留生成前后的独立验收；它只能降低由离散网格本身引起的错误，不能证明材料参数、载荷、边界条件、本构模型和求解算法正确。

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
- 共面片内部狭长三角形的保持几何重剖分。

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

包含严格模式和 Gmsh 宽松模式的完整测试：

```bash
python -m pip install \
  -r requirements-dev.txt \
  -r requirements-tetgen.txt \
  -r requirements-gmsh.txt
python -m pytest -q
```

完整测试包含 TetGen 边界锁定和真实 Gmsh 四面体生成，并验证 `.msh`、质量 `.vtk`、物理分组、单元方向、边界一致性、几何偏差和体积误差。

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
