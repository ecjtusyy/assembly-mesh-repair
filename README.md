# assembly-mesh-repair

装配体三角网格修复工具。主程序只写 Python，精确实体布尔运算由 Manifold3D 的 Python wheel 完成，不需要配置 CGAL、CMake 和 Gmsh。

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

`solid` 模式不会把失败伪装成成功。输入零件不是闭合正体积时，精确布尔运算会停止；只有显式开启 `--approximate_rebuild` 才允许 PCU 重新生成近似外壳。

`surface` 模式处理组合拓扑，不执行开放曲面的精确三角形求交切分。严重自交三角汤需要明确选择近似重建。

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
