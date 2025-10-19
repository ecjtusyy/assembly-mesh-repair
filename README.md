# 🧩 Assembly Mesh Non-manifold Repair

**——装配网格非流形修复管线（Octree + TriTri + T-Fix + Stitch + Manifold）**

一个 **简洁、可读、可扩展** 的教学型项目，用于修复装配网格（Assembly Mesh）在拼接与局部加密（Local Refinement）过程中产生的 **非流形问题**。

本项目以 **算法清晰度和可扩展性** 为首要目标，而非工业级鲁棒性。

---

## 🚀 管线概览

| 阶段 | 模块                                           | 功能描述                  |
| -- | -------------------------------------------- | --------------------- |
| ①  | **Octree Broad-phase**                       | 构建八叉树，快速筛选潜在相交三角形对    |
| ②  | **Triangle–Triangle Narrow-phase & Cutting** | 精确检测并在局部进行相交分割        |
| ③  | **T-junction Fix (Propagation)**             | 修复挂点（T形连接），传播式边拆分     |
| ④  | **Weld / Stitch**                            | 顶点与边的焊接与拼缝            |
| ⑤  | **Manifoldization**                          | 拓扑修复，确保边仅属于两面，实现流形化   |
| ⑥  | **Quality Cleanup**                          | 清除退化面、瘦长三角、重复顶点等低质量结构 |

> ⚠️ 若要在生产环境使用，请使用**精确几何谓词（exact predicates）**，并补全共面、重叠等复杂情况的处理。

---

## 🧠 项目背景

在 **装配建模（Assembly Modeling）** 与 **网格细化（Local Refinement）** 的流程中，经常出现以下问题：

* 部件间边界微重叠、缝隙；
* 顶点落在他人边上形成 **T-junction**；
* 交叠三角形导致非流形拓扑；
* 局部细化带来**不一致节点**与**几何裂缝**。

这些问题会直接影响后续的：

* 碰撞检测
* 仿真（FEA/CFD）
* 曲面重建与CAD反求

本项目提供一条**轻量、可读、可调**的完整修复流程，帮助你快速理解“非流形修复”的核心思想。

---

## 🧩 安装与环境

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
```

**依赖环境：**

* Python 3.9+
* numpy 等科学计算库（详见 `requirements.txt`）

---

## ⚙️ 快速上手

### 修复两个装配零件

```bash
python pipeline.py --input partA.obj partB.obj --output_dir out
```

输出：

```
out/partA_repaired.obj
out/partB_repaired.obj
```

### 调整修复容差（相对包围盒对角线）

```bash
python pipeline.py --input partA.obj partB.obj --output_dir out \
  --eps_v 1e-6 --eps_e 1e-6 --eps_p 1e-8
```

---

## 🧾 参数说明

| 参数             | 含义     | 说明                      |
| -------------- | ------ | ----------------------- |
| `--input`      | 输入文件   | 支持多个 `.obj`（三角形）文件      |
| `--output_dir` | 输出目录   | 每个输入会生成 `_repaired.obj` |
| `--eps_v`      | 顶点吸附容差 | 用于顶点焊接、合并判断             |
| `--eps_e`      | 边吸附容差  | 用于边重合、拼接判断              |
| `--eps_p`      | 相交判定容差 | 控制三角形相交与剪切判定精度          |

> 所有容差均为**相对尺度**，根据输入网格的包围盒对角线进行归一化，保证不同尺寸模型下表现一致。

---

## 📥 输入与输出

* **输入**：标准三角形 OBJ 文件
  （若含多边形，将自动进行 **扇形三角化**）
* **输出**：每个输入文件生成对应的 `<name>_repaired.obj`
* **拓扑目标**：生成的网格应为**2-流形（2-manifold）**
  即每条边恰好被两个面共享；边界可存在，但无非流形连接。

---

## 🧰 模块说明

### 1️⃣ Octree Broad-phase

`geom/octree.py`
构建八叉树加速结构，快速筛选出可能相交的三角形对。

### 2️⃣ Tri-Tri Narrow-phase & Cutting

`geom/intersection.py`, `ops/cut.py`
检测真实相交并执行局部分割（当前为近似算法，可替换为精确几何谓词）。

### 3️⃣ T-junction Fix

`ops/t_fix.py`
检测并传播修复 T-junction（顶点落在他人边上）。

### 4️⃣ Weld / Stitch

`ops/stitch.py`
合并接近的顶点和边；默认允许**跨零件焊接**（可自定义以保留边界）。

### 5️⃣ Manifoldization

`ops/manifold.py`
确保边邻接关系正确；移除非流形点和退化三角。

### 6️⃣ Quality Cleanup

`ops/quality.py`
清理悬挂面、极瘦三角形、孤立顶点等，提升网格质量。

---

## 🔧 容差调节建议

| 现象             | 建议调整                            |
| -------------- | ------------------------------- |
| 交叠未切开 / 孔洞残留   | 增大 `--eps_p`（必要时同时调高 `--eps_e`） |
| 跨件被误焊接 / 细节被抹平 | 减小 `--eps_v` 与 `--eps_e`        |
| 出现大量瘦长三角或碎片    | 提高 `--eps_p` 后再执行质量清理           |

> 推荐初始设置：`eps_p < eps_e ≈ eps_v`，逐步微调，每次改动 1/3 数量级。

---

## 💡 使用示例

```bash
# 仅修复局部加密导致的 T-junction
python pipeline.py --input refined_bracket.obj --output_dir out \
  --eps_v 5e-7 --eps_e 5e-7 --eps_p 1e-7

# 修复两个装配零件并进行跨件焊接
python pipeline.py --input housing.obj insert.obj --output_dir out \
  --eps_v 1e-6 --eps_e 1e-6 --eps_p 1e-7
```

---

## 🧱 扩展方向

| 模块                     | 可扩展内容                                               |
| ---------------------- | --------------------------------------------------- |
| `geom/intersection.py` | 替换为 **exact predicates**（Shewchuk / CGAL），处理共面/共线重叠 |
| `ops/stitch.py`        | 支持“保留部件边界”、“分组焊接”、“对齐后焊接”等策略                        |
| 数据结构                   | 引入 **Half-edge / Winged-edge** 结构，支持拓扑编辑            |
| 后处理                    | 增加 **受约束重三角化**、**Delaunay-like 翻边优化**               |
| 性能优化                   | 并行 BVH / tiled 区域修复                                 |

---

## ⚠️ 已知限制

* 当前几何判定为近似算法，对共面或退化情况不完全鲁棒；
* 长距离共线重叠仅作有限分割处理；
* 网格质量优化为轻量清理，不替代完整重建。

---

## ❓ 常见问题（FAQ）

**Q:** 会改变零件边界吗？
**A:** 默认允许跨件焊接。若需保持零件边界，请在 `ops/stitch.py` 中添加限制条件。

**Q:** 支持多边形 OBJ 吗？
**A:** 会自动扇形三角化，仅支持三角面。

**Q:** 不同单位的模型容差会乱吗？
**A:** 不会。所有容差均基于相对包围盒尺寸归一化，单位无关。

---

## 🛠️ 开发路线（Roadmap）

* ✅ 基础 Octree + Tri-Tri 流程
* 🔄 精确几何谓词与共面检测
* 🔄 边对齐（共线重叠检测 + 分割）
* 🔄 Half-edge 拓扑结构与编辑操作
* 🔄 重三角化与质量翻边
* 🔄 并行化加速（BVH / tiled 操作）

---

## 🧾 简要命令参考

```bash
# 基本用法
python pipeline.py --input <obj files> --output_dir <out dir>

# 详细控制
python pipeline.py \
  --input partA.obj partB.obj \
  --output_dir out \
  --eps_v 1e-6 --eps_e 1e-6 --eps_p 1e-8
