# Smith-Waterman 局部比对实现（基于 KSW2）

## 项目简介

本项目基于 [KSW2](https://github.com/lh3/ksw2) 的 Needleman-Wunsch (NW) 全局比对代码，将其改写为 Smith-Waterman (SW) 局部比对，支持仿射 gap 罚分（Gotoh 三状态 DP）。共提供三种实现：

| 文件 | 实现方式 | 说明 |
|------|---------|------|
| `ksw2_sw.c` | 标准 DP | 逐行填写 H/E/F 矩阵，逻辑最直观 |
| `ksw2_sw2.c` | 反对角线遍历 | 按反对角线 d=i+j 遍历 DP 矩阵，同一条对角线上无数据依赖可并行，空间 O(min(qlen,tlen)) |
| `ksw2_sw2_sse.c` | SSE 向量化 | 在对角线形式基础上用 SSE intrinsics 并行计算（int16 版本为骨架，未完整实现） |

## 文件结构

```
.
├── ksw2.h              # 头文件，接口声明与辅助宏
├── kalloc.h / kalloc.c # 内存池（kalloc）库
├── ksw2_sw.c           # 标准 DP 版 SW
├── ksw2_sw2.c          # 反对角线遍历版 SW
├── ksw2_sw2_sse.c      # SSE 向量化版 SW
├── test_sw.c           # 单独测试 ksw_sw（标准DP）
├── test_sw2.c          # 单独测试 ksw_sw_standard（对角线）
├── test_sw2_sse.c      # 单独测试 ksw_sw_sse_full（SSE）
├── test_compare.c      # 综合对比测试（验收示例）
└── README.md
```

## 编译与运行

### 依赖

- GCC（支持 C99 及 SSE4.1）
- Windows / Linux / macOS

### 编译综合对比测试

```bash
gcc -o test_compare test_compare.c kalloc.c ksw2_sw.c ksw2_sw2.c ksw2_sw2_sse.c -msse4.1 -O2
```

### 运行

```bash
./test_compare
```

### 单独编译各版本

```bash
# 标准 DP 版
gcc -o test_sw test_sw.c kalloc.c ksw2_sw.c -O2
./test_sw

# 对角线遍历版
gcc -o test_sw2 test_sw2.c kalloc.c ksw2_sw2.c -O2
./test_sw2

# SSE 向量化版
gcc -o test_sw2_sse test_sw2_sse.c kalloc.c ksw2_sw2_sse.c -msse4.1 -O2
./test_sw2_sse
```

## 接口说明

三个函数的签名风格与原始 `ksw_gg` / `ksw_gg2` / `ksw_gg2_sse` 保持一致：

```c
// 标准 DP 版（ksw2_sw.c）
int ksw_sw(void *km, int qlen, const uint8_t *query, int tlen, const uint8_t *target,
           int8_t m, const int8_t *mat, int8_t gapo, int8_t gape, int w,
           int *m_cigar_, int *n_cigar_, uint32_t **cigar_);

// 反对角线遍历版（ksw2_sw2.c）
int ksw_sw_standard(void *km, int qlen, const uint8_t *query, int tlen, const uint8_t *target,
                    int8_t m, const int8_t *mat, int8_t gapo, int8_t gape, int w,
                    int *m_cigar_, int *n_cigar_, uint32_t **cigar_);

// SSE 向量化版（ksw2_sw2_sse.c）
int ksw_sw_sse_full(void *km, int qlen, const uint8_t *query, int tlen, const uint8_t *target,
                    int8_t m, const int8_t *mat, int8_t gapo, int8_t gape, int w,
                    int *m_cigar_, int *n_cigar_, uint32_t **cigar_, int is_int8);
```

**参数说明**：

| 参数 | 含义 |
|------|------|
| `km` | kalloc 内存池，传 NULL 使用 malloc/free |
| `qlen` / `query` | query 序列长度及编码后序列（0 ≤ query[i] < m） |
| `tlen` / `target` | target 序列长度及编码后序列 |
| `m` | 字符表大小（DNA = 4） |
| `mat` | m×m 打分矩阵（一维数组） |
| `gapo` | gap open 罚分（≥ 0） |
| `gape` | gap extension 罚分（> 0） |
| `w` | 带宽限制（< 0 表示不限制） |
| `m_cigar_` / `n_cigar_` / `cigar_` | CIGAR 输出（BAM 编码） |
| `is_int8` | （SSE 版专用）1 = int8 模式，0 = int16 模式 |

**返回值**：SW 最优比对得分。

**CIGAR 编码**：使用 `KSW_CIGAR_MATCH`(M)、`KSW_CIGAR_INS`(I)、`KSW_CIGAR_DEL`(D)，与原始 KSW2 一致。每个 CIGAR 元素为 `uint32_t`，低 4 位为操作类型，高位为长度。

## NW → SW 改写要点

### 1. 初始化差异

| | NW（全局） | SW（局部） |
|---|-----------|-----------|
| H\[0\]\[j\], H\[i\]\[0\] | 累积 gap 罚分 | 0 |
| E\[0\]\[j\], F\[i\]\[0\] | -(gapo + j×gape) | -∞ 或 -(gapo + j×gape) |

### 2. 递推差异

SW 在 H 的递推中多一步 `max(0, ...)`：当 H 变为负数时重置为 0，表示从该位置可以自由开始新的比对。

### 3. 回溯差异

- NW：从矩阵右下角回溯到左上角
- SW：从矩阵中 H 最大值的位置回溯，直到遇到 H=0（重置点）停止

### 4. 三种实现的关键区别

**标准 DP（ksw2_sw.c）**：
- 逐行扫描，每格计算 H = max(0, diag+score, E, F)
- H < 0 时置 0，记录方向标记 d=3（SW 重置标记）
- 回溯从全局最大值出发，遇到 d=3 停止

**反对角线遍历（ksw2_sw2.c）**：
- 按反对角线 d = i + j 遍历，同一对角线上的单元格之间无依赖
- 使用滚动缓冲区保存最近 3 条对角线的 H 值和 2 条的 E/F 值
- 空间复杂度 O(min(qlen, tlen))，但回溯矩阵仍需 O((qlen+tlen) × min(qlen,tlen))
- SW 改动：边界 H=0，H 的 max(0,...) 通过负数置零实现

**SSE 向量化（ksw2_sw2_sse.c）**：
- 在对角线形式基础上，将同一对角线上的多个格用 128-bit SSE 寄存器并行计算
- `_mm_add_epi8` / `_mm_max_epi8` 等 intrinsics 替代标量运算
- 需要额外的 pack/unpack 处理以保持与标量版本等价
- SSE 缓冲区使用 `_mm_malloc` 分配（要求 16 字节对齐，`kalloc` 不保证对齐），回溯矩阵仍用 `kalloc`
- int8 版本（16 路并行）已完整实现；

## 验收示例说明

运行 `test_compare` 后输出两个示例：

### 示例 1：SW 局部性

```
Query  (31): TTTTTTTTTTACGTACGTACGGGGGGGGGGG
Target (29): AAAAAAAAAACGTACGTACCCCCCCCCCC
```

两端完全不匹配（T vs A, G vs C），中间 10 碱基 `ACGTACGTAC` 完全相同。

**预期输出**：
- 三种实现得分一致 = 20
- CIGAR = `10M`
- 比对覆盖 query 10/31、target 10/29

**说明**：SW 只对齐中间相似区域，两端不相关区域不参与比对。若使用 NW 全局比对，则端到端对齐，两端大量错配会大幅拉低总分。

### 示例 2：仿射 gap 罚分 vs 线性 gap 罚分

```
Query  (12): ACGTACGTACGT
Target (9):  ACGACGACG
```

同一输入分别用不同 gap 参数运行：

| 参数组 | gapo | gape | 含义 |
|--------|------|------|------|
| 线性 gap | 0 | 2 | 开 gap 无额外代价 |
| 仿射 gap | 4 | 2 | 每开一个新 gap 额外罚 4 |

**预期输出**：
- 线性 gap：得分 14，CIGAR 含多个 gap（如 `3M 1I 3M 1I 3M`）
- 仿射 gap：得分 6，CIGAR 中 gap 更少或比对区域更短（如 `3M`）

**说明**：
- 线性 gap（gapo=0）：gap 总罚分 = 长度 × gape，多个短 gap 和一个等长 gap 代价相同
- 仿射 gap（gapo>0）：每开新 gap 额外罚 gapo，算法更"吝啬"地开新 gap，倾向于合并相邻 gap 或缩短比对来避开 gap

## 参考文献

- Needleman, S. B., Wunsch, C. D. (1970). *A general method applicable to the search for similarities in the amino acid sequence of two proteins.*
- Smith, T. F., Waterman, M. S. (1981). *Identification of common molecular subsequences.*
- Gotoh, O. (1982). *An improved algorithm for matching biological sequences.*
- Suzuki, H., Kasahara, M. (2018). *Introducing difference recurrence relations for faster semi-global alignment of long sequences.*
