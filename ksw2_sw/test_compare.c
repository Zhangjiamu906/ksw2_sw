#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "kalloc.h"
#include "ksw2.h"

// SSE 版本尚未在 ksw2.h 中声明，此处前向声明
extern int ksw_sw_sse_full(void *km, int qlen, const uint8_t *query, int tlen, const uint8_t *target,
                           int8_t m, const int8_t *mat, int8_t gapo, int8_t gape, int w,
                           int *m_cigar_, int *n_cigar_, uint32_t **cigar_, int is_int8);

// 碱基编码表 A=0, C=1, G=2, T=3, 其他=4
static uint8_t base2code[256] = {[0 ... 255] = 4};

static void init_base2code(void)
{
    base2code['A'] = base2code['a'] = 0;
    base2code['C'] = base2code['c'] = 1;
    base2code['G'] = base2code['g'] = 2;
    base2code['T'] = base2code['t'] = 3;
}

// 打印 CIGAR
static void print_cigar(int n_cigar, const uint32_t *cigar)
{
    static const char *opchr = "MIDN=X";
    for (int i = 0; i < n_cigar; i++)
    {
        int len = cigar[i] >> 4;
        int op = cigar[i] & 0xF;
        if (op < 6)
            printf("%d%c ", len, opchr[op]);
        else if (op == 7)
            printf("%d= ", len);
        else if (op == 8)
            printf("%dX ", len);
    }
    printf("\n");
}

// 统计 CIGAR 消耗的 query 和 target 碱基数
static void count_cigar_bases(int n_cigar, const uint32_t *cigar, int *n_q, int *n_t)
{
    *n_q = 0;
    *n_t = 0;
    for (int i = 0; i < n_cigar; i++)
    {
        int len = cigar[i] >> 4;
        int op = cigar[i] & 0xF;
        if (op == 0 || op == 7 || op == 8) { *n_q += len; *n_t += len; } // M / = / X
        else if (op == 1) { *n_q += len; }  // I: 消耗 query
        else if (op == 2) { *n_t += len; }  // D: 消耗 target
    }
}

// 编码序列
static uint8_t *encode_seq(void *km, const char *seq, int len)
{
    uint8_t *enc = (uint8_t *)kmalloc(km, len);
    for (int i = 0; i < len; i++)
        enc[i] = base2code[(uint8_t)seq[i]];
    return enc;
}

// 分隔线
static void print_separator(const char *title)
{
    printf("\n============================================================\n");
    printf("  %s\n", title);
    printf("============================================================\n");
}

// 运行三种实现并对比
static void run_and_compare(void *km,
                             const char *query_str, const char *target_str,
                             int8_t m, const int8_t *mat,
                             int8_t gapo, int8_t gape, int w)
{
    int qlen = strlen(query_str);
    int tlen = strlen(target_str);

    uint8_t *query  = encode_seq(km, query_str, qlen);
    uint8_t *target = encode_seq(km, target_str, tlen);

    printf("Query  (%d): %s\n", qlen, query_str);
    printf("Target (%d): %s\n", tlen, target_str);
    printf("参数: gapo=%d, gape=%d, w=%d\n", gapo, gape, w);

    // 三种实现的得分和 CIGAR
    int scores[3] = {0, 0, 0};
    int n_cigars[3] = {0, 0, 0};
    uint32_t *cigars[3] = {NULL, NULL, NULL};
    int m_cigars[3] = {0, 0, 0};
    const char *names[3] = {"ksw_sw (标准DP)", "ksw_sw_standard (对角线)", "ksw_sw_sse_full (SSE)"};

    // 1) ksw_sw (ksw2_sw.c)
    scores[0] = ksw_sw(km, qlen, query, tlen, target,
                       m, mat, gapo, gape, w,
                       &m_cigars[0], &n_cigars[0], &cigars[0]);

    // 2) ksw_sw_standard (ksw2_sw2.c)
    scores[1] = ksw_sw_standard(km, qlen, query, tlen, target,
                                m, mat, gapo, gape, w,
                                &m_cigars[1], &n_cigars[1], &cigars[1]);

    // 3) ksw_sw_sse_full (ksw2_sw2_sse.c), 使用 int8 模式
    scores[2] = ksw_sw_sse_full(km, qlen, query, tlen, target,
                                m, mat, gapo, gape, w,
                                &m_cigars[2], &n_cigars[2], &cigars[2], 1);

    // 打印各实现结果
    printf("\n");
    for (int i = 0; i < 3; i++)
    {
        int nq = 0, nt = 0;
        if (n_cigars[i] > 0)
            count_cigar_bases(n_cigars[i], cigars[i], &nq, &nt);
        printf("  [%s]\n", names[i]);
        printf("    得分: %d\n", scores[i]);
        printf("    CIGAR: ");
        if (n_cigars[i] > 0)
            print_cigar(n_cigars[i], cigars[i]);
        else
            printf("(无)\n");
        printf("    比对覆盖: query %d/%d, target %d/%d\n", nq, qlen, nt, tlen);
    }

    // 对比得分
    printf("\n  --- 对比结果 ---\n");
    int all_same = 1;
    for (int i = 1; i < 3; i++)
    {
        if (scores[i] != scores[0])
        {
            printf("  ⚠ %s 得分(%d) 与 ksw_sw 得分(%d) 不一致！\n",
                   names[i], scores[i], scores[0]);
            all_same = 0;
        }
    }
    if (all_same && scores[0] > 0)
        printf("  ✓ 三种实现得分一致: %d\n", scores[0]);

    // 清理
    for (int i = 0; i < 3; i++)
    {
        if (cigars[i]) kfree(km, cigars[i]);
    }
    kfree(km, query);
    kfree(km, target);
}

// ==================== 示例1：SW 局部性 ====================
static void test_locality(void *km)
{
    print_separator("示例1：Smith-Waterman 局部比对特性");

    // 构造序列：两端完全不相关，中间有高度相似区域
    // Query:  10个T + ACGTACGTAC + 10个G  = 30 bases
    // Target: 10个A + ACGTACGTAC + 10个C  = 30 bases
    // 两端错配(T/A, G/C)分数为负，中间完全匹配分数为正
    // SW 应只比对中间区域，而非端到端

    const char *query  = "TTTTTTTTTTACGTACGTACGGGGGGGGGGG";
    const char *target = "AAAAAAAAAACGTACGTACCCCCCCCCCC";

    int8_t m = 4;
    int8_t mat[16] = {
        //  A  C  G  T
           2, -4, -4, -4, // A
          -4,  2, -4, -4, // C
          -4, -4,  2, -4, // G
          -4, -4, -4,  2  // T
    };

    run_and_compare(km, query, target, m, mat, /*gapo=*/4, /*gape=*/2, /*w=*/-1);

    // 说明
    printf("\n  [说明]\n");
    printf("  Query 和 Target 各30个碱基，两端完全不匹配，中间10个碱基完全相同。\n");
    printf("  SW 局部比对只对齐中间相似区域，不对齐两端不相关区域。\n");
    printf("  比对覆盖远小于序列总长度 → 体现 SW 的局部性。\n");
    printf("  若使用 NW 全局比对，则会端到端对齐，两端错配会大幅拉低总分。\n");
}

// ==================== 示例2：仿射 gap 罚分 ====================
static void test_affine_gap(void *km)
{
    print_separator("示例2：仿射 gap 罚分 vs 线性 gap 罚分");

    // 使用 ACGT 重复序列 vs ACG 重复序列
    // Query 比 Target 多出若干 T，需要通过 gap 来对齐
    // - 线性 gap (gapo=0): 开 gap 无额外代价，多个短 gap 和一个长 gap 等价
    // - 仿射 gap (gapo=4): 每开一个 gap 额外罚 gapo，算法倾向减少 gap 数量

    const char *query  = "ACGTACGTACGT";
    const char *target = "ACGACGACG";

    int8_t m = 4;
    int8_t mat[16] = {
        //  A  C  G  T
           2, -4, -4, -4,
          -4,  2, -4, -4,
          -4, -4,  2, -4,
          -4, -4, -4,  2
    };

    // --- 线性 gap (gapo=0) ---
    printf("\n  ---- 线性 gap (gapo=0, gape=2) ----\n");
    run_and_compare(km, query, target, m, mat, /*gapo=*/0, /*gape=*/2, /*w=*/-1);

    // --- 仿射 gap (gapo=4) ---
    printf("\n  ---- 仿射 gap (gapo=4, gape=2) ----\n");
    run_and_compare(km, query, target, m, mat, /*gapo=*/4, /*gape=*/2, /*w=*/-1);

    // 说明
    printf("\n  [说明]\n");
    printf("  线性 gap (gapo=0): 开 gap 无额外代价，gap 总罚分 = gap长度 × gape。\n");
    printf("    多个短 gap 和一个等长的 gap 代价相同，因此 CIGAR 中可能出现多个 gap。\n");
    printf("  仿射 gap (gapo>0): 每开一个新 gap 额外罚 gapo。\n");
    printf("    算法倾向减少 gap 数量，CIGAR 中 gap 更少但可能更长（合并）。\n");
    printf("    同时，gap 开放罚分可能使算法选择更短的比对区域来避免 gap。\n");
    printf("  对比两次运行的得分和 CIGAR，可观察到仿射罚分对比对结构的影响。\n");
}

int main()
{
    init_base2code();
    void *km = km_init();

    test_locality(km);
    test_affine_gap(km);

    km_destroy(km);
    printf("\n全部测试完成！\n");
    return 0;
}
