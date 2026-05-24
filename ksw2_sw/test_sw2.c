#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "kalloc.h"
#include "ksw2.h"

// 局部比对的测试序列（query 比 target 长）
#define TEST_QUERY "ACGTACGTACACGTACGTACACGTACGTACACGTACGTACACGTACGTAC"
#define TEST_TARGET "CGTACGTACACGTACGTACAC"

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

int main()
{
    init_base2code();
    void *km = km_init();
    printf("内存池初始化成功\n");

    int qlen = strlen(TEST_QUERY);
    int tlen = strlen(TEST_TARGET);
    printf("Query  长度: %d\n", qlen);
    printf("Target 长度: %d\n", tlen);
    printf("Query:  %s\n", TEST_QUERY);
    printf("Target: %s\n", TEST_TARGET);

    uint8_t *query = (uint8_t *)kmalloc(km, qlen);
    uint8_t *target = (uint8_t *)kmalloc(km, tlen);
    for (int i = 0; i < qlen; i++)
        query[i] = base2code[(uint8_t)TEST_QUERY[i]];
    for (int i = 0; i < tlen; i++)
        target[i] = base2code[(uint8_t)TEST_TARGET[i]];

    // DNA 打分矩阵（A,C,G,T 顺序）
    int8_t m = 4; // 4 种碱基
    int8_t mat[4 * 4] = {
        //   A  C  G  T
        2, -4, -4, -4, // A
        -4, 2, -4, -4, // C
        -4, -4, 2, -4, // G
        -4, -4, -4, 2  // T
    };

    int8_t gapo = 4; // gap open penalty
    int8_t gape = 2; // gap extension penalty
    int w = -1;      // 不使用带宽限制

    // ==================== 测试 ksw_sw_standard (ksw2_sw2.c) ====================
    printf("\n========== 测试 ksw_sw_standard (ksw2_sw2.c) ==========\n");
    int m_cigar = 0, n_cigar = 0;
    uint32_t *cigar = NULL;
    int score = ksw_sw_standard(km, qlen, query, tlen, target,
                                m, mat, gapo, gape, w,
                                &m_cigar, &n_cigar, &cigar);
    printf("最佳得分: %d\n", score);
    printf("CIGAR 长度: %d\n", n_cigar);
    if (n_cigar > 0)
    {
        printf("CIGAR: ");
        print_cigar(n_cigar, cigar);
    }

    // 清理
    if (cigar) kfree(km, cigar);
    kfree(km, query);
    kfree(km, target);
    km_destroy(km);

    printf("\n运行完成！\n");
    return 0;
}
