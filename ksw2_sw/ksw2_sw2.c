#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ksw2.h"

/**
 * Smith-Waterman 局部比对（反对角线遍历算法）
 *
 * 核心思想：DP 矩阵按反对角线 d = i + j 遍历。
 * 同一条反对角线上的所有单元格之间没有数据依赖，可以并行计算。
 * 每个单元格只依赖反对角线 d-1 和 d-2 上的值，因此只需保存
 * 最近 3 条反对角线的 H 值和 2 条的 E/F 值（滚动缓冲区）。
 *
 * 空间复杂度：O(min(qlen, tlen)) 用于 DP 计算
 *             O((qlen+tlen) * min(qlen,tlen)) 用于回溯矩阵
 *
 * @param km        内存池
 * @param qlen      查询序列长度
 * @param query     查询序列（编码后, 0 <= query[i] < m）
 * @param tlen      目标序列长度
 * @param target    目标序列（编码后, 0 <= target[i] < m）
 * @param m         残基类型数量
 * @param mat       m*m 打分矩阵
 * @param gapo      缺口开放罚分（正数）
 * @param gape      缺口延伸罚分（正数）
 * @param w         带宽限制（<0 表示不限制）
 * @param m_cigar_  CIGAR 最大长度（输入/输出）
 * @param n_cigar_  CIGAR 元素数量（输出）
 * @param cigar_    CIGAR 数组（输出）
 * @return          最大局部比对得分
 */
int ksw_sw_standard(void *km, int qlen, const uint8_t *query, int tlen, const uint8_t *target,
                    int8_t m, const int8_t *mat, int8_t gapo, int8_t gape, int w,
                    int *m_cigar_, int *n_cigar_, uint32_t **cigar_)
{
    if (qlen <= 0 || tlen <= 0) return 0;

    int gapoe = gapo + gape;  // 首次 gap 总罚分
    int best_score = 0, best_i = 0, best_j = 0;

    // 带宽：w < 0 表示不限制
    if (w < 0) w = (qlen > tlen) ? qlen : tlen;

    // 每条反对角线的最大单元格数
    int max_nd = (qlen < tlen ? qlen : tlen) + 1;
    if (2 * w + 1 < max_nd) max_nd = 2 * w + 1;
    if (max_nd < 2) max_nd = 2;  // 至少 2，保证边界对角线够用

    int n_diag = qlen + tlen + 1;  // 反对角线数量：d = 0 .. qlen+tlen

    // ===== 滚动缓冲区 =====
    // H 需要 3 条（d, d-1, d-2），E/F 各需要 2 条（d, d-1）
    int *H_buf[3], *E_buf[2], *F_buf[2];
    for (int b = 0; b < 3; b++)
        H_buf[b] = (int *)kcalloc(km, max_nd, sizeof(int));     // 边界 H=0
    for (int b = 0; b < 2; b++) {
        E_buf[b] = (int *)kmalloc(km, max_nd * sizeof(int));
        F_buf[b] = (int *)kmalloc(km, max_nd * sizeof(int));
        for (int k = 0; k < max_nd; k++) {
            E_buf[b][k] = KSW_NEG_INF;  // 边界 E=-inf
            F_buf[b][k] = KSW_NEG_INF;  // 边界 F=-inf
        }
    }

    // 回溯矩阵：按反对角线紧凑存储 trace[d * max_nd + k]
    uint8_t *trace = NULL;
    if (m_cigar_ && n_cigar_ && cigar_)
        trace = (uint8_t *)kcalloc(km, (size_t)n_diag * max_nd, 1);

    // ===== 反对角线遍历 =====
    // 反对角线 d：所有满足 i + j = d 的单元格 (i, j)
    //   i 范围：max(0, d-tlen) .. min(d, qlen)
    //   缓冲区索引 k = i - i_lo_full，其中 i_lo_full = max(0, d-tlen)
    //
    // 依赖关系：
    //   H[i][j] = max(0, H[i-1][j-1]+s, E[i][j], F[i][j])
    //   E[i][j] = max(H[i-1][j]-gapoe, E[i-1][j]-gape)   ← 来自 d-1
    //   F[i][j] = max(H[i][j-1]-gapoe, F[i][j-1]-gape)   ← 来自 d-1
    //   H[i-1][j-1]                                        ← 来自 d-2
    //
    // 同一反对角线内无依赖 → 可并行

    for (int d = 2; d <= qlen + tlen; d++) {
        int cur_h   = d % 3;
        int prev1_h = (d - 1) % 3;
        int prev2_h = (d - 2) % 3;
        int cur_ef  = d % 2;
        int prev_ef = (d - 1) % 2;

        // 当前反对角线的完整 i 范围（含边界 i=0 或 j=0）
        int i_lo_full = d - tlen; if (i_lo_full < 0) i_lo_full = 0;
        int i_hi_full = d;        if (i_hi_full > qlen) i_hi_full = qlen;

        // 前两条反对角线的 i_lo（用于索引映射）
        int i_lo_prev1 = (d - 1) - tlen; if (i_lo_prev1 < 0) i_lo_prev1 = 0;
        int i_lo_prev2 = (d - 2) - tlen; if (i_lo_prev2 < 0) i_lo_prev2 = 0;

        // 初始化当前缓冲区为边界默认值（H=0, E=F=-inf）
        for (int k = 0; k < max_nd; k++) {
            H_buf[cur_h][k] = 0;
            E_buf[cur_ef][k] = KSW_NEG_INF;
            F_buf[cur_ef][k] = KSW_NEG_INF;
        }

        // DP 计算：仅处理 i >= 1 且 j >= 1 的单元格
        int i_lo_dp = i_lo_full; if (i_lo_dp < 1) i_lo_dp = 1;
        int i_hi_dp = i_hi_full; if (i_hi_dp > d - 1) i_hi_dp = d - 1;

        for (int i = i_lo_dp; i <= i_hi_dp; i++) {
            int j = d - i;  // j >= 1 保证
            int k = i - i_lo_full;  // 缓冲区索引

            // 带宽约束
            if (abs(i - j) > w) {
                if (trace) trace[d * max_nd + k] = 3;  // reset
                continue;
            }

            // 匹配/错配分数
            int8_t ms = mat[query[i - 1] * m + target[j - 1]];

            // --- 对角线方向：H[i-1][j-1] + score（来自 d-2）---
            int k2 = (i - 1) - i_lo_prev2;
            int h_diag = H_buf[prev2_h][k2] + ms;

            // --- E[i][j]：上方删除（来自 d-1）---
            // E[i][j] = max(H[i-1][j] - gapoe, E[i-1][j] - gape)
            int k1e = (i - 1) - i_lo_prev1;
            int new_e = H_buf[prev1_h][k1e] - gapoe;
            int e_ext = E_buf[prev_ef][k1e] - gape;
            if (e_ext > new_e) new_e = e_ext;
            if (new_e < KSW_NEG_INF / 2) new_e = KSW_NEG_INF;

            // --- F[i][j]：左方插入（来自 d-1）---
            // F[i][j] = max(H[i][j-1] - gapoe, F[i][j-1] - gape)
            int k1f = i - i_lo_prev1;
            int new_f = H_buf[prev1_h][k1f] - gapoe;
            int f_ext = F_buf[prev_ef][k1f] - gape;
            if (f_ext > new_f) new_f = f_ext;
            if (new_f < KSW_NEG_INF / 2) new_f = KSW_NEG_INF;

            // --- H[i][j] = max(0, diag, E, F) ---
            int h = h_diag;
            int dir = 0;  // 对角线（匹配/错配）
            if (new_e > h) { h = new_e; dir = 1; }  // 上方（删除）
            if (new_f > h) { h = new_f; dir = 2; }  // 左方（插入）
            if (h < 0)     { h = 0;    dir = 3; }   // 重置（局部比对）

            H_buf[cur_h][k] = h;
            E_buf[cur_ef][k] = new_e;
            F_buf[cur_ef][k] = new_f;

            // 追踪最大得分
            if (h > best_score) {
                best_score = h;
                best_i = i;
                best_j = j;
            }

            // 记录回溯方向
            if (trace) trace[d * max_nd + k] = (uint8_t)dir;
        }
    }

    // ===== 回溯生成 CIGAR =====
    int n_cigar = 0, m_cigar = (m_cigar_ && *m_cigar_) ? *m_cigar_ : 0;
    uint32_t *cigar = cigar_ ? *cigar_ : NULL;

    if (trace && best_score > 0 && best_i > 0 && best_j > 0) {
        int i = best_i, j = best_j, d = i + j;

        while (i > 0 && j > 0) {
            int i_lo_full = d - tlen; if (i_lo_full < 0) i_lo_full = 0;
            int k = i - i_lo_full;
            if (k < 0 || k >= max_nd) break;

            int dir = trace[d * max_nd + k];
            if (dir == 3) break;  // 到达比对起点

            uint32_t op;
            if (dir == 0)      { op = KSW_CIGAR_MATCH; i--; j--; }
            else if (dir == 1) { op = KSW_CIGAR_DEL;   i--;      }
            else               { op = KSW_CIGAR_INS;        j--; }

            // 追加到 CIGAR（暂时反向）
            if (n_cigar == 0 || (cigar[n_cigar - 1] & 0xF) != op) {
                if (n_cigar >= m_cigar) {
                    m_cigar = m_cigar ? m_cigar * 2 : 8;
                    cigar = (uint32_t *)krealloc(km, cigar, m_cigar * sizeof(uint32_t));
                }
                cigar[n_cigar++] = (1 << 4) | op;
            } else {
                cigar[n_cigar - 1] += (1 << 4);
            }

            d = i + j;
        }

        // 反转 CIGAR（回溯是反向的）
        for (int k = 0; k < n_cigar / 2; k++) {
            uint32_t tmp = cigar[k];
            cigar[k] = cigar[n_cigar - 1 - k];
            cigar[n_cigar - 1 - k] = tmp;
        }
    }

    // ===== 清理 =====
    for (int b = 0; b < 3; b++) kfree(km, H_buf[b]);
    for (int b = 0; b < 2; b++) { kfree(km, E_buf[b]); kfree(km, F_buf[b]); }
    if (trace) kfree(km, trace);

    if (m_cigar_) *m_cigar_ = m_cigar;
    if (n_cigar_) *n_cigar_ = n_cigar;
    if (cigar_)   *cigar_   = cigar;

    return best_score;
}
