#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <emmintrin.h>  // SSE2
#include <smmintrin.h>  // SSE4.1
#include "ksw2.h"

#define SIMD_WIDTH 16  // 128-bit = 16 x int8_t 或 8 x int16_t

// 前向声明
static int ksw_sw_sse_full_int8(void *km, int qlen, const uint8_t *query, int tlen,
                                const uint8_t *target, int8_t m, const int8_t *mat,
                                int8_t gapo, int8_t gape, int w,
                                int *m_cigar_, int *n_cigar_, uint32_t **cigar_);



/**
 * 全向量化 Smith-Waterman 局部比对（int8_t，16路并行）
 * 
 * 使用 SSE2/SSE4.1 同时处理 16 个单元格
 */
int ksw_sw_sse_full(void *km, int qlen, const uint8_t *query, int tlen, const uint8_t *target,
                    int8_t m, const int8_t *mat, int8_t gapo, int8_t gape, int w,
                    int *m_cigar_, int *n_cigar_, uint32_t **cigar_, int is_int8)
{
    if (qlen <= 0 || tlen <= 0) return 0;
    return ksw_sw_sse_full_int8(km, qlen, query, tlen, target, 
                                 m, mat, gapo, gape, w, 
                                 m_cigar_, n_cigar_, cigar_);
}

// ==================== int8_t 版本（16路并行） ====================
static int ksw_sw_sse_full_int8(void *km, int qlen, const uint8_t *query, int tlen, 
                                 const uint8_t *target, int8_t m, const int8_t *mat,
                                 int8_t gapo, int8_t gape, int w,
                                 int *m_cigar_, int *n_cigar_, uint32_t **cigar_)
{
    int gapoe = gapo + gape;
    int best_score = 0, best_i = 0, best_j = 0;
    
    if (w < 0) w = (qlen > tlen) ? qlen : tlen;
    
    // 对齐到 SIMD 宽度
    int max_nd = (qlen < tlen ? qlen : tlen) + 1;
    if (2 * w + 1 < max_nd) max_nd = 2 * w + 1;
    int max_nd_aligned = ((max_nd + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
    if (max_nd_aligned < SIMD_WIDTH * 2) max_nd_aligned = SIMD_WIDTH * 2;
    
    int n_diag = qlen + tlen + 1;
    
    // ===== 内存分配（16字节对齐） =====
    int8_t *H_buf[3], *E_buf[2], *F_buf[2];
    for (int b = 0; b < 3; b++) {
        H_buf[b] = (int8_t *)_mm_malloc(max_nd_aligned * sizeof(int8_t), 16);
        memset(H_buf[b], 0, max_nd_aligned);
    }
    
    __m128i neg_inf_i8 = _mm_set1_epi8(-128);  // int8_t 最小值作为负无穷
    for (int b = 0; b < 2; b++) {
        E_buf[b] = (int8_t *)_mm_malloc(max_nd_aligned * sizeof(int8_t), 16);
        F_buf[b] = (int8_t *)_mm_malloc(max_nd_aligned * sizeof(int8_t), 16);
        for (int k = 0; k < max_nd_aligned; k += SIMD_WIDTH) {
            _mm_store_si128((__m128i *)(E_buf[b] + k), neg_inf_i8);
            _mm_store_si128((__m128i *)(F_buf[b] + k), neg_inf_i8);
        }
    }
    
    // 回溯矩阵（可选）
    uint8_t *trace = NULL;
    if (m_cigar_ && n_cigar_ && cigar_) {
        trace = (uint8_t *)kcalloc(km, (size_t)n_diag * max_nd, 1);
    }
    
    // 预计算打分矩阵的 SIMD 查找表
    // score_vec[t][v] = [mat[query[t]][target[v]], mat[query[t]][target[v+1]], ...]
    __m128i **score_table = (__m128i **)_mm_malloc(qlen * sizeof(__m128i *), 16);
    for (int i = 0; i < qlen; i++) {
        score_table[i] = (__m128i *)_mm_malloc(tlen * sizeof(__m128i), 16);
        uint8_t q_char = query[i];
        for (int j = 0; j < tlen; j += SIMD_WIDTH) {
            int8_t scores[SIMD_WIDTH];
            int vec_len = (j + SIMD_WIDTH <= tlen) ? SIMD_WIDTH : (tlen - j);
            for (int k = 0; k < vec_len; k++) {
                scores[k] = mat[q_char * m + target[j + k]];
            }
            for (int k = vec_len; k < SIMD_WIDTH; k++) {
                scores[k] = -128;  // 填充负无穷
            }
            score_table[i][j / SIMD_WIDTH] = _mm_loadu_si128((__m128i *)scores);
        }
    }
    
    // ===== SIMD 常量 =====
    __m128i zero_i8 = _mm_setzero_si128();
    __m128i gapoe_vec = _mm_set1_epi8(gapoe);
    __m128i gape_vec = _mm_set1_epi8(gape);
    __m128i dir_match_vec = _mm_set1_epi8(0);
    __m128i dir_del_vec = _mm_set1_epi8(1);
    __m128i dir_ins_vec = _mm_set1_epi8(2);
    __m128i dir_reset_vec = _mm_set1_epi8(3);
    
    // ===== 主循环：按反对角线遍历（全向量化） =====
    for (int d = 2; d <= qlen + tlen; d++) {
        int cur_h   = d % 3;
        int prev1_h = (d - 1) % 3;
        int prev2_h = (d - 2) % 3;
        int cur_ef  = d % 2;
        int prev_ef = (d - 1) % 2;
        
        int i_lo = d - tlen; if (i_lo < 0) i_lo = 0;
        int i_hi = d;        if (i_hi > qlen) i_hi = qlen;
        
        int i_lo_prev1 = (d - 1) - tlen; if (i_lo_prev1 < 0) i_lo_prev1 = 0;
        int i_lo_prev2 = (d - 2) - tlen; if (i_lo_prev2 < 0) i_lo_prev2 = 0;
        
        // 初始化当前缓冲区（向量化）
        for (int k = 0; k < max_nd_aligned; k += SIMD_WIDTH) {
            _mm_store_si128((__m128i *)(H_buf[cur_h] + k), zero_i8);
            _mm_store_si128((__m128i *)(E_buf[cur_ef] + k), neg_inf_i8);
            _mm_store_si128((__m128i *)(F_buf[cur_ef] + k), neg_inf_i8);
        }
        
        // 向量化 DP 计算：一次处理 16 个连续的 i
        int i_start = (i_lo < 1) ? 1 : i_lo;
        int i_end = (i_hi > d - 1) ? d - 1 : i_hi;
        
        for (int i = i_start; i <= i_end; i += SIMD_WIDTH) {
            int vec_len = (i + SIMD_WIDTH <= i_end + 1) ? SIMD_WIDTH : (i_end - i + 1);
            if (vec_len < SIMD_WIDTH) {
                // 处理剩余部分（标量）
                for (int offset = 0; offset < vec_len; offset++) {
                    int ii = i + offset;
                    int j = d - ii;
                    if (abs(ii - j) > w) continue;
                    
                    int k_idx = ii - i_lo;
                    int k2_idx = (ii - 1) - i_lo_prev2;
                    int k1e_idx = (ii - 1) - i_lo_prev1;
                    int k1f_idx = ii - i_lo_prev1;
                    
                    int8_t ms = mat[query[ii - 1] * m + target[j - 1]];
                    int h_diag = H_buf[prev2_h][k2_idx] + ms;
                    int new_e = H_buf[prev1_h][k1e_idx] - gapoe;
                    int e_ext = E_buf[prev_ef][k1e_idx] - gape;
                    if (e_ext > new_e) new_e = e_ext;
                    if (new_e < -128) new_e = -128;
                    
                    int new_f = H_buf[prev1_h][k1f_idx] - gapoe;
                    int f_ext = F_buf[prev_ef][k1f_idx] - gape;
                    if (f_ext > new_f) new_f = f_ext;
                    if (new_f < -128) new_f = -128;
                    
                    int h = h_diag;
                    int dir = 0;
                    if (new_e > h) { h = new_e; dir = 1; }
                    if (new_f > h) { h = new_f; dir = 2; }
                    if (h < 0) { h = 0; dir = 3; }
                    
                    H_buf[cur_h][k_idx] = h;
                    E_buf[cur_ef][k_idx] = new_e;
                    F_buf[cur_ef][k_idx] = new_f;
                    
                    if (h > best_score) {
                        best_score = h;
                        best_i = ii;
                        best_j = j;
                    }
                    if (trace) trace[d * max_nd + k_idx] = dir;
                }
                break;
            }
            
            // 向量化核心：同时处理 16 个 i
            // 1. 收集当前块的 j 索引（用于检查带宽）
            int j_vals[SIMD_WIDTH];
            bool within_band[SIMD_WIDTH];
            for (int offset = 0; offset < SIMD_WIDTH; offset++) {
                int ii = i + offset;
                j_vals[offset] = d - ii;
                within_band[offset] = (abs(ii - j_vals[offset]) <= w);
            }
            
            // 2. 从 score_table 加载匹配分数（向量化）
            int score_tile_idx = (j_vals[0] - 1) / SIMD_WIDTH;
            __m128i ms_vec = _mm_setzero_si128();
            for (int offset = 0; offset < SIMD_WIDTH; offset++) {
                int ii = i + offset;
                int jj = j_vals[offset];
                if (within_band[offset] && ii >= 1 && jj >= 1) {
                    int q_idx = ii - 1;
                    int t_idx = jj - 1;
                    int tile = t_idx / SIMD_WIDTH;
                    int sub = t_idx % SIMD_WIDTH;
                    // 从预计算的 score_table 获取
                    __m128i tile_vec = score_table[q_idx][tile];
                    int8_t score = ((int8_t *)&tile_vec)[sub];
                    ((int8_t *)&ms_vec)[offset] = score;
                } else {
                    ((int8_t *)&ms_vec)[offset] = -128;
                }
            }
            
            // 3. 加载对角线方向的值（来自 d-2）
            __m128i h_diag_vec = _mm_setzero_si128();
            for (int offset = 0; offset < SIMD_WIDTH; offset++) {
                int ii = i + offset;
                if (within_band[offset]) {
                    int k2 = (ii - 1) - i_lo_prev2;
                    ((int8_t *)&h_diag_vec)[offset] = H_buf[prev2_h][k2];
                } else {
                    ((int8_t *)&h_diag_vec)[offset] = -128;
                }
            }
            h_diag_vec = _mm_adds_epi8(h_diag_vec, ms_vec);  // 饱和加法
            
            // 4. 计算 E（上方删除）
            __m128i new_e_vec = _mm_setzero_si128();
            for (int offset = 0; offset < SIMD_WIDTH; offset++) {
                int ii = i + offset;
                if (within_band[offset]) {
                    int k1e = (ii - 1) - i_lo_prev1;
                    int val1 = H_buf[prev1_h][k1e] - gapoe;
                    int val2 = E_buf[prev_ef][k1e] - gape;
                    int val = (val1 > val2) ? val1 : val2;
                    if (val < -128) val = -128;
                    ((int8_t *)&new_e_vec)[offset] = val;
                } else {
                    ((int8_t *)&new_e_vec)[offset] = -128;
                }
            }
            
            // 5. 计算 F（左方插入）
            __m128i new_f_vec = _mm_setzero_si128();
            for (int offset = 0; offset < SIMD_WIDTH; offset++) {
                int ii = i + offset;
                if (within_band[offset]) {
                    int k1f = ii - i_lo_prev1;
                    int val1 = H_buf[prev1_h][k1f] - gapoe;
                    int val2 = F_buf[prev_ef][k1f] - gape;
                    int val = (val1 > val2) ? val1 : val2;
                    if (val < -128) val = -128;
                    ((int8_t *)&new_f_vec)[offset] = val;
                } else {
                    ((int8_t *)&new_f_vec)[offset] = -128;
                }
            }
            
            // 6. 计算 H = max(0, diag, E, F)（向量化）
            __m128i h_vec = h_diag_vec;
            __m128i dir_vec = dir_match_vec;
            
            // 比较 E
            __m128i cmp_e = _mm_cmpgt_epi8(new_e_vec, h_vec);
            h_vec = _mm_max_epi8(h_vec, new_e_vec);  // SSE4.1
            dir_vec = _mm_blendv_epi8(dir_vec, dir_del_vec, cmp_e);
            
            // 比较 F
            __m128i cmp_f = _mm_cmpgt_epi8(new_f_vec, h_vec);
            h_vec = _mm_max_epi8(h_vec, new_f_vec);
            dir_vec = _mm_blendv_epi8(dir_vec, dir_ins_vec, cmp_f);
            
            // 比较 0
            __m128i cmp_zero = _mm_cmpgt_epi8(zero_i8, h_vec);
            h_vec = _mm_max_epi8(h_vec, zero_i8);
            dir_vec = _mm_blendv_epi8(dir_vec, dir_reset_vec, cmp_zero);
            
            // 7. 存储结果
            for (int offset = 0; offset < SIMD_WIDTH; offset++) {
                int ii = i + offset;
                if (within_band[offset]) {
                    int k = ii - i_lo;
                    H_buf[cur_h][k] = ((int8_t *)&h_vec)[offset];
                    E_buf[cur_ef][k] = ((int8_t *)&new_e_vec)[offset];
                    F_buf[cur_ef][k] = ((int8_t *)&new_f_vec)[offset];
                    
                    int h_val = ((int8_t *)&h_vec)[offset];
                    if (h_val > best_score) {
                        best_score = h_val;
                        best_i = ii;
                        best_j = d - ii;
                    }
                    if (trace) {
                        trace[d * max_nd + k] = ((int8_t *)&dir_vec)[offset];
                    }
                }
            }
        }
    }
    
    // ===== 回溯生成 CIGAR =====
    int m_cigar = 0, n_cigar = 0;
    uint32_t *cigar = NULL;
    if (trace && best_score > 0) {
        int i = best_i;
        int j = best_j;

        while (i > 0 && j > 0) {
            int d = i + j;
            int i_lo_d = (d - tlen > 0) ? (d - tlen) : 0;
            int k = i - i_lo_d;
            if (k < 0 || k >= max_nd) break;

            uint8_t dir = trace[d * max_nd + k];
            if (dir == 3) {
                break;
            } else if (dir == 0) {
                // 对角移动：匹配/错配统一用 M
                cigar = ksw_push_cigar(km, &n_cigar, &m_cigar, cigar, KSW_CIGAR_MATCH, 1);
                --i; --j;
            } else if (dir == 1) {
                // 上移：deletion（跳过query碱基，消耗target碱基）
                cigar = ksw_push_cigar(km, &n_cigar, &m_cigar, cigar, KSW_CIGAR_DEL, 1);
                --i;
            } else if (dir == 2) {
                // 左移：insertion（跳过target碱基，消耗query碱基）
                cigar = ksw_push_cigar(km, &n_cigar, &m_cigar, cigar, KSW_CIGAR_INS, 1);
                --j;
            } else {
                break;
            }
        }

        for (int k = 0; k < n_cigar >> 1; k++) {
            uint32_t tmp = cigar[k];
            cigar[k] = cigar[n_cigar - 1 - k];
            cigar[n_cigar - 1 - k] = tmp;
        }
    }
    
    // ===== 清理 =====
    for (int b = 0; b < 3; b++) _mm_free(H_buf[b]);
    for (int b = 0; b < 2; b++) {
        _mm_free(E_buf[b]);
        _mm_free(F_buf[b]);
    }
    for (int i = 0; i < qlen; i++) {
        _mm_free(score_table[i]);
    }
    _mm_free(score_table);
    if (trace) kfree(km, trace);
    
    if (m_cigar_) *m_cigar_ = m_cigar;
    if (n_cigar_) *n_cigar_ = n_cigar;
    if (cigar_)   *cigar_   = cigar;

    return best_score;
}

