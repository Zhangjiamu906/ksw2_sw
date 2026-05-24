#include "ksw2.h"
#include <stdint.h>

typedef struct { int32_t h, e; } eh_t;

int ksw_sw(void *km, int qlen, const uint8_t *query, int tlen, const uint8_t *target,
            int8_t m, const int8_t *mat, int8_t gapo, int8_t gape, int w,
            int *m_cigar_, int *n_cigar_, uint32_t **cigar_)
{
    eh_t *eh;
    int8_t *qp;
    int32_t i, j, k, st, en, gapoe = gapo + gape;
    int32_t max_score = 0, max_i = 0, max_j = 0;
    int32_t n_col, *off = 0;
    uint8_t *z = 0;

    if (w < 0) w = qlen > tlen ? qlen : tlen;
    n_col = qlen < 2 * w + 1 ? qlen : 2 * w + 1;

    qp = (int8_t*)kmalloc(km, qlen * m);
    eh = (eh_t*)kcalloc(km, qlen + 1, sizeof(eh_t));

    if (m_cigar_ && n_cigar_ && cigar_) {
        *n_cigar_ = 0;
        z = (uint8_t*)kmalloc(km, (size_t)n_col * tlen);
        off = (int32_t*)kcalloc(km, tlen, 4);
    }

    // Precompute query profile
    for (k = 0; k < m; ++k) {
        const int8_t *p = &mat[k * m];
        for (j = 0; j < qlen; ++j)
            qp[k * qlen + j] = p[query[j]];
    }

    // Initialize: LOCAL → all start at 0
    for (j = 0; j <= qlen; ++j) {
        eh[j].h = 0;
        eh[j].e = KSW_NEG_INF;
    }

    for (i = 0; i < tlen; ++i) {
        int32_t f = KSW_NEG_INF, h1 = 0;
        int8_t *q = &qp[target[i] * qlen];

        st = i > w ? i - w : 0;
        en = i + w + 1 < qlen ? i + w + 1 : qlen;

        if (z) {
            uint8_t *zi = &z[(size_t)i * n_col];
            off[i] = st;
            for (j = st; j < en; ++j) {
                eh_t *p = &eh[j];
                int32_t h = p->h, e = p->e;
                uint8_t d = 0;
                p->h = h1;

                h += q[j];
                if (e > h) h = e, d = 1;
                if (f > h) h = f, d = 2;
                if (h < 0) h = 0, d = 3;

                if (h > max_score) {
                    max_score = h;
                    max_i = i;
                    max_j = j;
                }

                h1 = h;
                h -= gapoe;
                e -= gape;
                if (e > h) p->e = e, d |= 8;
                else       p->e = h;

                f -= gape;
                if (f > h) d |= 16; /* F continuation */
                else f = h;
                zi[j - st] = d;
            }
        } else {
            for (j = st; j < en; ++j) {
                eh_t *p = &eh[j];
                int32_t h = p->h, e = p->e;
                p->h = h1;

                h += q[j];
                if (e > h) h = e;
                if (f > h) h = f;
                if (h < 0) h = 0;

                if (h > max_score) {
                    max_score = h;
                    max_i = i;
                    max_j = j;
                }

                h1 = h;
                h -= gapoe;
                e -= gape;
                if (e > h) p->e = e;
                else       p->e = h;
            f -= gape;
            if (h > f) f = h;
            }
        }
        if (en <= qlen) eh[en].h = h1, eh[en].e = KSW_NEG_INF;
    }

    kfree(km, qp);
    kfree(km, eh);

    if (z) {
        /* Custom SW traceback: starts at (max_i, max_j), stops at reset (d&7==3) */
        int n_cigar = 0, mc = *m_cigar_;
        uint32_t *cigar = *cigar_;
        int ii = max_i, jj = max_j, state = 0;
        while (ii >= 0 && jj >= 0) {
            uint8_t tmp;
            if (jj < off[ii]) break;
            tmp = z[(size_t)ii * n_col + jj - off[ii]];
            if (state == 0) {
                state = tmp & 7;
                if (state == 3) break; /* SW reset → stop */
            } else {
                int cont = (state == 1) ? (tmp >> 3 & 1) : (tmp >> 4 & 1);
                if (!cont) {
                    state = 0;
                    state = tmp & 7;
                    if (state == 3) break;
                }
            }
            if (state == 0) {
                cigar = ksw_push_cigar(km, &n_cigar, &mc, cigar, KSW_CIGAR_MATCH, 1);
                --ii; --jj;
            } else if (state == 1) {
                cigar = ksw_push_cigar(km, &n_cigar, &mc, cigar, KSW_CIGAR_DEL, 1);
                --ii;
            } else {
                cigar = ksw_push_cigar(km, &n_cigar, &mc, cigar, KSW_CIGAR_INS, 1);
                --jj;
            }
        }
        for (ii = 0; ii < n_cigar >> 1; ++ii) {
            uint32_t t = cigar[ii];
            cigar[ii] = cigar[n_cigar - 1 - ii];
            cigar[n_cigar - 1 - ii] = t;
        }
        *m_cigar_ = mc; *n_cigar_ = n_cigar; *cigar_ = cigar;
        kfree(km, z);
        kfree(km, off);
    }
    return max_score;
}