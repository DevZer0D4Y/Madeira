// SPDX-License-Identifier: GPL-3.0-or-later
// Madeira Converter Exception: see LICENSE-EXCEPTION.md
//
// Programs for the No JIT self-check. gen-selfcheck.py compiles them twice:
// for x86-64 with the Windows calling convention (-mabi=ms), and for 32-bit
// x86 with x87 floating point (-m32 -mfpmath=387), as older games are built.
// Each build is one flat, position-independent blob; each entry runs natively
// for its expected result, and app/Madeira/NoJITSelfCheckData.h holds both. On
// the device the interpreter runs the same bytes and must give the same results.
//
// Every entry takes (seed, a zeroed 64 KiB buffer, a callback) and returns a
// checksum. Nothing here calls a library: the blob is all there is.

#include <stddef.h>
#include <stdint.h>
#if defined(__x86_64__)
#include <emmintrin.h>
#include <pmmintrin.h>
#else
#include <mmintrin.h>
#include <xmmintrin.h>
#endif

#define ENTRY __attribute__((used, noinline))

typedef uint64_t (*callback_fn)(uint64_t, uint64_t);

// The compiler may call these for copies and clears.
void *memcpy(void *dst, const void *src, size_t n)
{
    uint8_t *d = dst;
    const uint8_t *s = src;
    while (n--) *d++ = *s++;
    return dst;
}

void *memset(void *dst, int c, size_t n)
{
    uint8_t *d = dst;
    while (n--) *d++ = (uint8_t)c;
    return dst;
}

static uint64_t next(uint64_t *x)
{
    *x = *x * 6364136223846793005ull + 1442695040888963407ull;
    return *x >> 17;
}

static uint64_t mix(uint64_t h, uint64_t v)
{
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h * 0xff51afd7ed558ccdull;
}

ENTRY uint64_t wl_crc32(uint64_t seed, uint8_t *buf, callback_fn cb)
{
    (void)cb;
    uint32_t *table = (uint32_t *)buf;
    uint8_t *data = buf + 1024;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        table[i] = c;
    }
    uint64_t x = seed;
    for (int i = 0; i < 8192; i++) data[i] = (uint8_t)next(&x);
    uint32_t crc = ~0u, crc2 = ~0u;
    for (int i = 0; i < 8192; i++) crc = table[(crc ^ data[i]) & 0xff] ^ (crc >> 8);
    for (int i = 0; i < 8192; i++) {
        crc2 ^= data[i];
        for (int k = 0; k < 8; k++) crc2 = (crc2 >> 1) ^ (0xEDB88320u & -(crc2 & 1));
    }
    return ((uint64_t)~crc << 32) | ~crc2;
}

static void quicksort(int64_t *a, long lo, long hi)
{
    while (lo < hi) {
        int64_t pivot = a[lo + (hi - lo) / 2];
        long i = lo, j = hi;
        while (i <= j) {
            while (a[i] < pivot) i++;
            while (a[j] > pivot) j--;
            if (i <= j) { int64_t t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
        }
        if (j - lo < hi - i) { quicksort(a, lo, j); lo = i; }
        else { quicksort(a, i, hi); hi = j; }
    }
}

ENTRY uint64_t wl_sort(uint64_t seed, uint8_t *buf, callback_fn cb)
{
    (void)cb;
    int64_t *a = (int64_t *)buf;
    uint64_t x = seed, h = 0;
    for (int i = 0; i < 3000; i++) a[i] = (int64_t)(next(&x) << 20) - (1ll << 50);
    quicksort(a, 0, 2999);
    for (int i = 0; i < 3000; i++) {
        if (i && a[i - 1] > a[i]) return 0;
        h = mix(h, (uint64_t)a[i] * (uint64_t)(i + 1));
    }
    return h;
}

ENTRY uint64_t wl_sieve(uint64_t seed, uint8_t *buf, callback_fn cb)
{
    (void)cb;
    uint32_t limit = 40000 + (uint32_t)(seed % 20000);
    uint64_t count = 0, sum = 0;
    for (uint32_t i = 2; i * i < limit; i++)
        if (!buf[i])
            for (uint32_t j = i * i; j < limit; j += i) buf[j] = 1;
    for (uint32_t i = 2; i < limit; i++)
        if (!buf[i]) { count++; sum += i * (uint64_t)i; }
    return (count << 40) ^ sum;
}

ENTRY uint64_t wl_divide(uint64_t seed, uint8_t *buf, callback_fn cb)
{
    (void)buf; (void)cb;
    uint64_t x = seed, h = 1;
    for (int i = 0; i < 4000; i++) {
        uint64_t a = next(&x) << (i & 31), b = (next(&x) >> (i % 40)) | 1;
        int64_t sa = (int64_t)a, sb = (int64_t)b * ((i & 1) ? -1 : 1);
        h = mix(h, a / b);
        h = mix(h, a % b);
        h = mix(h, (uint64_t)(sa / sb));
        h = mix(h, (uint64_t)(sa % sb));
        h = mix(h, (uint32_t)a / (uint32_t)(b | 3));
        h = mix(h, (uint64_t)((int32_t)a % (int32_t)(sb | 5)));
        h = mix(h, (uint16_t)a / (uint16_t)(b | 7));
#if defined(__SIZEOF_INT128__)
        unsigned __int128 p = (unsigned __int128)a * b;
        __int128 q = (__int128)sa * sb;
        h = mix(h, (uint64_t)(p >> 64) ^ (uint64_t)p);
        h = mix(h, (uint64_t)(q >> 64) ^ (uint64_t)q);
#else
        h = mix(h, a * b);
        h = mix(h, (uint64_t)(sa * sb));
#endif
    }
    return h;
}

ENTRY uint64_t wl_float(uint64_t seed, uint8_t *buf, callback_fn cb)
{
    (void)cb;
    double *m1 = (double *)buf, *m2 = m1 + 144, *m3 = m2 + 144;
    float *f = (float *)(m3 + 144);
    uint64_t x = seed, h = 0;
    for (int i = 0; i < 144; i++) {
        m1[i] = (double)(int64_t)(next(&x) % 20001 - 10000) / 97.0;
        m2[i] = (double)(next(&x) % 1000) * 0.001 + 1e-3;
    }
    for (int r = 0; r < 12; r++)
        for (int c = 0; c < 12; c++) {
            double acc = 0;
            for (int k = 0; k < 12; k++) acc += m1[r * 12 + k] * m2[k * 12 + c];
            m3[r * 12 + c] = acc;
        }
    for (int i = 0; i < 144; i++) {
        double v = m3[i];
        union { double d; uint64_t u; } u = { v };
        h = mix(h, u.u);
        h = mix(h, (uint64_t)(int64_t)(v * 1000.0));
        h = mix(h, (uint64_t)(int32_t)v);
        double s = __builtin_sqrt(v < 0 ? -v : v);
        union { double d; uint64_t u; } us = { s };
        h = mix(h, us.u);
        f[i] = (float)v / 3.0f + (float)i;
        union { float f; uint32_t u; } uf = { f[i] };
        h = mix(h, uf.u);
        h = mix(h, v < m1[i] ? 1 : v == m1[i] ? 2 : 3);
    }
    float acc = 0;
    for (int i = 0; i < 144; i++) acc += f[i] * f[(i * 7) % 144];
    union { float f; uint32_t u; } ua = { acc };
    return mix(h, ua.u);
}

#if defined(__x86_64__)
ENTRY uint64_t wl_simd(uint64_t seed, uint8_t *buf, callback_fn cb)
{
    (void)cb;
    __m128i *v = (__m128i *)buf;
    uint64_t x = seed, h = 0;
    for (int i = 0; i < 256; i++) v[i] = _mm_set_epi64x((long long)next(&x) * 7919, (long long)next(&x) * 104729);
    __m128i acc = _mm_setzero_si128(), acc2 = _mm_set1_epi32(0x01020304);
    for (int i = 0; i < 255; i++) {
        __m128i a = v[i], b = v[i + 1];
        acc = _mm_add_epi32(acc, _mm_mullo_epi16(a, b));
        acc = _mm_xor_si128(acc, _mm_shuffle_epi32(b, 0x93));
        acc2 = _mm_adds_epu8(acc2, _mm_avg_epu8(a, b));
        acc2 = _mm_sub_epi16(acc2, _mm_madd_epi16(a, b));
        __m128i m = _mm_cmpgt_epi16(a, b);
        h = mix(h, (uint64_t)_mm_movemask_epi8(m));
        h = mix(h, (uint64_t)_mm_cvtsi128_si64(_mm_sad_epu8(a, b)));
        __m128i packed = _mm_packs_epi32(_mm_srai_epi32(a, 9), _mm_srli_epi32(b, 13));
        acc = _mm_add_epi64(acc, _mm_unpackhi_epi8(packed, _mm_slli_si128(a, 3)));
        acc2 = _mm_max_epi16(acc2, _mm_min_epi16(a, _mm_srli_si128(b, 5)));
        __m128 fa = _mm_cvtepi32_ps(a), fb = _mm_cvtepi32_ps(_mm_srai_epi32(b, 8));
        __m128 fr = _mm_add_ps(_mm_mul_ps(fa, _mm_set1_ps(0.25f)), _mm_div_ps(fb, _mm_set1_ps(3.0f)));
        fr = _mm_hadd_ps(fr, _mm_max_ps(fa, fb));
        acc = _mm_xor_si128(acc, _mm_cvttps_epi32(fr));
        __m128d da = _mm_cvtepi32_pd(a);
        da = _mm_sqrt_pd(_mm_mul_pd(da, da));
        h = mix(h, (uint64_t)_mm_cvtsd_si64(da));
        h = mix(h, (uint64_t)_mm_movemask_ps(_mm_cmplt_ps(fa, fb)));
    }
    h = mix(h, (uint64_t)_mm_cvtsi128_si64(acc));
    h = mix(h, (uint64_t)_mm_cvtsi128_si64(_mm_unpackhi_epi64(acc, acc)));
    h = mix(h, (uint64_t)_mm_cvtsi128_si64(acc2));
    return mix(h, (uint64_t)_mm_cvtsi128_si64(_mm_unpackhi_epi64(acc2, acc2)));
}
#else
// MMX and the integer SSE additions to it (PSHUFW, PSADBW, PMINUB...), as games
// of the Pentium III era used them.
ENTRY uint64_t wl_mmx(uint64_t seed, uint8_t *buf, callback_fn cb)
{
    (void)cb;
    __m64 *v = (__m64 *)buf;
    uint64_t x = seed, h = 0;
    for (int i = 0; i < 256; i++) v[i] = _mm_set_pi32((int)next(&x), (int)next(&x));
    __m64 acc = _mm_setzero_si64(), acc2 = _mm_set1_pi16(0x0102);
    for (int i = 0; i < 255; i++) {
        __m64 a = v[i], b = v[i + 1];
        acc = _mm_add_pi32(acc, _mm_madd_pi16(a, b));
        acc = _mm_xor_si64(acc, _mm_unpackhi_pi8(a, b));
        acc2 = _mm_adds_pu8(acc2, _mm_avg_pu8(a, b));
        acc2 = _mm_subs_pi16(acc2, _mm_mullo_pi16(a, b));
        acc2 = _mm_max_pi16(acc2, _mm_min_pi16(a, _mm_srli_si64(b, 13)));
        acc = _mm_add_pi16(acc, _mm_packs_pi32(_mm_srai_pi32(a, 9), _mm_srli_pi32(b, 13)));
        acc = _mm_xor_si64(acc, _mm_shuffle_pi16(b, 0x93));
        h = mix(h, (uint64_t)_mm_movemask_pi8(_mm_cmpgt_pi16(a, b)));
        h = mix(h, (uint64_t)(uint32_t)_mm_cvtsi64_si32(_mm_sad_pu8(a, b)));
        h = mix(h, (uint64_t)(uint32_t)_mm_cvtsi64_si32(_mm_unpacklo_pi16(acc, acc2)));
    }
    union { __m64 m; uint64_t u; } r1 = { acc }, r2 = { acc2 };
    _mm_empty();
    return mix(mix(h, r1.u), r2.u);
}

// x87 work as a 1990s compiler leaves it: double arithmetic on the stack,
// float and integer conversions (FISTP after FLDCW to truncation for C casts),
// 64-bit integers through FILD/FISTP, comparisons through FNSTSW.
ENTRY uint64_t wl_x87(uint64_t seed, uint8_t *buf, callback_fn cb)
{
    (void)cb;
    double *p = (double *)buf;
    float *f = (float *)(buf + 8192);
    uint64_t x = seed, h = 0;
    for (int i = 0; i < 512; i++) {
        p[i] = (double)(int64_t)(next(&x) % 2000001 - 1000000) / 64.0;
        f[i] = (float)(next(&x) % 10000) * 0.01f - 50.0f;
    }
    double acc = 0, poly = 0;
    for (int i = 0; i < 512; i++) {
        double t = p[i] * 0.001;
        poly = ((((t * 0.5 - 1.25) * t + 2.0) * t - 0.75) * t + 0.125);
        acc += poly * (double)f[i] - __builtin_sqrt(p[i] * p[i] + 1.0);
        int truncated = (int)(p[i] / 3.0);
        long long wide = (long long)(p[i] * 1048576.0);
        h = mix(h, (uint64_t)(uint32_t)truncated);
        h = mix(h, (uint64_t)wide);
        h = mix(h, (uint64_t)(int64_t)((double)wide / 7.0));
        h = mix(h, p[i] < f[i] ? 1 : p[i] > f[i] ? 2 : 3);
        float g = (float)(acc / 1024.0);
        union { float v; uint32_t u; } ug = { g };
        h = mix(h, ug.u);
    }
    union { double v; uint64_t u; } ua = { acc }, up = { poly };
    return mix(mix(h, ua.u), up.u);
}
#endif

ENTRY uint64_t wl_strings(uint64_t seed, uint8_t *buf, callback_fn cb)
{
    (void)cb;
    uint64_t x = seed, h = 0;
    uint8_t *a = buf, *b = buf + 8192;
    for (int i = 0; i < 4096; i++) a[i] = (uint8_t)(next(&x) % 26 + 'a');
    for (int round = 0; round < 40; round++) {
        size_t n = 64 + next(&x) % 2000, off = next(&x) % 1000;
        void *d = b + off;
        const void *s = a + (next(&x) % 1000);
        size_t c = n;
        __asm__ volatile("rep movsb" : "+D"(d), "+S"(s), "+c"(c) : : "memory");
        void *e = b + 4096 + round * 8;
#if defined(__x86_64__)
        uint64_t fill = 0x0101010101010101ull * (uint8_t)round;
        c = 16;
        __asm__ volatile("rep stosq" : "+D"(e), "+c"(c) : "a"(fill) : "memory");
#else
        uint32_t fill = 0x01010101u * (uint8_t)round;
        c = 32;
        __asm__ volatile("rep stosl" : "+D"(e), "+c"(c) : "a"(fill) : "memory");
#endif
        const void *p = b + off, *q = a;
        c = n;
        __asm__ volatile("repe cmpsb" : "+D"(p), "+S"(q), "+c"(c) : : "memory", "cc");
        h = mix(h, c);
        const void *r = b + off;
        c = n;
        __asm__ volatile("repne scasb" : "+D"(r), "+c"(c) : "a"('q') : "memory", "cc");
        h = mix(h, c);
        // Overlapping copies, forward, and backward with the direction flag.
        d = b + off + 3; s = b + off; c = 100;
        __asm__ volatile("rep movsb" : "+D"(d), "+S"(s), "+c"(c) : : "memory");
        d = b + off + 107; s = b + off + 100; c = 50;
        __asm__ volatile("std\n\trep movsb\n\tcld" : "+D"(d), "+S"(s), "+c"(c) : : "memory");
    }
    for (int i = 0; i < 4096 + 320; i++) h = mix(h, b[i]);
    // Text: decimal formatting and a UTF-8 to UTF-16 conversion.
    char *text = (char *)buf + 20000;
    uint16_t *wide = (uint16_t *)(buf + 30000);
    int len = 0;
    for (int i = 0; i < 300; i++) {
        uint64_t v = next(&x) >> (i % 50);
        char digits[24];
        int k = 0;
        do { digits[k++] = (char)('0' + v % 10); v /= 10; } while (v);
        while (k) text[len++] = digits[--k];
        text[len++] = (i % 7 == 0) ? (char)0xc3 : ' ';
        if (i % 7 == 0) text[len++] = (char)0xa9;
    }
    int w = 0;
    for (int i = 0; i < len;) {
        uint8_t c0 = (uint8_t)text[i];
        if (c0 < 0x80) { wide[w++] = c0; i++; }
        else { wide[w++] = (uint16_t)(((c0 & 0x1f) << 6) | ((uint8_t)text[i + 1] & 0x3f)); i += 2; }
    }
    for (int i = 0; i < w; i++) h = mix(h, wide[i]);
    return h;
}

typedef uint64_t (*op_fn)(uint64_t, uint64_t);
static uint64_t op_add(uint64_t a, uint64_t b) { return a + b; }
static uint64_t op_xor(uint64_t a, uint64_t b) { return a ^ (b << 3); }
static uint64_t op_mul(uint64_t a, uint64_t b) { return a * (b | 1); }
static uint64_t op_rot(uint64_t a, uint64_t b) { return (a << (b & 63)) | (a >> ((64 - (b & 63)) & 63)); }
static uint64_t op_sub(uint64_t a, uint64_t b) { return a - b * 3; }

ENTRY uint64_t wl_dispatch(uint64_t seed, uint8_t *buf, callback_fn cb)
{
    (void)cb;
    // Built at run time: a static table of pointers would need load-time relocation.
    op_fn table[5] = { op_add, op_xor, op_mul, op_rot, op_sub };
    uint64_t x = seed, h = 7;
    for (int i = 0; i < 20000; i++) {
        uint64_t r = next(&x);
        switch (r % 13) {
        case 0: h += r; break;
        case 1: h ^= r >> 3; break;
        case 2: h = h * 31 + 7; break;
        case 3: h -= r << 2; break;
        case 4: h = (h >> 5) | (h << 59); break;
        case 5: h += buf[r & 1023]; buf[r & 1023] = (uint8_t)h; break;
        case 6: h ^= 0x5555; break;
        case 7: h = ~h; break;
        case 8: h += i; break;
        case 9: h = table[r % 5](h, r); break;
        case 10: h ^= h >> 17; break;
        case 11: h *= 0x100000001b3ull; break;
        default: h = table[(h >> 7) % 5](r, h); break;
        }
    }
    return h;
}

ENTRY uint64_t wl_callback(uint64_t seed, uint8_t *buf, callback_fn cb)
{
    (void)buf;
    uint64_t x = seed, h = 0;
    for (int i = 0; i < 2000; i++) {
        uint64_t a = next(&x), b = next(&x);
        h = mix(h, cb(a ^ h, b + (uint64_t)i));
    }
    return h;
}

ENTRY uint64_t wl_bits(uint64_t seed, uint8_t *buf, callback_fn cb)
{
    (void)buf; (void)cb;
    uint64_t x = seed, h = 0;
    for (int i = 0; i < 8000; i++) {
        uint64_t a = next(&x) * 0x9e3779b97f4a7c15ull | 1, b = next(&x) ^ (a << 7);
        unsigned n = (unsigned)(b & 63);
        h = mix(h, (uint64_t)__builtin_popcountll(a));
        h = mix(h, (uint64_t)__builtin_clzll(a));
        h = mix(h, (uint64_t)__builtin_ctzll(a << (i & 31)));
        h = mix(h, __builtin_bswap64(b));
        h = mix(h, __builtin_bswap32((uint32_t)b));
        h = mix(h, n ? (a << n) | (b >> (64 - n)) : a);
        h = mix(h, n ? (a >> n) | (b << (64 - n)) : a);
#if defined(__SIZEOF_INT128__)
        unsigned __int128 s = (unsigned __int128)a + b + h;
        h = mix(h, (uint64_t)(s >> 64));
#else
        h = mix(h, a + b + h);
#endif
        int64_t o;
        uint32_t o32;
        h = mix(h, (uint64_t)__builtin_add_overflow((int64_t)a, (int64_t)b, &o) + (uint64_t)o);
        h = mix(h, (uint64_t)__builtin_sub_overflow((uint32_t)a, (uint32_t)b, &o32) + o32);
        h = mix(h, (int64_t)a < (int64_t)b ? a : b);
        h = mix(h, (uint8_t)a > (uint8_t)b);
        h = mix(h, (uint64_t)((int64_t)a >> (b & 63)));
        h = mix(h, (uint64_t)((int16_t)a >> 3) + (uint64_t)(int8_t)b);
    }
    return h;
}

ENTRY uint64_t wl_atomic(uint64_t seed, uint8_t *buf, callback_fn cb)
{
    (void)cb;
    uint64_t *slots = (uint64_t *)buf;
#if defined(__SIZEOF_INT128__)
    unsigned __int128 *wide = (unsigned __int128 *)(buf + 4096);
#else
    uint64_t *wide = (uint64_t *)(buf + 4096);   // CMPXCHG8B
#endif
    uint32_t *small = (uint32_t *)(buf + 8192);
    uint64_t x = seed, h = 0;
    for (int i = 0; i < 3000; i++) {
        uint64_t r = next(&x);
        uint64_t *p = &slots[r % 64];
        h = mix(h, __atomic_fetch_add(p, r, __ATOMIC_SEQ_CST));
        h = mix(h, __atomic_fetch_or(p, r >> 9, __ATOMIC_SEQ_CST));
        h = mix(h, __atomic_exchange_n(&slots[(r >> 8) % 64], h, __ATOMIC_SEQ_CST));
        uint64_t expected = *p;
        h = mix(h, __atomic_compare_exchange_n(p, &expected, expected + 1, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
        expected = r;
        h = mix(h, __atomic_compare_exchange_n(p, &expected, 5, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) + expected);
#if defined(__SIZEOF_INT128__)
        unsigned __int128 old = wide[r % 8];
#else
        uint64_t old = wide[r % 8];
#endif
        h = mix(h, __sync_bool_compare_and_swap(&wide[r % 8], old, old + r));
        h = mix(h, __atomic_add_fetch(&small[r % 16], (uint32_t)r, __ATOMIC_SEQ_CST));
        h = mix(h, __atomic_fetch_sub(&small[(r >> 4) % 16], 3u, __ATOMIC_SEQ_CST));
    }
#if defined(__SIZEOF_INT128__)
    for (int i = 0; i < 8; i++) h = mix(h, (uint64_t)wide[i] ^ (uint64_t)(wide[i] >> 64));
#else
    for (int i = 0; i < 8; i++) h = mix(h, wide[i]);
#endif
    return h;
}

#if defined(__x86_64__)
ENTRY uint64_t wl_rounding(uint64_t seed, uint8_t *buf, callback_fn cb)
{
    (void)buf; (void)cb;
    uint32_t saved, mode;
    __asm__ volatile("stmxcsr %0" : "=m"(saved));
    uint64_t x = seed, h = 0;
    for (unsigned rc = 0; rc < 4; rc++) {
        for (unsigned flush = 0; flush < 2; flush++) {
            mode = (saved & ~0xe040u) | (rc << 13) | (flush ? 0x8040u : 0);
            __asm__ volatile("ldmxcsr %0" : : "m"(mode));
            for (int i = 0; i < 200; i++) {
                double a = (double)(int64_t)(next(&x) % 2000001 - 1000000) / 8.0;
                double b = (double)(next(&x) % 999 + 1) * 1e-300;
                int64_t r;
                __asm__ volatile("cvtsd2si %1, %0" : "=r"(r) : "x"(a));
                h = mix(h, (uint64_t)r);
                double q = a / 3.0;
                double tiny = b * 1e-10;
                float fs = (float)q;
                union { double d; uint64_t u; } u1 = { q }, u2 = { tiny };
                union { float f; uint32_t u; } u3 = { fs };
                h = mix(h, u1.u); h = mix(h, u2.u); h = mix(h, u3.u);
            }
        }
    }
    __asm__ volatile("ldmxcsr %0" : : "m"(saved));
    return h;
}
#else
ENTRY uint64_t wl_rounding(uint64_t seed, uint8_t *buf, callback_fn cb)
{
    (void)buf; (void)cb;
    uint16_t saved, mode;
    __asm__ volatile("fnstcw %0" : "=m"(saved));
    uint64_t x = seed, h = 0;
    for (unsigned rc = 0; rc < 4; rc++) {
        mode = (uint16_t)((saved & ~0x0c00u) | (rc << 10));
        __asm__ volatile("fldcw %0" : : "m"(mode));
        for (int i = 0; i < 200; i++) {
            double a = (double)(int64_t)(next(&x) % 2000001 - 1000000) / 8.0;
            int32_t r32;
            int64_t r64;
            __asm__ volatile("fistpl %0" : "=m"(r32) : "t"(a) : "st");
            __asm__ volatile("fistpll %0" : "=m"(r64) : "t"(a * 4096.0) : "st");
            h = mix(h, (uint32_t)r32);
            h = mix(h, (uint64_t)r64);
            double q = a / 3.0;
            float fs = (float)q;
            union { double d; uint64_t u; } u1 = { q };
            union { float f; uint32_t u; } u2 = { fs };
            h = mix(h, u1.u); h = mix(h, u2.u);
        }
    }
    __asm__ volatile("fldcw %0" : : "m"(saved));
    return h;
}
#endif
