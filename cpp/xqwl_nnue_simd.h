// SIMD helpers for MyCChessSF NNUE (runtime CPU dispatch).
// Hot paths: acc_add_column (FT incremental), dot_i8_u8 (FC int8 matmul).
#ifndef XQWL_NNUE_SIMD_H
#define XQWL_NNUE_SIMD_H

#include <cstdint>

#include "xqwl_nnue_simd_vnni.h"

#if defined(__AVX2__)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace xqwl_nnue {
namespace simd {

enum class Backend : std::uint8_t {
  Scalar = 0,
  NEON = 1,
  AVX2 = 2,
  AVX512_VNNI = 3,
};

inline Backend &backend() {
  static Backend b = Backend::Scalar;
  return b;
}

inline const char *backend_name() {
  switch (backend()) {
  case Backend::AVX512_VNNI:
    return "AVX512_VNNI";
  case Backend::AVX2:
    return "AVX2";
  case Backend::NEON:
    return "NEON";
  default:
    return "Scalar";
  }
}

inline bool cpuid_leaf7_ebx(unsigned int &ebx_out) {
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
#if defined(_MSC_VER)
  int cpu_info[4];
  __cpuid(cpu_info, 0);
  if (cpu_info[0] < 7) {
    return false;
  }
  __cpuidex(cpu_info, 7, 0);
  ebx = static_cast<unsigned int>(cpu_info[1]);
#else
  __asm__ volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(0) : "memory");
  if (eax < 7) {
    return false;
  }
  eax = 7;
  ecx = 0;
  __asm__ volatile("cpuid" : "+a"(eax), "=b"(ebx), "+c"(ecx), "=d"(edx) : : "memory");
#endif
  ebx_out = ebx;
  return true;
}

inline bool os_supports_xcr_features(std::uint64_t required_mask) {
#if defined(__AVX2__)
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
#if defined(_MSC_VER)
  int cpu_info[4];
  __cpuid(cpu_info, 1);
  ecx = static_cast<unsigned int>(cpu_info[2]);
#else
  __asm__ volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(1) : "memory");
#endif
  if ((ecx & (1u << 27)) == 0) {
    return false;
  }
  std::uint64_t xcr0 = 0;
#if defined(_MSC_VER)
  xcr0 = static_cast<std::uint64_t>(_xgetbv(0));
#else
  xcr0 = static_cast<std::uint64_t>(__builtin_ia32_xgetbv(0));
#endif
  return (xcr0 & required_mask) == required_mask;
#else
  (void)required_mask;
  return false;
#endif
}

inline void init() {
  backend() = Backend::Scalar;

#if defined(__AVX2__)
  unsigned int ebx = 0;
  if (cpuid_leaf7_ebx(ebx) && (ebx & (1u << 5))) {
#if defined(MYCCHESSSF_HAS_VNNI)
    if ((ebx & (1u << 11)) && os_supports_xcr_features(0xe6)) {
      backend() = Backend::AVX512_VNNI;
      return;
    }
#endif
    if (os_supports_xcr_features(0x6)) {
      backend() = Backend::AVX2;
      return;
    }
  }
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  backend() = Backend::NEON;
#endif
}

inline int32_t dot_i8_u8_scalar(const std::int8_t *w, const std::uint8_t *x, int dim) {
  int32_t sum = 0;
  for (int i = 0; i < dim; ++i) {
    sum += static_cast<int32_t>(w[i]) * static_cast<int32_t>(x[i]);
  }
  return sum;
}

#if defined(__AVX2__)
inline int32_t dot_i8_u8_avx2(const std::int8_t *w, const std::uint8_t *x, int dim) {
  __m256i acc = _mm256_setzero_si256();
  int i = 0;
  for (; i + 32 <= dim; i += 32) {
    const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x + i));
    const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + i));
    const __m256i prod = _mm256_maddubs_epi16(xv, wv);
    __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod));
    __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod, 1));
    acc = _mm256_add_epi32(acc, lo);
    acc = _mm256_add_epi32(acc, hi);
  }
  alignas(32) int32_t parts[8];
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(parts), acc);
  int32_t sum = 0;
  for (int k = 0; k < 8; ++k) {
    sum += parts[k];
  }
  for (; i < dim; ++i) {
    sum += static_cast<int32_t>(w[i]) * static_cast<int32_t>(x[i]);
  }
  return sum;
}
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
inline int32_t dot_i8_u8_neon(const std::int8_t *w, const std::uint8_t *x, int dim) {
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  int i = 0;
  for (; i + 16 <= dim; i += 16) {
    const int8x16_t wv = vld1q_s8(w + i);
    const uint8x16_t xv = vld1q_u8(x + i);
    const int16x8_t p0 = vmull_s8(vget_low_s8(wv), vreinterpret_s8_u8(vget_low_u8(xv)));
    const int16x8_t p1 = vmull_s8(vget_high_s8(wv), vreinterpret_s8_u8(vget_high_u8(xv)));
    acc0 = vpadalq_s16(acc0, p0);
    acc1 = vpadalq_s16(acc1, p1);
  }
  int32x4_t acc = vaddq_s32(acc0, acc1);
  alignas(16) int32_t parts[4];
  vst1q_s32(parts, acc);
  int32_t sum = parts[0] + parts[1] + parts[2] + parts[3];
  for (; i < dim; ++i) {
    sum += static_cast<int32_t>(w[i]) * static_cast<int32_t>(x[i]);
  }
  return sum;
}
#endif

inline int32_t dot_i8_u8(const std::int8_t *w, const std::uint8_t *x, int dim) {
#if defined(MYCCHESSSF_HAS_VNNI)
  if (backend() == Backend::AVX512_VNNI) {
    return dot_i8_u8_avx512_vnni(w, x, dim);
  }
#endif
#if defined(__AVX2__)
  if (backend() == Backend::AVX2) {
    return dot_i8_u8_avx2(w, x, dim);
  }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  if (backend() == Backend::NEON) {
    return dot_i8_u8_neon(w, x, dim);
  }
#endif
  return dot_i8_u8_scalar(w, x, dim);
}

inline void acc_add_column(int32_t *acc, const std::int8_t *col, int dim, int sign) {
  if (sign == 0) {
    return;
  }
#if defined(__AVX2__)
  if (backend() == Backend::AVX2 || backend() == Backend::AVX512_VNNI) {
    int j = 0;
    for (; j + 8 <= dim; j += 8) {
      __m128i c8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(col + j));
      __m256i c32 = _mm256_cvtepi8_epi32(c8);
      __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(acc + j));
      if (sign > 0) {
        a = _mm256_add_epi32(a, c32);
      } else {
        a = _mm256_sub_epi32(a, c32);
      }
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(acc + j), a);
    }
    for (; j < dim; ++j) {
      acc[j] += sign * static_cast<int32_t>(col[j]);
    }
    return;
  }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  if (backend() == Backend::NEON) {
    const int32x4_t signv = vdupq_n_s32(sign > 0 ? 1 : -1);
    int j = 0;
    for (; j + 4 <= dim; j += 4) {
      int32x4_t c = vmovl_s16(vget_low_s16(vmovl_s8(vld1_s8(col + j))));
      int32x4_t a = vld1q_s32(acc + j);
      a = vaddq_s32(a, vmulq_s32(c, signv));
      vst1q_s32(acc + j, a);
    }
    for (; j < dim; ++j) {
      acc[j] += sign * static_cast<int32_t>(col[j]);
    }
    return;
  }
#endif
  if (sign > 0) {
    for (int j = 0; j < dim; ++j) {
      acc[j] += static_cast<int32_t>(col[j]);
    }
  } else {
    for (int j = 0; j < dim; ++j) {
      acc[j] -= static_cast<int32_t>(col[j]);
    }
  }
}

} // namespace simd
} // namespace xqwl_nnue

#endif // XQWL_NNUE_SIMD_H
