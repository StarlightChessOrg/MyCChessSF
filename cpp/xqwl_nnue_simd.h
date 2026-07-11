// SIMD helpers for MyCChessSF NNUE (runtime CPU dispatch).
// Hot paths: acc_add_column (FT incremental), fc_row_i8_f32 (FC dequant matmul).
#ifndef XQWL_NNUE_SIMD_H
#define XQWL_NNUE_SIMD_H

#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

namespace xqwl_nnue {
namespace simd {

enum class Backend : std::uint8_t { Scalar = 0, AVX2 = 1 };

inline Backend &backend() {
  static Backend b = Backend::Scalar;
  return b;
}

inline void init() {
  backend() = Backend::Scalar;
#if defined(__AVX2__)
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
#if defined(_MSC_VER)
  int cpu_info[4];
  __cpuid(cpu_info, 0);
  const int max_leaf = cpu_info[0];
  if (max_leaf >= 7) {
    __cpuidex(cpu_info, 7, 0);
    ebx = static_cast<unsigned int>(cpu_info[1]);
    if (ebx & (1u << 5)) {
      backend() = Backend::AVX2;
    }
  }
#else
  __asm__ volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(0) : "memory");
  if (eax >= 7) {
    eax = 7;
    ecx = 0;
    __asm__ volatile("cpuid" : "+a"(eax), "=b"(ebx), "+c"(ecx), "=d"(edx) : : "memory");
    if (ebx & (1u << 5)) {
      backend() = Backend::AVX2;
    }
  }
#endif
#endif
}

inline void acc_add_column(int32_t *acc, const std::int8_t *col, int dim, int sign) {
  if (sign == 0) {
    return;
  }
#if defined(__AVX2__)
  if (backend() == Backend::AVX2) {
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

inline float fc_row_i8_f32(const std::int8_t *row, const float *in, int dim, float bias_scaled, float scale) {
  float sum = bias_scaled;
#if defined(__AVX2__)
  if (backend() == Backend::AVX2 && dim >= 8) {
    __m256 acc = _mm256_setzero_ps();
    const __m256 scalev = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
      __m128i w8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(row + i));
      __m256i wi = _mm256_cvtepi8_epi32(w8);
      __m256 wf = _mm256_mul_ps(_mm256_cvtepi32_ps(wi), scalev);
      __m256 xf = _mm256_loadu_ps(in + i);
      acc = _mm256_fmadd_ps(wf, xf, acc);
    }
    alignas(32) float parts[8];
    _mm256_storeu_ps(parts, acc);
    for (int k = 0; k < 8; ++k) {
      sum += parts[k];
    }
    for (; i < dim; ++i) {
      sum += static_cast<float>(row[i]) * scale * in[i];
    }
    return sum;
  }
#endif
  for (int i = 0; i < dim; ++i) {
    sum += static_cast<float>(row[i]) * scale * in[i];
  }
  return sum;
}

} // namespace simd
} // namespace xqwl_nnue

#endif // XQWL_NNUE_SIMD_H
