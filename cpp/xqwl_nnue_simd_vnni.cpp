// AVX512-VNNI FC dot product (compiled with -mavx512f -mavx512vnni -mavx512vl).
#include "xqwl_nnue_simd_vnni.h"

#include <cstdint>
#include <immintrin.h>

namespace xqwl_nnue {
namespace simd {

int32_t dot_i8_u8_avx512_vnni(const std::int8_t *w, const std::uint8_t *x, int dim) {
  __m512i acc = _mm512_setzero_si512();
  int i = 0;
  for (; i + 64 <= dim; i += 64) {
    const __m512i xv = _mm512_loadu_si512(reinterpret_cast<const void *>(x + i));
    const __m512i wv = _mm512_loadu_si512(reinterpret_cast<const void *>(w + i));
    acc = _mm512_dpbusd_epi32(acc, xv, wv);
  }
  alignas(64) int32_t parts[16];
  _mm512_storeu_si512(parts, acc);
  int32_t sum = 0;
  for (int k = 0; k < 16; ++k) {
    sum += parts[k];
  }
  for (; i < dim; ++i) {
    sum += static_cast<int32_t>(w[i]) * static_cast<int32_t>(x[i]);
  }
  return sum;
}

} // namespace simd
} // namespace xqwl_nnue
