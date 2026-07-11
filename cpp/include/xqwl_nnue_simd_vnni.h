#ifndef XQWL_NNUE_SIMD_VNNI_H
#define XQWL_NNUE_SIMD_VNNI_H

#include <cstdint>

namespace xqwl_nnue {
namespace simd {

int32_t dot_i8_u8_avx512_vnni(const std::int8_t *w, const std::uint8_t *x, int dim);

} // namespace simd
} // namespace xqwl_nnue

#endif
