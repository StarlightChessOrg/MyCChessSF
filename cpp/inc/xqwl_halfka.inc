// HalfKAv2_hm feature extraction for MyCChessSF NNUE (Pikafish xiangqi).
#ifndef XQWL_HALFKA_INC
#define XQWL_HALFKA_INC

#include "xqwl_halfka_tables.inc"

#include <cstdint>

namespace xqwl_halfka {

inline int iccs_x_from_sq(int sq) { return (sq & 15) - 3; }
inline int iccs_y_from_sq(int sq) { return (sq >> 4) - 3; }
inline int coord_xy_iccs(int x, int y) { return (x + 3) + ((y + 3) << 4); }

inline int iccs_to_pikafish_sq(int sq) {
  const int x = iccs_x_from_sq(sq);
  const int y = iccs_y_from_sq(sq);
  return (9 - y) * 9 + x;
}

inline int halfka_piece(BYTE pc) {
  if (!pc) {
    return 0;
  }
  const int t = (pc < 16) ? static_cast<int>(pc) - 8 : static_cast<int>(pc) - 16;
  if (t < 0 || t > 6) {
    return 0;
  }
  static const int kMap[7] = {7, 2, 6, 5, 1, 3, 4};
  const int color_base = (pc < 16) ? 0 : 8;
  return kMap[t] + color_base;
}

inline bool valid_bb_test(int hk_pc, int psq) {
  if (hk_pc <= 0 || hk_pc >= 16 || psq < 0 || psq >= 90) {
    return false;
  }
  const auto &bb = kValidBB[hk_pc];
  if (psq < 64) {
    return (bb.lo >> psq) & 1ULL;
  }
  return (bb.hi >> (psq - 64)) & 1ULL;
}

inline int king_square(const PositionStruct &pos, int color) {
  const int want = (color == 0) ? 7 : 15;
  for (int y = 0; y < 10; ++y) {
    for (int x = 0; x < 9; ++x) {
      const int sq = coord_xy_iccs(x, y);
      const int hk = halfka_piece(pos.ucpcSquares[sq]);
      if (hk == want) {
        return iccs_to_pikafish_sq(sq);
      }
    }
  }
  return 0;
}

inline std::uint64_t mid_encoding(const PositionStruct &pos, int color) {
  std::uint64_t enc = kBalanceEncoding;
  for (int y = 0; y < 10; ++y) {
    for (int x = 0; x < 9; ++x) {
      const int sq = coord_xy_iccs(x, y);
      const BYTE pc = pos.ucpcSquares[sq];
      const int hk = halfka_piece(pc);
      if (!hk) {
        continue;
      }
      const int pc_color = (pc < 16) ? 0 : 1;
      if (pc_color != color) {
        continue;
      }
      const int psq = iccs_to_pikafish_sq(sq);
      enc += kMidMirrorEncoding[hk][psq];
    }
  }
  return enc;
}

inline bool requires_mid_mirror(const PositionStruct &pos, int color) {
  const std::uint64_t enc_us = mid_encoding(pos, color);
  const std::uint64_t enc_them = mid_encoding(pos, color ^ 1);
  return ((1ULL << 63) & enc_us & enc_them) &&
         (enc_us < kBalanceEncoding ||
          (enc_us == kBalanceEncoding && enc_them < kBalanceEncoding));
}

inline int count_pt(const PositionStruct &pos, int color, int pt) {
  int n = 0;
  for (int y = 0; y < 10; ++y) {
    for (int x = 0; x < 9; ++x) {
      const int sq = coord_xy_iccs(x, y);
      const BYTE pc = pos.ucpcSquares[sq];
      const int hk = halfka_piece(pc);
      if (!hk) {
        continue;
      }
      const int pc_color = (pc < 16) ? 0 : 1;
      if (pc_color == color && (hk & 7) == pt) {
        ++n;
      }
    }
  }
  return n > 2 ? 2 : n;
}

inline int make_attack_bucket(const PositionStruct &pos, int color) {
  return kAttackBucket[count_pt(pos, color, 1)][count_pt(pos, color, 5)][count_pt(pos, color, 3)];
}

struct FeatureBucket {
  int bucket = 0;
  bool mirror = false;
};

inline FeatureBucket make_feature_bucket(const PositionStruct &pos, int perspective) {
  const int ksq = king_square(pos, perspective);
  const int oksq = king_square(pos, perspective ^ 1);
  const int midm = requires_mid_mirror(pos, perspective) ? 1 : 0;
  const std::uint8_t kb = kKingBucket[ksq][oksq][midm];
  const int king_bucket = kb & 0x7;
  const bool mirror = (kb >> 7) != 0;
  const int attack_bucket = make_attack_bucket(pos, perspective);
  FeatureBucket out;
  out.bucket = king_bucket * kAttackBucketNb + attack_bucket;
  out.mirror = mirror;
  return out;
}

inline int make_index(int perspective, int psq, int hk_pc, int bucket, bool mirror) {
  int mapped = kIndexMap[mirror ? 1 : 0][perspective == 1 ? 1 : 0][psq];
  int pc = hk_pc;
  if (perspective == 1) {
    pc ^= 8;
  }
  const std::int16_t off = kPsqOffsets[pc][mapped];
  if (off < 0) {
    return -1;
  }
  return static_cast<int>(off) + kPsNb * bucket;
}

inline int feature_from_iccs(int iccs_sq, BYTE pc, int perspective, int bucket, bool mirror) {
  const int hk = halfka_piece(pc);
  if (!hk) {
    return -1;
  }
  const int psq = iccs_to_pikafish_sq(iccs_sq);
  if (!valid_bb_test(hk, psq)) {
    return -1;
  }
  return make_index(perspective, psq, hk, bucket, mirror);
}

inline void extract_features_perspective(const PositionStruct &pos, int perspective, int *out, int &count) {
  count = 0;
  const FeatureBucket fb = make_feature_bucket(pos, perspective);
  for (int y = 0; y < 10; ++y) {
    for (int x = 0; x < 9; ++x) {
      const int sq = coord_xy_iccs(x, y);
      const BYTE pc = pos.ucpcSquares[sq];
      if (!pc) {
        continue;
      }
      const int idx = feature_from_iccs(sq, pc, perspective, fb.bucket, fb.mirror);
      if (idx >= 0 && count < kMaxActive) {
        out[count++] = idx;
      }
    }
  }
}

inline bool feature_bucket_equal(const PositionStruct &before, const PositionStruct &after, int perspective) {
  const FeatureBucket a = make_feature_bucket(before, perspective);
  const FeatureBucket b = make_feature_bucket(after, perspective);
  return a.bucket == b.bucket && a.mirror == b.mirror;
}

} // namespace xqwl_halfka

#endif // XQWL_HALFKA_INC
