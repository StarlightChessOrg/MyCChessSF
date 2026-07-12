// MyCChessSF INT8 NNUE: dual-perspective FT accumulators, SIMD, mate-aware scoring.
//
// Dual accumulator (NOT dual scoring):
//   acc[0] / acc[1] hold FT int32 sums under virtual_stm=0 (red) / 1 (black).
//   Feature indices depend on perspective; after MakeMove sdPlayer flips, so we
//   keep both FT states incrementally and evaluate with acc[pos.sdPlayer] only.
//   FC forward runs once per eval — never two final scores.
#ifndef XQWL_NNUE_INC
#define XQWL_NNUE_INC

#include "xqwl_halfka.inc"
#include "xqwl_nnue_simd.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace xqwl_nnue {

constexpr int kSquares = 90;
constexpr int kFeatures = xqwl_halfka::kFeatures;
constexpr int kMaxActive = xqwl_halfka::kMaxActive;
constexpr int kMaxL1 = 512;
constexpr int kMaxDelta = 4;

inline int coord_xy_iccs(int x, int y) { return xqwl_halfka::coord_xy_iccs(x, y); }

// Mate / extreme score handling (XQWL centipawn scale).
constexpr int kMateZonePst = WIN_VALUE - 800;
constexpr int kNnueHardClamp = 3200;
constexpr int kNnueSoftClamp = 2400;
constexpr int kTacticalBlendPst = 1500;

inline void extract_features_perspective(const PositionStruct &pos, int virtual_stm, int *out, int &count) {
  xqwl_halfka::extract_features_perspective(pos, virtual_stm, out, count);
}

struct Weights {
  int n_features = 0;
  int l1 = 0;
  int l2 = 0;
  int l3 = 0;
  float ft_clip = 256.f;
  float act_clip = 1.f;
  double label_mean = 0.0;
  double label_std = 1.0;

  std::vector<std::int8_t> ft_w;
  std::vector<std::int32_t> ft_b;
  std::vector<float> ft_scales;

  std::vector<std::int8_t> fc0_w;
  std::vector<std::int32_t> fc0_b;
  std::vector<float> fc0_scales;

  std::vector<std::int8_t> fc1_w;
  std::vector<std::int32_t> fc1_b;
  std::vector<float> fc1_scales;

  std::vector<std::int8_t> fc2_w;
  std::vector<std::int32_t> fc2_b;
  std::vector<float> fc2_scales;

  float fc0_input_scale = 1.f;
  float fc1_input_scale = 1.f;
  float fc2_input_scale = 1.f;
  std::uint32_t format_version = 1;
};

struct AccDelta {
  bool refresh[2]{};
  std::int8_t n_rem[2]{};
  std::int8_t n_add[2]{};
  std::int16_t rem[2][kMaxDelta]{};
  std::int16_t add[2][kMaxDelta]{};
};

struct SearchState {
  alignas(64) std::int32_t acc[2][kMaxL1]{};
  AccDelta stack[LIMIT_DEPTH + 16];
  int stack_top = 0;
  int l1 = 0;
  bool ready = false;
};

struct Runtime {
  Weights w;
  bool loaded = false;

  static bool read_exact(FILE *f, void *buf, size_t n) {
    return std::fread(buf, 1, n, f) == n;
  }

  static bool read_tensor_i8(FILE *f, std::vector<std::int8_t> &out, size_t n) {
    out.resize(n);
    return read_exact(f, out.data(), n);
  }

  static bool read_tensor_i32(FILE *f, std::vector<std::int32_t> &out, size_t n) {
    out.resize(n);
    return read_exact(f, out.data(), n * sizeof(std::int32_t));
  }

  static bool read_tensor_f32(FILE *f, std::vector<float> &out, size_t n) {
    out.resize(n);
    return read_exact(f, out.data(), n * sizeof(float));
  }

  bool load(const char *path) {
    loaded = false;
    if (path == nullptr || path[0] == '\0') {
      return false;
    }
    FILE *f = std::fopen(path, "rb");
    if (f == nullptr) {
      return false;
    }

    char magic[8];
    std::uint32_t version = 0;
    double label_mean = 0.0;
    double label_std = 1.0;
    std::uint32_t n_features = 0;
    std::uint32_t l1 = 0;
    std::uint32_t l2 = 0;
    std::uint32_t l3 = 0;
    float ft_clip = 256.f;
    float act_clip = 1.f;

    bool ok = read_exact(f, magic, 8) && std::memcmp(magic, "XQNNUE01", 8) == 0 &&
              read_exact(f, &version, sizeof(version)) && (version == 1 || version == 2) &&
              read_exact(f, &label_mean, sizeof(label_mean)) && read_exact(f, &label_std, sizeof(label_std)) &&
              read_exact(f, &n_features, sizeof(n_features)) && read_exact(f, &l1, sizeof(l1)) &&
              read_exact(f, &l2, sizeof(l2)) && read_exact(f, &l3, sizeof(l3)) &&
              read_exact(f, &ft_clip, sizeof(ft_clip)) && read_exact(f, &act_clip, sizeof(act_clip));

    float fc0_in = 0.f;
    float fc1_in = 0.f;
    float fc2_in = 0.f;
    if (ok && version >= 2) {
      ok = read_exact(f, &fc0_in, sizeof(fc0_in)) && read_exact(f, &fc1_in, sizeof(fc1_in)) &&
           read_exact(f, &fc2_in, sizeof(fc2_in));
    }

    if (!ok || n_features != static_cast<std::uint32_t>(kFeatures) || l1 == 0 || l2 == 0 || l3 == 0 ||
        l1 > static_cast<std::uint32_t>(kMaxL1)) {
      std::fclose(f);
      return false;
    }

    Weights nw;
    nw.n_features = static_cast<int>(n_features);
    nw.l1 = static_cast<int>(l1);
    nw.l2 = static_cast<int>(l2);
    nw.l3 = static_cast<int>(l3);
    nw.ft_clip = ft_clip;
    nw.act_clip = act_clip;
    nw.label_mean = label_mean;
    nw.label_std = label_std;
    nw.format_version = version;
    if (version >= 2) {
      nw.fc0_input_scale = fc0_in;
      nw.fc1_input_scale = fc1_in;
      nw.fc2_input_scale = fc2_in;
    } else {
      nw.fc0_input_scale = std::max((ft_clip * ft_clip / 512.f) / 127.f, 1e-6f);
      nw.fc1_input_scale = std::max(act_clip / 127.f, 1e-6f);
      nw.fc2_input_scale = std::max(act_clip / 127.f, 1e-6f);
    }

    ok = read_tensor_i8(f, nw.ft_w, static_cast<size_t>(nw.n_features) * static_cast<size_t>(nw.l1)) &&
         read_tensor_i32(f, nw.ft_b, static_cast<size_t>(nw.l1)) &&
         read_tensor_f32(f, nw.ft_scales, static_cast<size_t>(nw.l1)) &&
         read_tensor_i8(f, nw.fc0_w, static_cast<size_t>(nw.l2) * static_cast<size_t>(nw.l1)) &&
         read_tensor_i32(f, nw.fc0_b, static_cast<size_t>(nw.l2)) &&
         read_tensor_f32(f, nw.fc0_scales, static_cast<size_t>(nw.l2)) &&
         read_tensor_i8(f, nw.fc1_w, static_cast<size_t>(nw.l3) * static_cast<size_t>(nw.l2 * 2)) &&
         read_tensor_i32(f, nw.fc1_b, static_cast<size_t>(nw.l3)) &&
         read_tensor_f32(f, nw.fc1_scales, static_cast<size_t>(nw.l3)) &&
         read_tensor_i8(f, nw.fc2_w, static_cast<size_t>(nw.l3 * 2)) &&
         read_tensor_i32(f, nw.fc2_b, 1) && read_tensor_f32(f, nw.fc2_scales, 1);

    std::fclose(f);
    if (!ok) {
      return false;
    }

    w = std::move(nw);
    loaded = true;
    simd::init();
    return true;
  }

  void acc_refresh_perspective(SearchState &st, int persp, const PositionStruct &pos) const {
    const int l1 = w.l1;
    st.l1 = l1;
    for (int j = 0; j < l1; ++j) {
      st.acc[persp][j] = w.ft_b[static_cast<size_t>(j)];
    }
    int feats[kMaxActive];
    int n_feat = 0;
    extract_features_perspective(pos, persp, feats, n_feat);
    for (int fi = 0; fi < n_feat; ++fi) {
      const int idx = feats[fi];
      if (idx < 0 || idx >= w.n_features) {
        continue;
      }
      simd::acc_add_column(st.acc[persp], w.ft_w.data() + static_cast<size_t>(idx) * static_cast<size_t>(l1), l1, +1);
    }
  }

  void acc_init(SearchState &st, const PositionStruct &pos) const {
    st.stack_top = 0;
    st.l1 = w.l1;
    // Both FT views: cheap incremental updates on MakeMove; eval uses acc[sdPlayer] only.
    acc_refresh_perspective(st, 0, pos);
    acc_refresh_perspective(st, 1, pos);
    st.ready = true;
  }

  void acc_apply_column_delta(SearchState &st, int persp, const std::int16_t *idx, int n, int sign) const {
    const int l1 = w.l1;
    for (int i = 0; i < n; ++i) {
      const int f = idx[i];
      if (f < 0 || f >= w.n_features) {
        continue;
      }
      simd::acc_add_column(st.acc[persp], w.ft_w.data() + static_cast<size_t>(f) * static_cast<size_t>(l1), l1, sign);
    }
  }

  void acc_apply_move(SearchState &st, const PositionStruct &pos, int mv) const {
    if (!st.ready || st.stack_top >= LIMIT_DEPTH + 15) {
      return;
    }
    AccDelta &d = st.stack[st.stack_top++];

    const int sq_src = mv & 255;
    const int sq_dst = mv >> 8;
    const BYTE pc = pos.ucpcSquares[sq_src];
    const BYTE cap = pos.ucpcSquares[sq_dst];

    PositionStruct after = pos;
    if (!after.MakeMove(mv)) {
      --st.stack_top;
      return;
    }

    for (int persp = 0; persp < 2; ++persp) {
      d.refresh[persp] = false;
      d.n_rem[persp] = 0;
      d.n_add[persp] = 0;

      if (!xqwl_halfka::feature_bucket_equal(pos, after, persp)) {
        d.refresh[persp] = true;
        acc_refresh_perspective(st, persp, after);
        continue;
      }

      const xqwl_halfka::FeatureBucket fb = xqwl_halfka::make_feature_bucket(pos, persp);
      const int f_src = xqwl_halfka::feature_from_iccs(sq_src, pc, persp, fb.bucket, fb.mirror);
      if (f_src >= 0) {
        d.rem[persp][d.n_rem[persp]++] = static_cast<std::int16_t>(f_src);
      }
      if (cap) {
        const int f_cap = xqwl_halfka::feature_from_iccs(sq_dst, cap, persp, fb.bucket, fb.mirror);
        if (f_cap >= 0) {
          d.rem[persp][d.n_rem[persp]++] = static_cast<std::int16_t>(f_cap);
        }
      }
      const int f_dst = xqwl_halfka::feature_from_iccs(sq_dst, pc, persp, fb.bucket, fb.mirror);
      if (f_dst >= 0) {
        d.add[persp][d.n_add[persp]++] = static_cast<std::int16_t>(f_dst);
      }
      acc_apply_column_delta(st, persp, d.rem[persp], d.n_rem[persp], -1);
      acc_apply_column_delta(st, persp, d.add[persp], d.n_add[persp], +1);
    }
  }

  void acc_undo(SearchState &st, const PositionStruct &pos) const {
    if (!st.ready || st.stack_top <= 0) {
      return;
    }
    AccDelta &d = st.stack[--st.stack_top];
    for (int persp = 0; persp < 2; ++persp) {
      if (d.refresh[persp]) {
        acc_refresh_perspective(st, persp, pos);
        continue;
      }
      acc_apply_column_delta(st, persp, d.add[persp], d.n_add[persp], -1);
      acc_apply_column_delta(st, persp, d.rem[persp], d.n_rem[persp], +1);
    }
  }

  static float clipped_relu(float x, float max_val) {
    if (x <= 0.f) {
      return 0.f;
    }
    return x > max_val ? max_val : x;
  }

  static float sq_clipped_relu(float x, float max_val) {
    const float sq = x * x;
    return sq > max_val ? max_val : sq;
  }

  static std::uint8_t quantize_input(float v, float input_scale) {
    if (v <= 0.f || input_scale <= 0.f) {
      return 0;
    }
    const int q = static_cast<int>(std::lround(v / input_scale));
    if (q >= 127) {
      return 127;
    }
    return static_cast<std::uint8_t>(q);
  }

  void quantize_inputs(const float *in, int dim, float input_scale, std::uint8_t *out) const {
    for (int i = 0; i < dim; ++i) {
      out[i] = quantize_input(in[i], input_scale);
    }
  }

  void fc_forward_int8(const std::int8_t *w, const std::int32_t *bias, const float *scales, int out_dim, int in_dim,
                       const float *in, float input_scale, float *out) const {
    std::uint8_t in_q[kMaxL1];
    quantize_inputs(in, in_dim, input_scale, in_q);
    for (int o = 0; o < out_dim; ++o) {
      const std::int8_t *row = w + static_cast<size_t>(o) * static_cast<size_t>(in_dim);
      const int32_t dot = simd::dot_i8_u8(row, in_q, in_dim);
      const float scale = scales[o];
      out[o] = scale * static_cast<float>(bias[o]) + scale * static_cast<float>(dot) * input_scale;
    }
  }

  float forward_norm_from_acc(const SearchState &st, int persp) const {
    const int l1 = w.l1;
    const int l2 = w.l2;
    const int l3 = w.l3;

    float h[kMaxL1];
    for (int j = 0; j < l1; ++j) {
      float v = static_cast<float>(st.acc[persp][j]) * w.ft_scales[static_cast<size_t>(j)];
      if (v < 0.f) {
        v = 0.f;
      } else if (v > w.ft_clip) {
        v = w.ft_clip;
      }
      h[j] = v;
    }
    for (int i = 0; i < l1; i += 2) {
      const float prod = h[i] * h[i + 1] / 512.f;
      h[i] = prod;
      h[i + 1] = prod;
    }

    float x0[64];
    float x0_cat[128];
    fc_forward_int8(w.fc0_w.data(), w.fc0_b.data(), w.fc0_scales.data(), l2, l1, h, w.fc0_input_scale, x0);
    for (int i = 0; i < l2; ++i) {
      x0_cat[i] = sq_clipped_relu(x0[i], w.act_clip);
      x0_cat[l2 + i] = clipped_relu(x0[i], w.act_clip);
    }

    float x1[64];
    float x1_cat[128];
    fc_forward_int8(w.fc1_w.data(), w.fc1_b.data(), w.fc1_scales.data(), l3, l2 * 2, x0_cat, w.fc1_input_scale, x1);
    for (int i = 0; i < l3; ++i) {
      x1_cat[i] = sq_clipped_relu(x1[i], w.act_clip);
      x1_cat[l3 + i] = clipped_relu(x1[i], w.act_clip);
    }

    float x2[1];
    fc_forward_int8(w.fc2_w.data(), w.fc2_b.data(), w.fc2_scales.data(), 1, l3 * 2, x1_cat, w.fc2_input_scale, x2);
    return x2[0];
  }

  float forward_norm_full(const PositionStruct &pos) const {
    SearchState tmp{};
    acc_init(tmp, pos);
    return forward_norm_from_acc(tmp, pos.sdPlayer);
  }

  static int compress_extreme_vl(int vl) {
    if (vl > kNnueHardClamp) {
      return kNnueHardClamp + (vl - kNnueHardClamp) / 4;
    }
    if (vl < -kNnueHardClamp) {
      return -kNnueHardClamp + (vl + kNnueHardClamp) / 4;
    }
    return vl;
  }

  int finalize_vl(int nnue_vl, const PositionStruct &pos) const {
    const int pst = pos.Evaluate();

    if (std::abs(pst) >= kMateZonePst) {
      return pst;
    }

    int vl = compress_extreme_vl(nnue_vl);

    if (pos.InCheck()) {
      vl = (vl * 3 + pst) / 4;
    }

    if (std::abs(pst) >= kTacticalBlendPst && std::abs(pst - vl) > 2000) {
      vl = (vl + pst * 2) / 3;
    }

    if (vl > kNnueSoftClamp) {
      vl = kNnueSoftClamp + (vl - kNnueSoftClamp) * 2 / 3;
    } else if (vl < -kNnueSoftClamp) {
      vl = -kNnueSoftClamp + (vl + kNnueSoftClamp) * 2 / 3;
    }

    return vl;
  }

  int evaluate_vl(const PositionStruct &pos, SearchState *st) const {
    if (!loaded) {
      return 0;
    }
    float norm = 0.f;
    if (st != nullptr && st->ready) {
      norm = forward_norm_from_acc(*st, pos.sdPlayer);
    } else {
      norm = forward_norm_full(pos);
    }
    const int raw = static_cast<int>(std::lround(norm * w.label_std + w.label_mean));
    return finalize_vl(raw, pos);
  }

  int evaluate_vl_raw(const PositionStruct &pos, SearchState *st) const {
    if (!loaded) {
      return 0;
    }
    float norm = 0.f;
    if (st != nullptr && st->ready) {
      norm = forward_norm_from_acc(*st, pos.sdPlayer);
    } else {
      norm = forward_norm_full(pos);
    }
    return static_cast<int>(std::lround(norm * w.label_std + w.label_mean));
  }

  /** Max |incremental - full| over root and one-ply legal positions (diagnostics). */
  int verify_incremental(const PositionStruct &pos) const {
    if (!loaded) {
      return 0;
    }
    int max_diff = 0;
    SearchState st{};
    acc_init(st, pos);
    max_diff = std::max(max_diff, std::abs(evaluate_vl_raw(pos, &st) - evaluate_vl_raw(pos, nullptr)));

    int mvs[MAX_GEN_MOVES];
    const int n_gen = pos.GenerateMoves(mvs, FALSE);
    for (int i = 0; i < n_gen; ++i) {
      PositionStruct trial = pos;
      SearchState st_move = st;
      acc_apply_move(st_move, pos, mvs[i]);
      if (!trial.MakeMove(mvs[i])) {
        continue;
      }
      const int inc = evaluate_vl_raw(trial, &st_move);
      const int full = evaluate_vl_raw(trial, nullptr);
      max_diff = std::max(max_diff, std::abs(inc - full));
    }
    return max_diff;
  }
};

} // namespace xqwl_nnue

#endif // XQWL_NNUE_INC
