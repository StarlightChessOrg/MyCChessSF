// MyCChessSF INT8 NNUE inference (XQWL-PSQ features, mycchesssf_xqint8_v1 weights).
// Include after xqwl_extract.inc (needs PositionStruct).
#ifndef XQWL_NNUE_INC
#define XQWL_NNUE_INC

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace xqwl_nnue {

constexpr int kSquares = 90;
constexpr int kKinds = 7;
constexpr int kFeatures = kKinds * kSquares * 2;
constexpr int kEnemyOffset = kKinds * kSquares;
constexpr int kMaxActive = 40;

inline int coord_xy_iccs(int x, int y) { return (x + 3) + ((y + 3) << 4); }

inline int feature_index(int kind, int sq, bool friendly) {
  const int base = friendly ? 0 : kEnemyOffset;
  return base + kind * kSquares + sq;
}

inline int piece_kind(BYTE pc) {
  if (pc < 16) {
    return static_cast<int>(pc) - 8;
  }
  return static_cast<int>(pc) - 16;
}

inline void extract_features(const PositionStruct &pos, int *out, int &count) {
  count = 0;
  for (int y = 0; y < 10; ++y) {
    for (int x = 0; x < 9; ++x) {
      const int sq = coord_xy_iccs(x, y);
      const BYTE pc = pos.ucpcSquares[sq];
      if (!pc) {
        continue;
      }
      const int kind = piece_kind(pc);
      if (kind < 0 || kind > 6) {
        continue;
      }
      int view_y;
      bool friendly;
      if (pos.sdPlayer == 0) {
        view_y = y;
        friendly = pc < 16;
      } else {
        view_y = 9 - y;
        friendly = pc >= 16;
      }
      const int sq_idx = view_y * 9 + x;
      if (count < kMaxActive) {
        out[count++] = feature_index(kind, sq_idx, friendly);
      }
    }
  }
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
};

struct EvalWork {
  const Weights *w = nullptr;
  std::vector<float> acc;
  std::vector<float> h;
  std::vector<float> x0;
  std::vector<float> x0_cat;
  std::vector<float> x1;
  std::vector<float> x1_cat;

  explicit EvalWork(const Weights &weights) : w(&weights) {
    acc.assign(static_cast<size_t>(w->l1), 0.f);
    h.assign(static_cast<size_t>(w->l1), 0.f);
    x0.assign(static_cast<size_t>(w->l2), 0.f);
    x0_cat.assign(static_cast<size_t>(w->l2 * 2), 0.f);
    x1.assign(static_cast<size_t>(w->l3), 0.f);
    x1_cat.assign(static_cast<size_t>(w->l3 * 2), 0.f);
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

  void feature_transform() {
    const int l1 = w->l1;
    for (int i = 0; i < l1; ++i) {
      float v = acc[static_cast<size_t>(i)];
      if (v < 0.f) {
        v = 0.f;
      } else if (v > w->ft_clip) {
        v = w->ft_clip;
      }
      acc[static_cast<size_t>(i)] = v;
    }
    for (int i = 0; i < l1; i += 2) {
      const float prod = acc[static_cast<size_t>(i)] * acc[static_cast<size_t>(i + 1)] / 512.f;
      h[static_cast<size_t>(i)] = prod;
      h[static_cast<size_t>(i + 1)] = prod;
    }
  }

  void linear_fc(const std::vector<std::int8_t> &weight, const std::vector<std::int32_t> &bias,
                 const std::vector<float> &scales, const float *input, int in_dim, float *output,
                 int out_dim) {
    for (int o = 0; o < out_dim; ++o) {
      float sum = static_cast<float>(bias[static_cast<size_t>(o)]) * scales[static_cast<size_t>(o)];
      const float scale = scales[static_cast<size_t>(o)];
      const std::int8_t *row = weight.data() + static_cast<size_t>(o) * static_cast<size_t>(in_dim);
      for (int i = 0; i < in_dim; ++i) {
        sum += static_cast<float>(row[i]) * scale * input[i];
      }
      output[o] = sum;
    }
  }

  float forward(const PositionStruct &pos) {
    int feats[kMaxActive];
    int n_feat = 0;
    extract_features(pos, feats, n_feat);

    const int l1 = w->l1;
    for (int j = 0; j < l1; ++j) {
      acc[static_cast<size_t>(j)] =
          static_cast<float>(w->ft_b[static_cast<size_t>(j)]) * w->ft_scales[static_cast<size_t>(j)];
    }
    for (int fi = 0; fi < n_feat; ++fi) {
      const int idx = feats[fi];
      if (idx < 0 || idx >= w->n_features) {
        continue;
      }
      const std::int8_t *row = w->ft_w.data() + static_cast<size_t>(idx) * static_cast<size_t>(l1);
      for (int j = 0; j < l1; ++j) {
        acc[static_cast<size_t>(j)] += static_cast<float>(row[j]) * w->ft_scales[static_cast<size_t>(j)];
      }
    }

    feature_transform();
    linear_fc(w->fc0_w, w->fc0_b, w->fc0_scales, h.data(), l1, x0.data(), w->l2);
    for (int i = 0; i < w->l2; ++i) {
      x0_cat[static_cast<size_t>(i)] = sq_clipped_relu(x0[static_cast<size_t>(i)], w->act_clip);
      x0_cat[static_cast<size_t>(w->l2 + i)] = clipped_relu(x0[static_cast<size_t>(i)], w->act_clip);
    }
    linear_fc(w->fc1_w, w->fc1_b, w->fc1_scales, x0_cat.data(), w->l2 * 2, x1.data(), w->l3);
    for (int i = 0; i < w->l3; ++i) {
      x1_cat[static_cast<size_t>(i)] = sq_clipped_relu(x1[static_cast<size_t>(i)], w->act_clip);
      x1_cat[static_cast<size_t>(w->l3 + i)] = clipped_relu(x1[static_cast<size_t>(i)], w->act_clip);
    }
    float out = 0.f;
    linear_fc(w->fc2_w, w->fc2_b, w->fc2_scales, x1_cat.data(), w->l3 * 2, &out, 1);
    return out;
  }
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
              read_exact(f, &version, sizeof(version)) && version == 1 &&
              read_exact(f, &label_mean, sizeof(label_mean)) && read_exact(f, &label_std, sizeof(label_std)) &&
              read_exact(f, &n_features, sizeof(n_features)) && read_exact(f, &l1, sizeof(l1)) &&
              read_exact(f, &l2, sizeof(l2)) && read_exact(f, &l3, sizeof(l3)) &&
              read_exact(f, &ft_clip, sizeof(ft_clip)) && read_exact(f, &act_clip, sizeof(act_clip));

    if (!ok || n_features != static_cast<std::uint32_t>(kFeatures) || l1 == 0 || l2 == 0 || l3 == 0) {
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
    return true;
  }

  int evaluate_vl(const PositionStruct &pos) const {
    if (!loaded) {
      return 0;
    }
    const double pred_norm = static_cast<double>(EvalWork(w).forward(pos));
    return static_cast<int>(std::lround(pred_norm * w.label_std + w.label_mean));
  }
};

} // namespace xqwl_nnue

#endif // XQWL_NNUE_INC
