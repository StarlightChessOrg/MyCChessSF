// Parallel NNUE training data generator: FEN + root search score (side-to-move perspective).
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define XQWL_NO_BOOK_LOADER
#include "xqwl_portable_prefix.h"
#include "xqwl_portable.cpp"
#include "xqwl_fen.inc"

namespace {

constexpr int kIoBufferBytes = 8192;
constexpr long long kFlushEveryPositions = 500;
constexpr long long kProgressEveryPositions = 10000;
constexpr long long kDefaultMaxPositions = 10'000'000LL;

struct GenConfig {
  std::string output_dir = "nnue_data";
  long long max_positions = kDefaultMaxPositions;
  int think_ms = 100;
  int jobs = 0;
  int random_pct = 20;
};

void usage(const char *prog) {
  std::fprintf(stderr,
               "Usage: %s [options]\n"
               "  --output-dir DIR       Output directory (default: nnue_data)\n"
               "  --max-positions N      Total position cap (default: 10000000)\n"
               "  --think-ms MS          Search time per position (default: 100)\n"
               "  --jobs N               Parallel workers (0 = CPU cores, default: 0)\n"
               "  --random-pct PCT       Random move probability 0-100 (default: 20)\n",
               prog);
}

bool parse_i64(const char *s, long long &out) {
  if (s == nullptr || *s == '\0')
    return false;
  char *end = nullptr;
  errno = 0;
  const long long v = std::strtoll(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0')
    return false;
  out = v;
  return true;
}

bool parse_i32(const char *s, int &out) {
  long long v = 0;
  if (!parse_i64(s, v) || v < 0 || v > 0x7fffffff)
    return false;
  out = static_cast<int>(v);
  return true;
}

bool parse_args(int argc, char **argv, GenConfig &cfg) {
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    auto need = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", name);
        return nullptr;
      }
      return argv[++i];
    };
    if (std::strcmp(arg, "--output-dir") == 0) {
      const char *v = need(arg);
      if (v == nullptr)
        return false;
      cfg.output_dir = v;
    } else if (std::strcmp(arg, "--max-positions") == 0) {
      const char *v = need(arg);
      if (v == nullptr || !parse_i64(v, cfg.max_positions) || cfg.max_positions <= 0) {
        std::fprintf(stderr, "Invalid --max-positions\n");
        return false;
      }
    } else if (std::strcmp(arg, "--think-ms") == 0) {
      const char *v = need(arg);
      if (v == nullptr || !parse_i32(v, cfg.think_ms) || cfg.think_ms < 1) {
        std::fprintf(stderr, "Invalid --think-ms\n");
        return false;
      }
    } else if (std::strcmp(arg, "--jobs") == 0) {
      const char *v = need(arg);
      if (v == nullptr || !parse_i32(v, cfg.jobs) || cfg.jobs < 0) {
        std::fprintf(stderr, "Invalid --jobs\n");
        return false;
      }
    } else if (std::strcmp(arg, "--random-pct") == 0) {
      const char *v = need(arg);
      if (v == nullptr || !parse_i32(v, cfg.random_pct) || cfg.random_pct < 0 || cfg.random_pct > 100) {
        std::fprintf(stderr, "Invalid --random-pct (0-100)\n");
        return false;
      }
    } else if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", arg);
      return false;
    }
  }
  return true;
}

bool ensure_dir(const std::string &path) {
  if (path.empty())
    return false;
  if (::mkdir(path.c_str(), 0755) == 0)
    return true;
  if (errno == EEXIST)
    return true;
  std::perror(path.c_str());
  return false;
}

int terminal_kind(PositionStruct &pos) {
  if (pos.IsMate() != 0)
    return 1;
  if (pos.RepStatus(3) > 0)
    return 2;
  if (pos.nMoveNum > 100)
    return 3;
  return 0;
}

int collect_legal_moves(const PositionStruct &pos, int *out, int cap) {
  int mvs[MAX_GEN_MOVES];
  const int n = pos.GenerateMoves(mvs, FALSE);
  int nLegal = 0;
  const int limit = n < cap ? n : cap;
  for (int i = 0; i < limit; ++i) {
    PositionStruct trial = pos;
    if (trial.MakeMove(mvs[i]) && nLegal < cap) {
      out[nLegal++] = mvs[i];
    }
  }
  return nLegal;
}

int pick_random_move(PositionStruct &pos) {
  int legal[MAX_GEN_MOVES];
  const int nLegal = collect_legal_moves(pos, legal, MAX_GEN_MOVES);
  if (nLegal <= 0)
    return 0;
  return legal[std::rand() % nLegal];
}

void worker_main(int worker_id, long long quota, const GenConfig &cfg) {
  InitZobrist();
  std::srand(static_cast<unsigned>(std::time(nullptr)) ^ static_cast<unsigned>(worker_id * 7919 + 1));

  const std::string out_path = cfg.output_dir + "/worker_" + std::to_string(worker_id) + ".txt";
  FILE *fp = std::fopen(out_path.c_str(), "wb");
  if (fp == nullptr) {
    std::perror(out_path.c_str());
    std::exit(1);
  }
  std::vector<char> io_buf(kIoBufferBytes);
  setvbuf(fp, io_buf.data(), _IOLBF, io_buf.size());
  std::fprintf(stderr, "[worker %d] started -> %s (quota %lld)\n", worker_id, out_path.c_str(),
               static_cast<long long>(quota));
  std::fflush(stderr);

  // Hash table is ~16 MiB; keep off the stack (default thread stack is often 8 MiB).
  auto tab = std::make_unique<XqwlSearchTables>();
  *tab = XqwlSearchTables{};
  PositionStruct pos;
  long long written = 0;

  while (written < quota) {
    pos.Startup();
    for (;;) {
      const std::string fen = xqwl_position_to_fen(pos);
      PositionStruct search_pos = pos;
      const XqwlSearchOutcome search =
          XqwlSearchBestMoveEx(search_pos, *tab, cfg.think_ms, /*use_book=*/false);

      std::fprintf(fp, "%s\t%d\n", fen.c_str(), search.score);
      ++written;
      if (written % kFlushEveryPositions == 0) {
        std::fflush(fp);
      }
      if (written % kProgressEveryPositions == 0) {
        std::fprintf(stderr, "[worker %d] progress %lld / %lld\n", worker_id,
                     static_cast<long long>(written), static_cast<long long>(quota));
        std::fflush(stderr);
      }
      if (written >= quota)
        break;

      int mv = 0;
      if (cfg.random_pct > 0 && (std::rand() % 100) < cfg.random_pct) {
        mv = pick_random_move(pos);
      } else {
        mv = search.mv;
      }
      if (mv == 0)
        mv = pick_random_move(pos);
      if (mv == 0 || !pos.MakeMove(mv))
        break;
      if (terminal_kind(pos) != 0)
        break;
    }
  }

  std::fclose(fp);
  std::fprintf(stderr, "[worker %d] wrote %lld positions -> %s\n", worker_id,
               static_cast<long long>(written), out_path.c_str());
}

} // namespace

int main(int argc, char **argv) {
  GenConfig cfg;
  if (!parse_args(argc, argv, cfg)) {
    usage(argv[0]);
    return 1;
  }
  if (!ensure_dir(cfg.output_dir)) {
    return 1;
  }

  int jobs = cfg.jobs;
  if (jobs <= 0) {
    jobs = static_cast<int>(std::thread::hardware_concurrency());
    if (jobs <= 0)
      jobs = 1;
  }

  const long long base = cfg.max_positions / jobs;
  const long long rem = cfg.max_positions % jobs;
  std::fprintf(stderr,
               "[xqwl_gen_nnue] output=%s positions=%lld jobs=%d think_ms=%d random_pct=%d%%\n",
               cfg.output_dir.c_str(), cfg.max_positions, jobs, cfg.think_ms, cfg.random_pct);

  std::vector<pid_t> children;
  children.reserve(static_cast<size_t>(jobs));
  for (int i = 0; i < jobs; ++i) {
    const long long quota = base + (i < static_cast<int>(rem) ? 1 : 0);
    const pid_t pid = ::fork();
    if (pid < 0) {
      std::perror("fork");
      return 1;
    }
    if (pid == 0) {
      worker_main(i, quota, cfg);
      std::exit(0);
    }
    children.push_back(pid);
  }

  int failed = 0;
  for (size_t i = 0; i < children.size(); ++i) {
    int status = 0;
    const pid_t pid = children[i];
    if (::waitpid(pid, &status, 0) < 0) {
      std::perror("waitpid");
      failed = 1;
      continue;
    }
    if (WIFSIGNALED(status)) {
      std::fprintf(stderr, "[xqwl_gen_nnue] worker %zu (pid %d) killed by signal %d\n", i, static_cast<int>(pid),
                   WTERMSIG(status));
      failed = 1;
    } else if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      std::fprintf(stderr, "[xqwl_gen_nnue] worker %zu (pid %d) exited abnormally (status=0x%x)\n", i,
                   static_cast<int>(pid), status);
      failed = 1;
    }
  }
  if (failed) {
    std::fprintf(stderr, "[xqwl_gen_nnue] one or more workers failed\n");
  } else {
    std::fprintf(stderr, "[xqwl_gen_nnue] all workers finished\n");
  }
  return failed ? 1 : 0;
}
