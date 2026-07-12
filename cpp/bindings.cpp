#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>
#include <vector>

#include "xqwl_portable_prefix.h"
#include "xqwl_portable.cpp"
#include "xqwl_fen.inc"

namespace {

struct ZobInitOnce {
  ZobInitOnce() {
    InitZobrist();
    std::srand(static_cast<unsigned>(std::time(nullptr)));
  }
} g_zob_once;

inline int COORD_XY_ICCS(int x, int y) { return (x + 3) + ((y + 3) << 4); }

inline int SRC(int mv) { return mv & 255; }
inline int DST(int mv) { return mv >> 8; }
inline int MOVE(int sqSrc, int sqDst) { return sqSrc + sqDst * 256; }

inline void sq_to_iccs(int sq, int &x, int &y) {
  x = (sq & 15) - 3;
  y = (sq >> 4) - 3;
}

std::string mv_to_iccs(int mv) {
  int x1, y1, x2, y2;
  sq_to_iccs(SRC(mv), x1, y1);
  sq_to_iccs(DST(mv), x2, y2);
  return std::to_string(x1) + std::to_string(y1) + "-" + std::to_string(x2) + std::to_string(y2);
}

std::optional<int> iccs_to_move(const std::string &s) {
  if (s.size() != 5 || s[2] != '-')
    return std::nullopt;
  int x1 = s[0] - '0', y1 = s[1] - '0', x2 = s[3] - '0', y2 = s[4] - '0';
  if (x1 < 0 || x1 > 8 || y1 < 0 || y1 > 9 || x2 < 0 || x2 > 8 || y2 < 0 || y2 > 9)
    return std::nullopt;
  return MOVE(COORD_XY_ICCS(x1, y1), COORD_XY_ICCS(x2, y2));
}

bool fen_char_to_pc(char ch, BYTE &pc) {
  static const char *RED = "KABNRCP";
  static const char *BLK = "kabnrcp";
  for (int i = 0; i < 7; ++i) {
    if (RED[i] == ch) {
      pc = static_cast<BYTE>(8 + i);
      return true;
    }
    if (BLK[i] == ch) {
      pc = static_cast<BYTE>(16 + i);
      return true;
    }
  }
  return false;
}

bool set_board_from_fen(PositionStruct &board, const std::string &fen) {
  const auto sp = fen.find(' ');
  const std::string ranks = sp == std::string::npos ? fen : fen.substr(0, sp);
  const std::string rest = sp == std::string::npos ? " w" : fen.substr(sp + 1);
  const bool red_to_move = rest.empty() || rest[0] != 'b';

  board.ClearBoard();
  int y = 0;
  int x = 0;
  for (char ch : ranks) {
    if (ch == '/') {
      ++y;
      x = 0;
      continue;
    }
    if (ch >= '1' && ch <= '9') {
      x += ch - '0';
      continue;
    }
    if (x >= 9 || y >= 10) {
      return false;
    }
    BYTE pc = 0;
    if (!fen_char_to_pc(ch, pc)) {
      return false;
    }
    board.AddPiece(COORD_XY_ICCS(x, y), pc);
    ++x;
  }
  if (y != 9 || x != 9) {
    return false;
  }
  board.sdPlayer = red_to_move ? 0 : 1;
  board.nDistance = 0;
  board.RefreshPst();
  board.SetIrrev();
  return true;
}

} // namespace

namespace py = pybind11;

struct XQWLPosition {
  PositionStruct board{};

  XQWLPosition() { reset(); }

  void reset() { board.Startup(); }

  bool set_fen(const std::string &fen) { return set_board_from_fen(board, fen); }

  std::vector<int> legal_moves_mv() const {
    int mvs[MAX_GEN_MOVES];
    int n = board.GenerateMoves(mvs, FALSE);
    if (n > MAX_GEN_MOVES) {
      n = MAX_GEN_MOVES;
    }
    std::vector<int> out;
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      PositionStruct trial = board;
      if (trial.MakeMove(mvs[i])) {
        out.push_back(mvs[i]);
      }
    }
    return out;
  }

  std::vector<std::string> legal_moves_iccs() const {
    auto mvs = legal_moves_mv();
    std::vector<std::string> s;
    s.reserve(mvs.size());
    for (int mv : mvs)
      s.push_back(mv_to_iccs(mv));
    return s;
  }

  bool make_move_mv(int mv) {
    if (!board.MakeMove(mv)) {
      return false;
    }
    board.RefreshPst();
    return true;
  }

  bool make_move_iccs(const std::string &iccs) {
    auto mv = iccs_to_move(iccs);
    if (!mv) {
      return false;
    }
    return make_move_mv(*mv);
  }

  bool pseudo_legal_iccs(const std::string &iccs) const {
    auto mv = iccs_to_move(iccs);
    if (!mv)
      return false;
    return board.LegalMove(*mv);
  }

  bool try_make_move_iccs(const std::string &iccs) {
    auto mv = iccs_to_move(iccs);
    if (!mv || !board.LegalMove(*mv)) {
      return false;
    }
    return make_move_mv(*mv);
  }

  bool captured_last() const { return board.Captured() != 0; }

  void undo() {
    board.UndoMakeMove();
    board.RefreshPst();
  }

  bool in_check() const { return board.InCheck(); }

  int rep_status(int n_recur = 3) const { return board.RepStatus(n_recur); }

  int rep_value(int st) const { return board.RepValue(st); }

  int ply_count() const { return board.nMoveNum; }

  bool is_mate() const {
    PositionStruct trial = board;
    return trial.IsMate() != 0;
  }

  int terminal_kind() const {
    if (is_mate())
      return 1;
    int rs = board.RepStatus(3);
    if (rs > 0)
      return 2;
    if (board.nMoveNum > 100)
      return 3;
    return 0;
  }

  int rep_value_if_any() const {
    int rs = board.RepStatus(3);
    if (rs > 0)
      return board.RepValue(rs);
    return 0;
  }

  std::string fen() const { return xqwl_position_to_fen(board); }

  int evaluate() const { return board.Evaluate(); }

  int side_to_move() const { return board.sdPlayer; }

  std::vector<int> halfka_feature_indices() const {
    int feats[xqwl_halfka::kMaxActive];
    int n = 0;
    xqwl_halfka::extract_features_perspective(board, board.sdPlayer, feats, n);
    return std::vector<int>(feats, feats + n);
  }

  XQWLPosition copy() const {
    XQWLPosition q;
    q.board = board;
    return q;
  }
};

struct XQWLEngine {
  XqwlSearchTables tab{};
  xqwl_nnue::Runtime nnue{};

  XQWLEngine() { tab.nnue = &nnue; }

  void load_book(const std::string &path) { XqwlLoadBookFromFile(path.c_str(), tab); }

  int book_size() const { return tab.nBookSize; }

  bool load_nnue(const std::string &path) {
    const bool ok = nnue.load(path.c_str());
    tab.nnue = ok ? &nnue : nullptr;
    return ok;
  }

  void clear_nnue() {
    nnue.loaded = false;
    tab.nnue = nullptr;
  }

  bool nnue_loaded() const { return nnue.loaded; }

  int evaluate_nnue(const XQWLPosition &position) const {
    if (!nnue.loaded) {
      return 0;
    }
    return nnue.evaluate_vl(position.board, nullptr);
  }

  int evaluate_nnue_raw(const XQWLPosition &position) const {
    if (!nnue.loaded) {
      return 0;
    }
    return nnue.evaluate_vl_raw(position.board, nullptr);
  }

  int verify_nnue_incremental(const XQWLPosition &position) const {
    if (!nnue.loaded) {
      return 0;
    }
    return nnue.verify_incremental(position.board);
  }

  std::string nnue_simd_backend() const {
    if (!nnue.loaded) {
      return "none";
    }
    return xqwl_nnue::simd::backend_name();
  }

  int evaluate_static(const XQWLPosition &position) const { return position.board.Evaluate(); }

  std::string search_best_iccs(const XQWLPosition &position, int time_ms = 1000, bool use_book = true) {
    PositionStruct work = position.board;
    const int mv = XqwlSearchBestMoveEx(work, tab, time_ms, use_book).mv;
    if (mv == 0)
      return "";
    return mv_to_iccs(mv);
  }

  int search_best_mv(const XQWLPosition &position, int time_ms = 1000, bool use_book = true) {
    PositionStruct work = position.board;
    return XqwlSearchBestMoveEx(work, tab, time_ms, use_book).mv;
  }

  py::dict search_best_detail(const XQWLPosition &position, int time_ms = 1000, bool use_book = true) {
    PositionStruct work = position.board;
    const XqwlSearchOutcome out = XqwlSearchBestMoveEx(work, tab, time_ms, use_book);
    py::dict d;
    d["iccs"] = out.mv != 0 ? mv_to_iccs(out.mv) : "";
    d["depth"] = out.depth;
    d["score"] = out.score;
    d["from_book"] = out.from_book;
    return d;
  }
};

PYBIND11_MODULE(xqwlight_core, m) {
  m.doc() = "XQWL06 xiangqi: position rules + Alpha-Beta search (Linux .so via pybind11)";

  py::class_<XQWLPosition>(m, "Position")
      .def(py::init<>())
      .def("reset", &XQWLPosition::reset)
      .def("set_fen", &XQWLPosition::set_fen, py::arg("fen"))
      .def("legal_moves_mv", &XQWLPosition::legal_moves_mv)
      .def("legal_moves_iccs", &XQWLPosition::legal_moves_iccs)
      .def("make_move_mv", &XQWLPosition::make_move_mv)
      .def("make_move_iccs", &XQWLPosition::make_move_iccs)
      .def("pseudo_legal_iccs", &XQWLPosition::pseudo_legal_iccs)
      .def("try_make_move_iccs", &XQWLPosition::try_make_move_iccs)
      .def("captured_last", &XQWLPosition::captured_last)
      .def("undo", &XQWLPosition::undo)
      .def("in_check", &XQWLPosition::in_check)
      .def("rep_status", &XQWLPosition::rep_status, py::arg("n_recur") = 3)
      .def("rep_value", &XQWLPosition::rep_value)
      .def("rep_value_if_any", &XQWLPosition::rep_value_if_any)
      .def("ply_count", &XQWLPosition::ply_count)
      .def("is_mate", &XQWLPosition::is_mate)
      .def("terminal_kind", &XQWLPosition::terminal_kind)
      .def("fen", &XQWLPosition::fen)
      .def("halfka_feature_indices", &XQWLPosition::halfka_feature_indices)
      .def("evaluate", &XQWLPosition::evaluate)
      .def("side_to_move", &XQWLPosition::side_to_move)
      .def("copy", &XQWLPosition::copy);

  py::class_<XQWLEngine>(m, "Engine")
      .def(py::init<>())
      .def("load_book", &XQWLEngine::load_book, py::arg("path"))
      .def("book_size", &XQWLEngine::book_size)
      .def("load_nnue", &XQWLEngine::load_nnue, py::arg("path"))
      .def("clear_nnue", &XQWLEngine::clear_nnue)
      .def("nnue_loaded", &XQWLEngine::nnue_loaded)
      .def("evaluate_static", &XQWLEngine::evaluate_static, py::arg("position"))
      .def("evaluate_nnue", &XQWLEngine::evaluate_nnue, py::arg("position"))
      .def("evaluate_nnue_raw", &XQWLEngine::evaluate_nnue_raw, py::arg("position"))
      .def("verify_nnue_incremental", &XQWLEngine::verify_nnue_incremental, py::arg("position"))
      .def("nnue_simd_backend", &XQWLEngine::nnue_simd_backend)
      .def("search_best_iccs", &XQWLEngine::search_best_iccs, py::arg("position"), py::arg("time_ms") = 1000,
           py::arg("use_book") = true)
      .def("search_best_mv", &XQWLEngine::search_best_mv, py::arg("position"), py::arg("time_ms") = 1000,
           py::arg("use_book") = true)
      .def("search_best_detail", &XQWLEngine::search_best_detail, py::arg("position"), py::arg("time_ms") = 1000,
           py::arg("use_book") = true);

  m.attr("MATE_VALUE") = MATE_VALUE;
  m.attr("BAN_VALUE") = BAN_VALUE;
  m.attr("WIN_VALUE") = WIN_VALUE;
}
