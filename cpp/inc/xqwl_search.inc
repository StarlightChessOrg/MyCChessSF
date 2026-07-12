// XQWL06 search + opening book (ported from Win32/XQWL06.CPP; LoadBook reads a file)
#ifndef XQWL_SEARCH_INC
#define XQWL_SEARCH_INC

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <vector>

struct HashItem {
  BYTE ucDepth, ucFlag;
  short svl;
  short sEval;
  WORD wmv;
  DWORD dwLock0, dwLock1;
};

struct BookItem {
  DWORD dwLock;
  WORD wmv, wvl;
};

struct XqwlSearchTables {
  int mvResult;
  int nHistoryTable[65536];
  int mvKillers[LIMIT_DEPTH][2];
  HashItem HashTable[HASH_SIZE];
  int nBookSize = 0;
  std::vector<BookItem> book_table;
  xqwl_nnue::Runtime *nnue = nullptr;

  XqwlSearchTables() { clear(); }

  void clear() {
    mvResult = 0;
    nBookSize = 0;
    nnue = nullptr;
    book_table.clear();
    std::memset(nHistoryTable, 0, sizeof(nHistoryTable));
    std::memset(mvKillers, 0, sizeof(mvKillers));
    std::memset(HashTable, 0, sizeof(HashTable));
  }
};

struct XqwlSearchCtx {
  PositionStruct *pos;
  XqwlSearchTables *tab;
  int time_limit_ms;
  std::chrono::steady_clock::time_point t_start;
  bool use_book;
  int max_depth;
  int nLastDepth;
  int nLastScore;
  bool from_book;
  xqwl_nnue::SearchState nnue_st;
  short staticEval[LIMIT_DEPTH + 16];
};

static thread_local XqwlSearchCtx *g_xqwl_sc = nullptr;

inline PositionStruct &xqwl_pos() { return *g_xqwl_sc->pos; }
inline XqwlSearchTables &xqwl_tab() { return *g_xqwl_sc->tab; }

#define XQWL_SEARCH_POS (xqwl_pos())
#define XQWL_SEARCH_TAB (xqwl_tab())

enum class XqwlEvalMode { Raw, Full };

static inline int XqwlComputeEval(const PositionStruct &pos, XqwlEvalMode mode) {
  if (g_xqwl_sc != nullptr && g_xqwl_sc->tab != nullptr && g_xqwl_sc->tab->nnue != nullptr &&
      g_xqwl_sc->tab->nnue->loaded) {
    if (mode == XqwlEvalMode::Full) {
      return g_xqwl_sc->tab->nnue->evaluate_vl(pos, &g_xqwl_sc->nnue_st);
    }
    return g_xqwl_sc->tab->nnue->evaluate_vl_raw(pos, &g_xqwl_sc->nnue_st);
  }
  return pos.Evaluate();
}

static inline int XqwlEvalOrTt(int ply, short ttEval, XqwlEvalMode mode) {
  if (ttEval != HASH_EVAL_NONE) {
    g_xqwl_sc->staticEval[ply] = ttEval;
    return ttEval;
  }
  const int vl = XqwlComputeEval(XQWL_SEARCH_POS, mode);
  g_xqwl_sc->staticEval[ply] = static_cast<short>(std::max(-32767, std::min(32767, vl)));
  return vl;
}

static inline int XqwlStaticEval(int ply) { return g_xqwl_sc->staticEval[ply]; }

static inline BOOL XqwlInScoreMateZone(int vl) { return vl > WIN_VALUE || vl < -WIN_VALUE; }

static inline BOOL XqwlImproving(int ply) {
  return ply >= 2 && g_xqwl_sc->staticEval[ply] > g_xqwl_sc->staticEval[ply - 2];
}

static inline BOOL XqwlOpponentWorsening(int ply) {
  return ply >= 1 && g_xqwl_sc->staticEval[ply] > -g_xqwl_sc->staticEval[ply - 1];
}

static inline int XqwlFutilityMargin(int depth, BOOL improving, BOOL oppWorsening) {
  int m = FUTILITY_MARGIN_BASE + FUTILITY_MARGIN_DEPTH * depth;
  if (improving) {
    m -= m / 3;
  }
  if (oppWorsening) {
    m -= m / 8;
  }
  return m;
}

static inline int XqwlCaptureValueCp(BYTE victim) {
  if (!victim) {
    return 0;
  }
  const int kind = (victim < 16) ? static_cast<int>(victim) - 8 : static_cast<int>(victim) - 16;
  static const int cp[7] = {0, 200, 200, 450, 550, 450, 100};
  if (kind < 0 || kind > 6) {
    return 0;
  }
  return cp[kind];
}

/** Search/quiescence eval: NNUE when loaded, else PST ``Evaluate()``. */
inline int XqwlEvalForSearch(const PositionStruct &pos, XqwlEvalMode mode = XqwlEvalMode::Raw) {
  return XqwlComputeEval(pos, mode);
}

static int ProbeHash(int vlAlpha, int vlBeta, int nDepth, int &mv, short *pStaticEval = nullptr);
static void RecordHash(int nFlag, int vl, int nDepth, int mv, short sEval = HASH_EVAL_NONE);

inline bool XqwlSearchMakeMove(int mv) {
  xqwl_nnue::Runtime *nnue =
      (g_xqwl_sc != nullptr && g_xqwl_sc->tab != nullptr) ? g_xqwl_sc->tab->nnue : nullptr;
  if (nnue != nullptr && nnue->loaded) {
    nnue->acc_apply_move(g_xqwl_sc->nnue_st, XQWL_SEARCH_POS, mv);
  }
  if (!XQWL_SEARCH_POS.MakeMove(mv)) {
    if (nnue != nullptr && nnue->loaded) {
      nnue->acc_undo(g_xqwl_sc->nnue_st);
    }
    return false;
  }
  return true;
}

inline void XqwlSearchUndoMakeMove() {
  xqwl_nnue::Runtime *nnue =
      (g_xqwl_sc != nullptr && g_xqwl_sc->tab != nullptr) ? g_xqwl_sc->tab->nnue : nullptr;
  if (nnue != nullptr && nnue->loaded) {
    nnue->acc_undo(g_xqwl_sc->nnue_st);
  }
  XQWL_SEARCH_POS.UndoMakeMove();
}

#ifndef XQWL_NO_BOOK_LOADER
static void XqwlLoadBookFromFile(const char *path, XqwlSearchTables &srch) {
  srch.nBookSize = 0;
  if (path == nullptr || path[0] == '\0') {
    return;
  }
  FILE *f = std::fopen(path, "rb");
  if (f == nullptr) {
    return;
  }
  if (std::fseek(f, 0, SEEK_END) != 0) {
    std::fclose(f);
    return;
  }
  long sz = std::ftell(f);
  if (sz <= 0) {
    std::fclose(f);
    return;
  }
  std::rewind(f);
  const int n_items = static_cast<int>(sz / static_cast<long>(sizeof(BookItem)));
  if (n_items <= 0) {
    std::fclose(f);
    return;
  }
  if (n_items > BOOK_MAX_ENTRIES) {
    std::fclose(f);
    return;
  }
  srch.book_table.resize(static_cast<size_t>(n_items));
  if (static_cast<int>(std::fread(srch.book_table.data(), sizeof(BookItem), static_cast<size_t>(n_items), f)) != n_items) {
    srch.book_table.clear();
    srch.nBookSize = 0;
  } else {
    srch.nBookSize = n_items;
  }
  std::fclose(f);
}
#endif // XQWL_NO_BOOK_LOADER

static int CompareBook(const void *lpbk1, const void *lpbk2) {
  return static_cast<int>(static_cast<const BookItem *>(lpbk1)->dwLock -
                          static_cast<const BookItem *>(lpbk2)->dwLock);
}

static int SearchBook(void) {
  int i, vl, nBookMoves, mv;
  int mvs[MAX_GEN_MOVES], vls[MAX_GEN_MOVES];
  BOOL bMirror;
  BookItem bkToSearch, *lpbk;
  PositionStruct posMirror;

  if (XQWL_SEARCH_TAB.nBookSize == 0) {
    return 0;
  }
  bMirror = FALSE;
  bkToSearch.dwLock = XQWL_SEARCH_POS.zobr.dwLock1;
  BookItem *const book_base = XQWL_SEARCH_TAB.book_table.data();
  lpbk = static_cast<BookItem *>(
      bsearch(&bkToSearch, book_base, static_cast<size_t>(XQWL_SEARCH_TAB.nBookSize), sizeof(BookItem), CompareBook));
  if (lpbk == nullptr) {
    bMirror = TRUE;
    XQWL_SEARCH_POS.Mirror(posMirror);
    bkToSearch.dwLock = posMirror.zobr.dwLock1;
    lpbk = static_cast<BookItem *>(bsearch(
        &bkToSearch, book_base, static_cast<size_t>(XQWL_SEARCH_TAB.nBookSize), sizeof(BookItem), CompareBook));
  }
  if (lpbk == nullptr) {
    return 0;
  }
  while (lpbk >= book_base && lpbk->dwLock == bkToSearch.dwLock) {
    lpbk--;
  }
  lpbk++;
  vl = nBookMoves = 0;
  while (lpbk < book_base + XQWL_SEARCH_TAB.nBookSize && lpbk->dwLock == bkToSearch.dwLock) {
    mv = (bMirror ? MIRROR_MOVE(lpbk->wmv) : lpbk->wmv);
    if (XQWL_SEARCH_POS.LegalMove(mv)) {
      mvs[nBookMoves] = mv;
      vls[nBookMoves] = lpbk->wvl;
      vl += vls[nBookMoves];
      nBookMoves++;
      if (nBookMoves == MAX_GEN_MOVES) {
        break;
      }
    }
    lpbk++;
  }
  if (vl == 0) {
    return 0;
  }
  vl = std::rand() % vl;
  for (i = 0; i < nBookMoves; i++) {
    vl -= vls[i];
    if (vl < 0) {
      break;
    }
  }
  return mvs[i];
}

static int ProbeHash(int vlAlpha, int vlBeta, int nDepth, int &mv, short *pStaticEval) {
  BOOL bMate;
  HashItem hsh;

  if (pStaticEval != nullptr) {
    *pStaticEval = HASH_EVAL_NONE;
  }
  hsh = XQWL_SEARCH_TAB.HashTable[XQWL_SEARCH_POS.zobr.dwKey & (HASH_SIZE - 1)];
  if (hsh.dwLock0 != XQWL_SEARCH_POS.zobr.dwLock0 || hsh.dwLock1 != XQWL_SEARCH_POS.zobr.dwLock1) {
    mv = 0;
    return -MATE_VALUE;
  }
  mv = hsh.wmv;
  if (pStaticEval != nullptr) {
    *pStaticEval = hsh.sEval;
  }
  bMate = FALSE;
  if (hsh.svl > WIN_VALUE) {
    if (hsh.svl < BAN_VALUE) {
      return -MATE_VALUE;
    }
    hsh.svl -= XQWL_SEARCH_POS.nDistance;
    bMate = TRUE;
  } else if (hsh.svl < -WIN_VALUE) {
    if (hsh.svl > -BAN_VALUE) {
      return -MATE_VALUE;
    }
    hsh.svl += XQWL_SEARCH_POS.nDistance;
    bMate = TRUE;
  } else if (hsh.svl == XQWL_SEARCH_POS.DrawValue()) {
    return -MATE_VALUE;
  }
  if (hsh.ucDepth >= nDepth || bMate) {
    if (hsh.ucFlag == HASH_BETA) {
      return (hsh.svl >= vlBeta ? hsh.svl : -MATE_VALUE);
    } else if (hsh.ucFlag == HASH_ALPHA) {
      return (hsh.svl <= vlAlpha ? hsh.svl : -MATE_VALUE);
    }
    return hsh.svl;
  }
  return -MATE_VALUE;
}

static inline int MateValueScore(const PositionStruct &pos) { return pos.nDistance - MATE_VALUE; }

static void SortCheckMoves(int *mvs, int *vls, int nGenMoves) {
  for (int i = 1; i < nGenMoves; ++i) {
    const int mvBest = mvs[i];
    const int vlBest = vls[i];
    int j = i - 1;
    while (j >= 0 && vlBest > vls[j]) {
      mvs[j + 1] = mvs[j];
      vls[j + 1] = vls[j];
      --j;
    }
    mvs[j + 1] = mvBest;
    vls[j + 1] = vlBest;
  }
}

static void RecordHash(int nFlag, int vl, int nDepth, int mv, short sEval) {
  HashItem hsh;
  hsh = XQWL_SEARCH_TAB.HashTable[XQWL_SEARCH_POS.zobr.dwKey & (HASH_SIZE - 1)];
  if (hsh.ucDepth > nDepth) {
    return;
  }
  hsh.ucFlag = static_cast<BYTE>(nFlag);
  hsh.ucDepth = static_cast<BYTE>(nDepth);
  if (sEval != HASH_EVAL_NONE) {
    hsh.sEval = sEval;
  }
  if (vl > WIN_VALUE) {
    if (mv == 0 && vl <= BAN_VALUE) {
      return;
    }
    hsh.svl = static_cast<short>(vl + XQWL_SEARCH_POS.nDistance);
  } else if (vl < -WIN_VALUE) {
    if (mv == 0 && vl >= -BAN_VALUE) {
      return;
    }
    hsh.svl = static_cast<short>(vl - XQWL_SEARCH_POS.nDistance);
  } else if (vl == XQWL_SEARCH_POS.DrawValue() && mv == 0) {
    return;
  } else {
    hsh.svl = static_cast<short>(vl);
  }
  hsh.wmv = static_cast<WORD>(mv);
  hsh.dwLock0 = XQWL_SEARCH_POS.zobr.dwLock0;
  hsh.dwLock1 = XQWL_SEARCH_POS.zobr.dwLock1;
  XQWL_SEARCH_TAB.HashTable[XQWL_SEARCH_POS.zobr.dwKey & (HASH_SIZE - 1)] = hsh;
}

static BYTE cucMvvLva[24] = {
    0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 1, 3, 4, 3, 2, 0, 5, 1, 1, 3, 4, 3, 2, 0};

inline int MvvLva(int mv) {
  return (cucMvvLva[XQWL_SEARCH_POS.ucpcSquares[DST(mv)]] << 3) - cucMvvLva[XQWL_SEARCH_POS.ucpcSquares[SRC(mv)]];
}

static int CompareMvvLva(const void *lpmv1, const void *lpmv2) {
  return MvvLva(*static_cast<const int *>(lpmv2)) - MvvLva(*static_cast<const int *>(lpmv1));
}

/** Trim low-value captures in quiescence (matches xqwlight search.js). */
static int TrimQuiescCaptureCount(int *mvs, int nGenMoves) {
  for (int i = 0; i < nGenMoves; i++) {
    const int vl = MvvLva(mvs[i]);
    if (vl < 10 || (vl < 20 && HOME_HALF(DST(mvs[i]), XQWL_SEARCH_POS.sdPlayer))) {
      return i;
    }
  }
  return nGenMoves;
}

static int CompareHistory(const void *lpmv1, const void *lpmv2) {
  return XQWL_SEARCH_TAB.nHistoryTable[*static_cast<const int *>(lpmv2)] -
         XQWL_SEARCH_TAB.nHistoryTable[*static_cast<const int *>(lpmv1)];
}

const int PHASE_HASH = 0;
const int PHASE_KILLER_1 = 1;
const int PHASE_KILLER_2 = 2;
const int PHASE_GEN_MOVES = 3;
const int PHASE_REST = 4;

struct SortStruct {
  int mvHash, mvKiller1, mvKiller2;
  int nPhase, nIndex, nGenMoves;
  int mvs[MAX_GEN_MOVES];
  int vls[MAX_GEN_MOVES];
  BOOL singleReply;
  BOOL preSortedInCheck;

  void Init(int mvHash_) {
    mvHash = mvHash_;
    mvKiller1 = XQWL_SEARCH_TAB.mvKillers[XQWL_SEARCH_POS.nDistance][0];
    mvKiller2 = XQWL_SEARCH_TAB.mvKillers[XQWL_SEARCH_POS.nDistance][1];
    singleReply = FALSE;
    preSortedInCheck = FALSE;
    if (XQWL_SEARCH_POS.InCheck()) {
      preSortedInCheck = TRUE;
      nPhase = PHASE_REST;
      nGenMoves = 0;
      int tmp[MAX_GEN_MOVES];
      const int n = XQWL_SEARCH_POS.GenerateMoves(tmp);
      for (int j = 0; j < n; ++j) {
        if (XQWL_SEARCH_POS.LegalMove(tmp[j])) {
          mvs[nGenMoves] = tmp[j];
          vls[nGenMoves] = (tmp[j] == mvHash) ? 0x7fffffff : XQWL_SEARCH_TAB.nHistoryTable[tmp[j]];
          ++nGenMoves;
        }
      }
      singleReply = (nGenMoves == 1);
      SortCheckMoves(mvs, vls, nGenMoves);
      nIndex = 0;
    } else {
      nPhase = PHASE_HASH;
    }
  }
  int Next(void);
};

int SortStruct::Next(void) {
  int mv;
  if (preSortedInCheck) {
    if (nIndex < nGenMoves) {
      return mvs[nIndex++];
    }
    return 0;
  }
  switch (nPhase) {
  case PHASE_HASH:
    nPhase = PHASE_KILLER_1;
    if (mvHash != 0) {
      return mvHash;
    }
    [[fallthrough]];
  case PHASE_KILLER_1:
    nPhase = PHASE_KILLER_2;
    if (mvKiller1 != mvHash && mvKiller1 != 0 && XQWL_SEARCH_POS.LegalMove(mvKiller1)) {
      return mvKiller1;
    }
    [[fallthrough]];
  case PHASE_KILLER_2:
    nPhase = PHASE_GEN_MOVES;
    if (mvKiller2 != mvHash && mvKiller2 != 0 && XQWL_SEARCH_POS.LegalMove(mvKiller2)) {
      return mvKiller2;
    }
    [[fallthrough]];
  case PHASE_GEN_MOVES:
    nPhase = PHASE_REST;
    nGenMoves = XQWL_SEARCH_POS.GenerateMoves(mvs);
    qsort(mvs, static_cast<size_t>(nGenMoves), sizeof(int), CompareHistory);
    nIndex = 0;
    [[fallthrough]];
  case PHASE_REST:
    while (nIndex < nGenMoves) {
      mv = mvs[nIndex];
      nIndex++;
      if (mv != mvHash && mv != mvKiller1 && mv != mvKiller2) {
        return mv;
      }
    }
    [[fallthrough]];
  default:
    return 0;
  }
}

inline void SetBestMove(int mv, int nDepth) {
  int *lpmvKillers;
  XQWL_SEARCH_TAB.nHistoryTable[mv] += nDepth * nDepth;
  lpmvKillers = XQWL_SEARCH_TAB.mvKillers[XQWL_SEARCH_POS.nDistance];
  if (lpmvKillers[0] != mv) {
    lpmvKillers[1] = lpmvKillers[0];
    lpmvKillers[0] = mv;
  }
}

inline BOOL IsCaptureMove(int mv) { return XQWL_SEARCH_POS.ucpcSquares[DST(mv)] != 0; }

static int SearchQuiesc(int vlAlpha, int vlBeta) {
  int i, nGenMoves, nHashFlag;
  int vl, vlBest, mvHash, mvBest;
  short ttEval = HASH_EVAL_NONE;
  int mvs[MAX_GEN_MOVES];
  int vls[MAX_GEN_MOVES];
  const int ply = XQWL_SEARCH_POS.nDistance;

  vl = MateValueScore(XQWL_SEARCH_POS);
  if (vl >= vlBeta) {
    return vl;
  }
  vl = XQWL_SEARCH_POS.RepStatus();
  if (vl != 0) {
    return XQWL_SEARCH_POS.RepValue(vl);
  }
  if (XQWL_SEARCH_POS.InCheck() && XQWL_SEARCH_POS.IsMate()) {
    return MateValueScore(XQWL_SEARCH_POS);
  }
  if (ply == LIMIT_DEPTH) {
    return XqwlEvalOrTt(ply, ttEval, XqwlEvalMode::Full);
  }
  vl = ProbeHash(vlAlpha, vlBeta, QUIESCENCE_TT_DEPTH, mvHash, &ttEval);
  if (vl > -MATE_VALUE) {
    return vl;
  }
  if (XQWL_SEARCH_POS.InCheck()) {
    (void)XqwlEvalOrTt(ply, ttEval, XqwlEvalMode::Raw);
  }
  vlBest = -MATE_VALUE;
  mvBest = 0;
  nHashFlag = HASH_ALPHA;

  if (XQWL_SEARCH_POS.InCheck()) {
    nGenMoves = XQWL_SEARCH_POS.GenerateMoves(mvs);
    for (i = 0; i < nGenMoves; ++i) {
      vls[i] = (mvs[i] == mvHash) ? 0x7fffffff : XQWL_SEARCH_TAB.nHistoryTable[mvs[i]];
    }
    SortCheckMoves(mvs, vls, nGenMoves);
  } else {
    vl = XqwlEvalOrTt(ply, ttEval, XqwlEvalMode::Full);
    if (vl > vlBest) {
      vlBest = vl;
      if (vl >= vlBeta) {
        RecordHash(HASH_BETA, vl, QUIESCENCE_TT_DEPTH, 0, g_xqwl_sc->staticEval[ply]);
        return vl;
      }
      if (vl > vlAlpha) {
        vlAlpha = vl;
        nHashFlag = HASH_PV;
      }
    }
    nGenMoves = XQWL_SEARCH_POS.GenerateMoves(mvs, GEN_CAPTURE);
    qsort(mvs, static_cast<size_t>(nGenMoves), sizeof(int), CompareMvvLva);
    nGenMoves = TrimQuiescCaptureCount(mvs, nGenMoves);
  }

  for (i = 0; i < nGenMoves; i++) {
    if (!XQWL_SEARCH_POS.InCheck() && vlBest > -WIN_VALUE) {
      const int capCp = XqwlCaptureValueCp(XQWL_SEARCH_POS.ucpcSquares[DST(mvs[i])]);
      if (vlBest + capCp + QCAP_FUTILITY_MARGIN <= vlAlpha) {
        continue;
      }
    }
    if (XqwlSearchMakeMove(mvs[i])) {
      vl = -SearchQuiesc(-vlBeta, -vlAlpha);
      XqwlSearchUndoMakeMove();
      if (vl > vlBest) {
        vlBest = vl;
        mvBest = mvs[i];
        if (vl >= vlBeta) {
          RecordHash(HASH_BETA, vlBest, QUIESCENCE_TT_DEPTH, mvBest, g_xqwl_sc->staticEval[ply]);
          return vl;
        }
        if (vl > vlAlpha) {
          vlAlpha = vl;
          nHashFlag = HASH_PV;
        }
      }
    }
  }
  if (vlBest == -MATE_VALUE) {
    return XQWL_SEARCH_POS.nDistance - MATE_VALUE;
  }
  RecordHash(nHashFlag, vlBest, QUIESCENCE_TT_DEPTH, mvBest, g_xqwl_sc->staticEval[ply]);
  return vlBest;
}

const BOOL NO_NULL = TRUE;

static int SearchFull(int vlAlpha, int vlBeta, int nDepth, BOOL bNoNull = FALSE) {
  int nHashFlag, vl, vlBest;
  int mv, mvBest, mvHash, nNewDepth, moveCount;
  short ttEval = HASH_EVAL_NONE;
  BOOL bInCheck;
  SortStruct Sort;
  const int ply = XQWL_SEARCH_POS.nDistance;

  if (nDepth <= 0) {
    return SearchQuiesc(vlAlpha, vlBeta);
  }
  vl = MateValueScore(XQWL_SEARCH_POS);
  if (vl >= vlBeta) {
    return vl;
  }
  vl = XQWL_SEARCH_POS.RepStatus();
  if (vl != 0) {
    return XQWL_SEARCH_POS.RepValue(vl);
  }
  if (XQWL_SEARCH_POS.InCheck() && XQWL_SEARCH_POS.IsMate()) {
    return MateValueScore(XQWL_SEARCH_POS);
  }
  if (ply == LIMIT_DEPTH) {
    return XqwlEvalOrTt(ply, ttEval, XqwlEvalMode::Full);
  }
  vl = ProbeHash(vlAlpha, vlBeta, nDepth, mvHash, &ttEval);
  if (vl > -MATE_VALUE) {
    return vl;
  }
  (void)XqwlEvalOrTt(ply, ttEval, XqwlEvalMode::Raw);
  const int sEval = XqwlStaticEval(ply);
  const BOOL improving = XqwlImproving(ply);
  const BOOL oppWorsening = XqwlOpponentWorsening(ply);
  bInCheck = XQWL_SEARCH_POS.InCheck();
  if (!bInCheck && nDepth <= RAZOR_MAX_DEPTH && !XqwlInScoreMateZone(vlAlpha) && !XqwlInScoreMateZone(vlBeta)) {
    const int razor = RAZOR_MARGIN_BASE + RAZOR_MARGIN_DEPTH * nDepth;
    if (sEval + razor < vlAlpha) {
      return SearchQuiesc(vlAlpha, vlBeta);
    }
  }
  if (!bInCheck && nDepth <= FUTILITY_MAX_DEPTH && vlBeta < WIN_VALUE && !XqwlInScoreMateZone(vlBeta)) {
    int rfMargin = REVERSE_FUT_MARGIN_BASE + REVERSE_FUT_MARGIN_DEPTH * nDepth;
    if (improving) {
      rfMargin -= rfMargin / 4;
    }
    if (sEval - rfMargin >= vlBeta) {
      return sEval;
    }
  }
  if (nDepth >= IID_MIN_DEPTH && mvHash == 0 && !bInCheck) {
    SearchFull(vlAlpha, vlBeta, nDepth - 2, NO_NULL);
    (void)ProbeHash(vlAlpha, vlBeta, nDepth, mvHash, &ttEval);
  }
  if (!bNoNull && !bInCheck && XQWL_SEARCH_POS.NullOkay()) {
    const int nullStaticMargin =
        NULL_STATIC_MARGIN_DEPTH * nDepth + (improving ? 0 : NULL_STATIC_MARGIN_IMPROVING);
    if (sEval >= vlBeta - nullStaticMargin) {
      const int attack = XQWL_SEARCH_POS.AttackingPieceCount(XQWL_SEARCH_POS.sdPlayer);
      if (attack > 0) {
        const BOOL verifyNull = (attack < NULL_ATTACKING_MIN);
        XQWL_SEARCH_POS.NullMove();
        vl = -SearchFull(-vlBeta, 1 - vlBeta, nDepth - NULL_DEPTH - 1, NO_NULL);
        XQWL_SEARCH_POS.UndoNullMove();
        if (vl >= vlBeta) {
          if (!verifyNull || XQWL_SEARCH_POS.NullSafe() ||
              -SearchFull(-vlBeta, 1 - vlBeta, nDepth - NULL_DEPTH, NO_NULL) >= vlBeta) {
            return vl;
          }
        }
      }
    }
  }
  nHashFlag = HASH_ALPHA;
  vlBest = -MATE_VALUE;
  mvBest = 0;
  moveCount = 0;
  BOOL ttSingularExtend = FALSE;
  if (mvHash != 0 && nDepth >= SINGULAR_MIN_DEPTH && !bInCheck) {
    const int singularBeta = vlAlpha;
    BOOL hasRefutation = FALSE;
    SortStruct sTest;
    sTest.Init(mvHash);
    (void)sTest.Next();
    int othMv;
    while ((othMv = sTest.Next()) != 0) {
      if (XqwlSearchMakeMove(othMv)) {
        const int td = (nDepth - 3) > 1 ? (nDepth - 3) : 1;
        const int sv = -SearchFull(-singularBeta, -singularBeta + 1, td, NO_NULL);
        XqwlSearchUndoMakeMove();
        if (sv >= singularBeta) {
          hasRefutation = TRUE;
          break;
        }
      }
    }
    if (!hasRefutation) {
      ttSingularExtend = TRUE;
    }
  }
  Sort.Init(mvHash);
  while ((mv = Sort.Next()) != 0) {
    moveCount++;
    const BOOL bCapture = IsCaptureMove(mv);
    if (!bInCheck && !bCapture && vlBest != -MATE_VALUE) {
      const int quietDepth = (Sort.singleReply) ? nDepth : nDepth - 1;
      const int futMargin = XqwlFutilityMargin(nDepth, improving, oppWorsening);
      if (nDepth <= FUTILITY_MAX_DEPTH && moveCount > 1 && sEval + futMargin <= vlAlpha && quietDepth <= 3) {
        continue;
      }
      const int lmpLimit = (3 + nDepth * nDepth) / (LMP_MOVE_DIV + static_cast<int>(improving));
      if (nDepth <= LMP_MAX_DEPTH && moveCount >= lmpLimit && vlBest > -WIN_VALUE) {
        continue;
      }
    }
    if (XqwlSearchMakeMove(mv)) {
      nNewDepth = (XQWL_SEARCH_POS.InCheck() || Sort.singleReply) ? nDepth : nDepth - 1;
      if (mv == mvHash && ttSingularExtend) {
        ++nNewDepth;
      }
      const BOOL bReduced = moveCount > LMR_MOVE_THRESHOLD && nDepth >= LMR_MIN_DEPTH && !bInCheck &&
                            !bCapture && nNewDepth >= 2 && !(mv == mvHash && ttSingularExtend);
      const int reducedDepth = nNewDepth - 1;
      if (vlBest == -MATE_VALUE) {
        if (bReduced) {
          vl = -SearchFull(-vlBeta, -vlAlpha, reducedDepth);
          if (vl > vlAlpha) {
            vl = -SearchFull(-vlBeta, -vlAlpha, nNewDepth);
          }
        } else {
          vl = -SearchFull(-vlBeta, -vlAlpha, nNewDepth);
        }
      } else {
        if (bReduced) {
          vl = -SearchFull(-vlAlpha - 1, -vlAlpha, reducedDepth);
        } else {
          vl = -SearchFull(-vlAlpha - 1, -vlAlpha, nNewDepth);
        }
        if (vl > vlAlpha && vl < vlBeta) {
          vl = -SearchFull(-vlBeta, -vlAlpha, nNewDepth);
        }
      }
      XqwlSearchUndoMakeMove();
      if (vl > vlBest) {
        vlBest = vl;
        if (vl >= vlBeta) {
          nHashFlag = HASH_BETA;
          mvBest = mv;
          break;
        }
        if (vl > vlAlpha) {
          nHashFlag = HASH_PV;
          mvBest = mv;
          vlAlpha = vl;
        }
      }
    }
  }
  if (vlBest == -MATE_VALUE) {
    return XQWL_SEARCH_POS.nDistance - MATE_VALUE;
  }
  RecordHash(nHashFlag, vlBest, nDepth, mvBest, g_xqwl_sc->staticEval[ply]);
  if (mvBest != 0) {
    SetBestMove(mvBest, nDepth);
  }
  return vlBest;
}

static BOOL SearchUnique(int vlBeta, int nDepth) {
  SortStruct Sort;
  int mv;

  Sort.Init(XQWL_SEARCH_TAB.mvResult);
  Sort.Next();
  while ((mv = Sort.Next()) != 0) {
    if (XqwlSearchMakeMove(mv)) {
      const int nNewDepth = (XQWL_SEARCH_POS.InCheck() || Sort.singleReply) ? nDepth : nDepth - 1;
      const int vl = -SearchFull(-vlBeta, 1 - vlBeta, nNewDepth);
      XqwlSearchUndoMakeMove();
      if (vl >= vlBeta) {
        return FALSE;
      }
    }
  }
  return TRUE;
}

static int SearchRoot(int nDepth, int vlAlpha, int vlBeta) {
  int vl, vlBest, mv, nNewDepth;
  SortStruct Sort;

  vlBest = -MATE_VALUE;
  Sort.Init(XQWL_SEARCH_TAB.mvResult);
  while ((mv = Sort.Next()) != 0) {
    if (XqwlSearchMakeMove(mv)) {
      nNewDepth = (XQWL_SEARCH_POS.InCheck() || Sort.singleReply) ? nDepth : nDepth - 1;
      if (vlBest == -MATE_VALUE) {
        vl = -SearchFull(-vlBeta, -vlAlpha, nNewDepth, NO_NULL);
      } else {
        vl = -SearchFull(-vlBest - 1, -vlBest, nNewDepth);
        if (vl > vlBest) {
          vl = -SearchFull(-vlBeta, -vlBest, nNewDepth, NO_NULL);
        }
      }
      XqwlSearchUndoMakeMove();
      if (vl > vlBest) {
        vlBest = vl;
        XQWL_SEARCH_TAB.mvResult = mv;
        if (vlBest > -WIN_VALUE && vlBest < WIN_VALUE) {
          vlBest += (std::rand() & RANDOM_MASK) - (std::rand() & RANDOM_MASK);
        }
      }
    }
  }
  RecordHash(HASH_PV, vlBest, nDepth, XQWL_SEARCH_TAB.mvResult, g_xqwl_sc->staticEval[0]);
  SetBestMove(XQWL_SEARCH_TAB.mvResult, nDepth);
  return vlBest;
}

static void SearchMain(void) {
  int i, vl, nGenMoves;
  int mvs[MAX_GEN_MOVES];
  const auto deadline =
      g_xqwl_sc->t_start + std::chrono::milliseconds(g_xqwl_sc->time_limit_ms);

  g_xqwl_sc->nLastDepth = 0;
  g_xqwl_sc->nLastScore = XqwlEvalForSearch(XQWL_SEARCH_POS, XqwlEvalMode::Full);
  g_xqwl_sc->staticEval[0] =
      static_cast<short>(std::max(-32767, std::min(32767, g_xqwl_sc->nLastScore)));
  g_xqwl_sc->from_book = false;
  std::memset(XQWL_SEARCH_TAB.nHistoryTable, 0, 65536 * sizeof(int));
  std::memset(XQWL_SEARCH_TAB.mvKillers, 0, LIMIT_DEPTH * 2 * sizeof(int));
  XQWL_SEARCH_POS.nDistance = 0;

  if (g_xqwl_sc->tab->nnue != nullptr && g_xqwl_sc->tab->nnue->loaded) {
    g_xqwl_sc->tab->nnue->acc_init(g_xqwl_sc->nnue_st, XQWL_SEARCH_POS);
  } else {
    g_xqwl_sc->nnue_st.ready = false;
  }

  if (g_xqwl_sc->use_book) {
    XQWL_SEARCH_TAB.mvResult = SearchBook();
    if (XQWL_SEARCH_TAB.mvResult != 0) {
      if (XqwlSearchMakeMove(XQWL_SEARCH_TAB.mvResult)) {
        if (XQWL_SEARCH_POS.RepStatus(3) == 0) {
          XqwlSearchUndoMakeMove();
          g_xqwl_sc->from_book = true;
          g_xqwl_sc->nLastDepth = 0;
          g_xqwl_sc->nLastScore = XqwlEvalForSearch(XQWL_SEARCH_POS, XqwlEvalMode::Full);
          return;
        }
        XqwlSearchUndoMakeMove();
      }
    }
  }

  vl = 0;
  nGenMoves = XQWL_SEARCH_POS.GenerateMoves(mvs);
  for (i = 0; i < nGenMoves; i++) {
    if (XQWL_SEARCH_POS.LegalMove(mvs[i])) {
      XQWL_SEARCH_TAB.mvResult = mvs[i];
      vl++;
    }
  }
  if (vl == 1) {
    g_xqwl_sc->nLastDepth = 1;
    g_xqwl_sc->nLastScore = SearchRoot(1, -MATE_VALUE, MATE_VALUE);
    return;
  }

  for (i = 1; i <= ((g_xqwl_sc->max_depth > 0) ? g_xqwl_sc->max_depth : LIMIT_DEPTH); i++) {
    if (i < ASPIRATION_MIN_DEPTH) {
      vl = SearchRoot(i, -MATE_VALUE, MATE_VALUE);
    } else {
      int delta = ASPIRATION_DELTA_INIT + std::abs(g_xqwl_sc->nLastScore) / 32;
      if (delta > ASPIRATION_DELTA_MAX) {
        delta = ASPIRATION_DELTA_MAX;
      }
      int alpha = std::max(-MATE_VALUE, g_xqwl_sc->nLastScore - delta);
      int beta = std::min(MATE_VALUE, g_xqwl_sc->nLastScore + delta);
      int failLow = 0;
      int failHigh = 0;
      while (true) {
        vl = SearchRoot(i, alpha, beta);
        if (vl > alpha && vl < beta) {
          break;
        }
        if (vl <= alpha) {
          beta = (alpha + beta) / 2;
          alpha = std::max(-MATE_VALUE, vl - delta);
          delta += delta / 4;
          if (delta > ASPIRATION_DELTA_MAX) {
            delta = ASPIRATION_DELTA_MAX;
          }
          if (++failLow >= ASPIRATION_MAX_FAILS) {
            alpha = -MATE_VALUE;
          }
        } else {
          alpha = (alpha + beta) / 2;
          beta = std::min(MATE_VALUE, vl + delta);
          delta += delta / 4;
          if (delta > ASPIRATION_DELTA_MAX) {
            delta = ASPIRATION_DELTA_MAX;
          }
          if (++failHigh >= ASPIRATION_MAX_FAILS) {
            beta = MATE_VALUE;
          }
        }
        if (alpha <= -WIN_VALUE && beta >= WIN_VALUE) {
          vl = SearchRoot(i, alpha, beta);
          break;
        }
      }
    }
    g_xqwl_sc->nLastDepth = i;
    g_xqwl_sc->nLastScore = vl;
    if (vl > WIN_VALUE || vl < -WIN_VALUE) {
      break;
    }
    if (SearchUnique(1 - WIN_VALUE, i)) {
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      break;
    }
  }
}

/** Outcome of a search call (move + completed depth + root score + book flag). */
struct XqwlSearchOutcome {
  int mv;
  int depth;
  int score;
  bool from_book;
};

/** Search best move on a copy of ``position``; ``tables`` holds TT/book across calls. */
inline XqwlSearchOutcome XqwlSearchBestMoveEx(PositionStruct &position, XqwlSearchTables &tables, int time_limit_ms,
                                              bool use_book = true, int max_depth = 0) {
  if (time_limit_ms < 50) {
    time_limit_ms = 50;
  }
  if (max_depth < 0) {
    max_depth = 0;
  }
  if (max_depth > LIMIT_DEPTH) {
    max_depth = LIMIT_DEPTH;
  }
  XqwlSearchCtx ctx{&position, &tables, time_limit_ms, std::chrono::steady_clock::now(),
                    use_book, max_depth, 0, 0, false, {}, {}};
  // Blend PST once for this search from the root position; tree MakeMove stays incremental.
  position.RefreshPst();
  g_xqwl_sc = &ctx;
  ctx.nLastScore = XqwlEvalForSearch(position, XqwlEvalMode::Full);
  tables.mvResult = 0;
  SearchMain();
  const XqwlSearchOutcome out{tables.mvResult, ctx.nLastDepth, ctx.nLastScore, ctx.from_book};
  g_xqwl_sc = nullptr;
  return out;
}

inline int XqwlSearchBestMove(PositionStruct &position, XqwlSearchTables &tables, int time_limit_ms,
                              bool use_book = true) {
  return XqwlSearchBestMoveEx(position, tables, time_limit_ms, use_book).mv;
}

#endif // XQWL_SEARCH_INC
