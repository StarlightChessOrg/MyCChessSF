// XQWL06 search + opening book (ported from Win32/XQWL06.CPP; LoadBook reads a file)
#ifndef XQWL_SEARCH_INC
#define XQWL_SEARCH_INC

#include <cstdio>
#include <cstdlib>
#include <ctime>

struct HashItem {
  BYTE ucDepth, ucFlag;
  short svl;
  WORD wmv, wReserved;
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
  int nBookSize;
  BookItem BookTable[BOOK_SIZE];
};

struct XqwlSearchCtx {
  PositionStruct *pos;
  XqwlSearchTables *tab;
  int time_limit_ms;
  clock_t t_start;
  bool use_book;
  int nLastDepth;
  bool from_book;
};

static thread_local XqwlSearchCtx *g_xqwl_sc = nullptr;

inline PositionStruct &xqwl_pos() { return *g_xqwl_sc->pos; }
inline XqwlSearchTables &xqwl_tab() { return *g_xqwl_sc->tab; }

#define XQWL_SEARCH_POS (xqwl_pos())
#define XQWL_SEARCH_TAB (xqwl_tab())

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
  int n = static_cast<int>(sz / static_cast<long>(sizeof(BookItem)));
  if (n > BOOK_SIZE) {
    n = BOOK_SIZE;
  }
  if (static_cast<int>(std::fread(srch.BookTable, sizeof(BookItem), static_cast<size_t>(n), f)) != n) {
    srch.nBookSize = 0;
  } else {
    srch.nBookSize = n;
  }
  std::fclose(f);
}

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
  lpbk = static_cast<BookItem *>(
      bsearch(&bkToSearch, XQWL_SEARCH_TAB.BookTable, static_cast<size_t>(XQWL_SEARCH_TAB.nBookSize), sizeof(BookItem), CompareBook));
  if (lpbk == nullptr) {
    bMirror = TRUE;
    XQWL_SEARCH_POS.Mirror(posMirror);
    bkToSearch.dwLock = posMirror.zobr.dwLock1;
    lpbk = static_cast<BookItem *>(bsearch(
        &bkToSearch, XQWL_SEARCH_TAB.BookTable, static_cast<size_t>(XQWL_SEARCH_TAB.nBookSize), sizeof(BookItem), CompareBook));
  }
  if (lpbk == nullptr) {
    return 0;
  }
  while (lpbk >= XQWL_SEARCH_TAB.BookTable && lpbk->dwLock == bkToSearch.dwLock) {
    lpbk--;
  }
  lpbk++;
  vl = nBookMoves = 0;
  while (lpbk < XQWL_SEARCH_TAB.BookTable + XQWL_SEARCH_TAB.nBookSize && lpbk->dwLock == bkToSearch.dwLock) {
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

static int ProbeHash(int vlAlpha, int vlBeta, int nDepth, int &mv) {
  BOOL bMate;
  HashItem hsh;

  hsh = XQWL_SEARCH_TAB.HashTable[XQWL_SEARCH_POS.zobr.dwKey & (HASH_SIZE - 1)];
  if (hsh.dwLock0 != XQWL_SEARCH_POS.zobr.dwLock0 || hsh.dwLock1 != XQWL_SEARCH_POS.zobr.dwLock1) {
    mv = 0;
    return -MATE_VALUE;
  }
  mv = hsh.wmv;
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

static void RecordHash(int nFlag, int vl, int nDepth, int mv) {
  HashItem hsh;
  hsh = XQWL_SEARCH_TAB.HashTable[XQWL_SEARCH_POS.zobr.dwKey & (HASH_SIZE - 1)];
  if (hsh.ucDepth > nDepth) {
    return;
  }
  hsh.ucFlag = static_cast<BYTE>(nFlag);
  hsh.ucDepth = static_cast<BYTE>(nDepth);
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

  void Init(int mvHash_) {
    mvHash = mvHash_;
    mvKiller1 = XQWL_SEARCH_TAB.mvKillers[XQWL_SEARCH_POS.nDistance][0];
    mvKiller2 = XQWL_SEARCH_TAB.mvKillers[XQWL_SEARCH_POS.nDistance][1];
    nPhase = PHASE_HASH;
  }
  int Next(void);
};

int SortStruct::Next(void) {
  int mv;
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

static int SearchQuiesc(int vlAlpha, int vlBeta) {
  int i, nGenMoves;
  int vl, vlBest;
  int mvs[MAX_GEN_MOVES];

  vl = XQWL_SEARCH_POS.RepStatus();
  if (vl != 0) {
    return XQWL_SEARCH_POS.RepValue(vl);
  }
  if (XQWL_SEARCH_POS.nDistance == LIMIT_DEPTH) {
    return XQWL_SEARCH_POS.Evaluate();
  }
  vlBest = -MATE_VALUE;

  if (XQWL_SEARCH_POS.InCheck()) {
    nGenMoves = XQWL_SEARCH_POS.GenerateMoves(mvs);
    qsort(mvs, static_cast<size_t>(nGenMoves), sizeof(int), CompareHistory);
  } else {
    vl = XQWL_SEARCH_POS.Evaluate();
    if (vl > vlBest) {
      vlBest = vl;
      if (vl >= vlBeta) {
        return vl;
      }
      if (vl > vlAlpha) {
        vlAlpha = vl;
      }
    }
    nGenMoves = XQWL_SEARCH_POS.GenerateMoves(mvs, GEN_CAPTURE);
    qsort(mvs, static_cast<size_t>(nGenMoves), sizeof(int), CompareMvvLva);
  }

  for (i = 0; i < nGenMoves; i++) {
    if (XQWL_SEARCH_POS.MakeMove(mvs[i])) {
      vl = -SearchQuiesc(-vlBeta, -vlAlpha);
      XQWL_SEARCH_POS.UndoMakeMove();
      if (vl > vlBest) {
        vlBest = vl;
        if (vl >= vlBeta) {
          return vl;
        }
        if (vl > vlAlpha) {
          vlAlpha = vl;
        }
      }
    }
  }
  return vlBest == -MATE_VALUE ? XQWL_SEARCH_POS.nDistance - MATE_VALUE : vlBest;
}

const BOOL NO_NULL = TRUE;

static int SearchFull(int vlAlpha, int vlBeta, int nDepth, BOOL bNoNull = FALSE) {
  int nHashFlag, vl, vlBest;
  int mv, mvBest, mvHash, nNewDepth;
  SortStruct Sort;

  if (nDepth <= 0) {
    return SearchQuiesc(vlAlpha, vlBeta);
  }
  vl = XQWL_SEARCH_POS.RepStatus();
  if (vl != 0) {
    return XQWL_SEARCH_POS.RepValue(vl);
  }
  if (XQWL_SEARCH_POS.nDistance == LIMIT_DEPTH) {
    return XQWL_SEARCH_POS.Evaluate();
  }
  vl = ProbeHash(vlAlpha, vlBeta, nDepth, mvHash);
  if (vl > -MATE_VALUE) {
    return vl;
  }
  if (!bNoNull && !XQWL_SEARCH_POS.InCheck() && XQWL_SEARCH_POS.NullOkay()) {
    XQWL_SEARCH_POS.NullMove();
    vl = -SearchFull(-vlBeta, 1 - vlBeta, nDepth - NULL_DEPTH - 1, NO_NULL);
    XQWL_SEARCH_POS.UndoNullMove();
    if (vl >= vlBeta) {
      return vl;
    }
  }
  nHashFlag = HASH_ALPHA;
  vlBest = -MATE_VALUE;
  mvBest = 0;
  Sort.Init(mvHash);
  while ((mv = Sort.Next()) != 0) {
    if (XQWL_SEARCH_POS.MakeMove(mv)) {
      nNewDepth = XQWL_SEARCH_POS.InCheck() ? nDepth : nDepth - 1;
      if (vlBest == -MATE_VALUE) {
        vl = -SearchFull(-vlBeta, -vlAlpha, nNewDepth);
      } else {
        vl = -SearchFull(-vlAlpha - 1, -vlAlpha, nNewDepth);
        if (vl > vlAlpha && vl < vlBeta) {
          vl = -SearchFull(-vlBeta, -vlAlpha, nNewDepth);
        }
      }
      XQWL_SEARCH_POS.UndoMakeMove();
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
  RecordHash(nHashFlag, vlBest, nDepth, mvBest);
  if (mvBest != 0) {
    SetBestMove(mvBest, nDepth);
  }
  return vlBest;
}

static int SearchRoot(int nDepth) {
  int vl, vlBest, mv, nNewDepth;
  SortStruct Sort;

  vlBest = -MATE_VALUE;
  Sort.Init(XQWL_SEARCH_TAB.mvResult);
  while ((mv = Sort.Next()) != 0) {
    if (XQWL_SEARCH_POS.MakeMove(mv)) {
      nNewDepth = XQWL_SEARCH_POS.InCheck() ? nDepth : nDepth - 1;
      if (vlBest == -MATE_VALUE) {
        vl = -SearchFull(-MATE_VALUE, MATE_VALUE, nNewDepth, NO_NULL);
      } else {
        vl = -SearchFull(-vlBest - 1, -vlBest, nNewDepth);
        if (vl > vlBest) {
          vl = -SearchFull(-MATE_VALUE, -vlBest, nNewDepth, NO_NULL);
        }
      }
      XQWL_SEARCH_POS.UndoMakeMove();
      if (vl > vlBest) {
        vlBest = vl;
        XQWL_SEARCH_TAB.mvResult = mv;
        if (vlBest > -WIN_VALUE && vlBest < WIN_VALUE) {
          vlBest += (std::rand() & RANDOM_MASK) - (std::rand() & RANDOM_MASK);
        }
      }
    }
  }
  RecordHash(HASH_PV, vlBest, nDepth, XQWL_SEARCH_TAB.mvResult);
  SetBestMove(XQWL_SEARCH_TAB.mvResult, nDepth);
  return vlBest;
}

static void SearchMain(void) {
  int i, vl, nGenMoves;
  int mvs[MAX_GEN_MOVES];
  clock_t deadline =
      g_xqwl_sc->t_start + static_cast<clock_t>(g_xqwl_sc->time_limit_ms) * CLOCKS_PER_SEC / 1000;

  g_xqwl_sc->nLastDepth = 0;
  g_xqwl_sc->from_book = false;
  std::memset(XQWL_SEARCH_TAB.nHistoryTable, 0, 65536 * sizeof(int));
  std::memset(XQWL_SEARCH_TAB.mvKillers, 0, LIMIT_DEPTH * 2 * sizeof(int));
  std::memset(XQWL_SEARCH_TAB.HashTable, 0, HASH_SIZE * sizeof(HashItem));
  XQWL_SEARCH_POS.nDistance = 0;

  if (g_xqwl_sc->use_book) {
    XQWL_SEARCH_TAB.mvResult = SearchBook();
    if (XQWL_SEARCH_TAB.mvResult != 0) {
      if (XQWL_SEARCH_POS.MakeMove(XQWL_SEARCH_TAB.mvResult)) {
        if (XQWL_SEARCH_POS.RepStatus(3) == 0) {
          XQWL_SEARCH_POS.UndoMakeMove();
          g_xqwl_sc->from_book = true;
          g_xqwl_sc->nLastDepth = 0;
          return;
        }
        XQWL_SEARCH_POS.UndoMakeMove();
      }
    }
  }

  vl = 0;
  nGenMoves = XQWL_SEARCH_POS.GenerateMoves(mvs);
  for (i = 0; i < nGenMoves; i++) {
    if (XQWL_SEARCH_POS.MakeMove(mvs[i])) {
      XQWL_SEARCH_POS.UndoMakeMove();
      XQWL_SEARCH_TAB.mvResult = mvs[i];
      vl++;
    }
  }
  if (vl == 1) {
    return;
  }

  for (i = 1; i <= LIMIT_DEPTH; i++) {
    vl = SearchRoot(i);
    g_xqwl_sc->nLastDepth = i;
    if (vl > WIN_VALUE || vl < -WIN_VALUE) {
      break;
    }
    if (std::clock() >= deadline) {
      break;
    }
  }
}

/** Outcome of a search call (move + completed depth + book flag). */
struct XqwlSearchOutcome {
  int mv;
  int depth;
  bool from_book;
};

/** Search best move on a copy of ``position``; ``tables`` holds TT/book across calls. */
inline XqwlSearchOutcome XqwlSearchBestMoveEx(PositionStruct &position, XqwlSearchTables &tables, int time_limit_ms,
                                              bool use_book = true) {
  if (time_limit_ms < 50) {
    time_limit_ms = 50;
  }
  XqwlSearchCtx ctx{&position, &tables, time_limit_ms, std::clock(), use_book, 0, false};
  g_xqwl_sc = &ctx;
  tables.mvResult = 0;
  SearchMain();
  const XqwlSearchOutcome out{tables.mvResult, ctx.nLastDepth, ctx.from_book};
  g_xqwl_sc = nullptr;
  return out;
}

inline int XqwlSearchBestMove(PositionStruct &position, XqwlSearchTables &tables, int time_limit_ms,
                              bool use_book = true) {
  return XqwlSearchBestMoveEx(position, tables, time_limit_ms, use_book).mv;
}

#endif // XQWL_SEARCH_INC
