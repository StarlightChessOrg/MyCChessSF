// XQWL06 搜索与开局库（自 Win32/XQWL06.CPP 移植；LoadBook 改为读文件）
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
};

static thread_local XqwlSearchCtx *g_xqwl_sc = nullptr;

inline PositionStruct &xqwl_pos() { return *g_xqwl_sc->pos; }
inline XqwlSearchTables &xqwl_tab() { return *g_xqwl_sc->tab; }

#define pos xqwl_pos()
#define Search xqwl_tab()

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

  if (Search.nBookSize == 0) {
    return 0;
  }
  bMirror = FALSE;
  bkToSearch.dwLock = pos.zobr.dwLock1;
  lpbk = static_cast<BookItem *>(
      bsearch(&bkToSearch, Search.BookTable, static_cast<size_t>(Search.nBookSize), sizeof(BookItem), CompareBook));
  if (lpbk == nullptr) {
    bMirror = TRUE;
    pos.Mirror(posMirror);
    bkToSearch.dwLock = posMirror.zobr.dwLock1;
    lpbk = static_cast<BookItem *>(bsearch(
        &bkToSearch, Search.BookTable, static_cast<size_t>(Search.nBookSize), sizeof(BookItem), CompareBook));
  }
  if (lpbk == nullptr) {
    return 0;
  }
  while (lpbk >= Search.BookTable && lpbk->dwLock == bkToSearch.dwLock) {
    lpbk--;
  }
  lpbk++;
  vl = nBookMoves = 0;
  while (lpbk < Search.BookTable + Search.nBookSize && lpbk->dwLock == bkToSearch.dwLock) {
    mv = (bMirror ? MIRROR_MOVE(lpbk->wmv) : lpbk->wmv);
    if (pos.LegalMove(mv)) {
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

  hsh = Search.HashTable[pos.zobr.dwKey & (HASH_SIZE - 1)];
  if (hsh.dwLock0 != pos.zobr.dwLock0 || hsh.dwLock1 != pos.zobr.dwLock1) {
    mv = 0;
    return -MATE_VALUE;
  }
  mv = hsh.wmv;
  bMate = FALSE;
  if (hsh.svl > WIN_VALUE) {
    if (hsh.svl < BAN_VALUE) {
      return -MATE_VALUE;
    }
    hsh.svl -= pos.nDistance;
    bMate = TRUE;
  } else if (hsh.svl < -WIN_VALUE) {
    if (hsh.svl > -BAN_VALUE) {
      return -MATE_VALUE;
    }
    hsh.svl += pos.nDistance;
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
  hsh = Search.HashTable[pos.zobr.dwKey & (HASH_SIZE - 1)];
  if (hsh.ucDepth > nDepth) {
    return;
  }
  hsh.ucFlag = static_cast<BYTE>(nFlag);
  hsh.ucDepth = static_cast<BYTE>(nDepth);
  if (vl > WIN_VALUE) {
    if (mv == 0 && vl <= BAN_VALUE) {
      return;
    }
    hsh.svl = static_cast<short>(vl + pos.nDistance);
  } else if (vl < -WIN_VALUE) {
    if (mv == 0 && vl >= -BAN_VALUE) {
      return;
    }
    hsh.svl = static_cast<short>(vl - pos.nDistance);
  } else {
    hsh.svl = static_cast<short>(vl);
  }
  hsh.wmv = static_cast<WORD>(mv);
  hsh.dwLock0 = pos.zobr.dwLock0;
  hsh.dwLock1 = pos.zobr.dwLock1;
  Search.HashTable[pos.zobr.dwKey & (HASH_SIZE - 1)] = hsh;
}

static BYTE cucMvvLva[24] = {
    0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 1, 3, 4, 3, 2, 0, 5, 1, 1, 3, 4, 3, 2, 0};

inline int MvvLva(int mv) {
  return (cucMvvLva[pos.ucpcSquares[DST(mv)]] << 3) - cucMvvLva[pos.ucpcSquares[SRC(mv)]];
}

static int CompareMvvLva(const void *lpmv1, const void *lpmv2) {
  return MvvLva(*static_cast<const int *>(lpmv2)) - MvvLva(*static_cast<const int *>(lpmv1));
}

static int CompareHistory(const void *lpmv1, const void *lpmv2) {
  return Search.nHistoryTable[*static_cast<const int *>(lpmv2)] -
         Search.nHistoryTable[*static_cast<const int *>(lpmv1)];
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
    mvKiller1 = Search.mvKillers[pos.nDistance][0];
    mvKiller2 = Search.mvKillers[pos.nDistance][1];
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
  case PHASE_KILLER_1:
    nPhase = PHASE_KILLER_2;
    if (mvKiller1 != mvHash && mvKiller1 != 0 && pos.LegalMove(mvKiller1)) {
      return mvKiller1;
    }
  case PHASE_KILLER_2:
    nPhase = PHASE_GEN_MOVES;
    if (mvKiller2 != mvHash && mvKiller2 != 0 && pos.LegalMove(mvKiller2)) {
      return mvKiller2;
    }
  case PHASE_GEN_MOVES:
    nPhase = PHASE_REST;
    nGenMoves = pos.GenerateMoves(mvs);
    qsort(mvs, static_cast<size_t>(nGenMoves), sizeof(int), CompareHistory);
    nIndex = 0;
  case PHASE_REST:
    while (nIndex < nGenMoves) {
      mv = mvs[nIndex];
      nIndex++;
      if (mv != mvHash && mv != mvKiller1 && mv != mvKiller2) {
        return mv;
      }
    }
  default:
    return 0;
  }
}

inline void SetBestMove(int mv, int nDepth) {
  int *lpmvKillers;
  Search.nHistoryTable[mv] += nDepth * nDepth;
  lpmvKillers = Search.mvKillers[pos.nDistance];
  if (lpmvKillers[0] != mv) {
    lpmvKillers[1] = lpmvKillers[0];
    lpmvKillers[0] = mv;
  }
}

static int SearchQuiesc(int vlAlpha, int vlBeta) {
  int i, nGenMoves;
  int vl, vlBest;
  int mvs[MAX_GEN_MOVES];

  vl = pos.RepStatus();
  if (vl != 0) {
    return pos.RepValue(vl);
  }
  if (pos.nDistance == LIMIT_DEPTH) {
    return pos.Evaluate();
  }
  vlBest = -MATE_VALUE;

  if (pos.InCheck()) {
    nGenMoves = pos.GenerateMoves(mvs);
    qsort(mvs, static_cast<size_t>(nGenMoves), sizeof(int), CompareHistory);
  } else {
    vl = pos.Evaluate();
    if (vl > vlBest) {
      vlBest = vl;
      if (vl >= vlBeta) {
        return vl;
      }
      if (vl > vlAlpha) {
        vlAlpha = vl;
      }
    }
    nGenMoves = pos.GenerateMoves(mvs, GEN_CAPTURE);
    qsort(mvs, static_cast<size_t>(nGenMoves), sizeof(int), CompareMvvLva);
  }

  for (i = 0; i < nGenMoves; i++) {
    if (pos.MakeMove(mvs[i])) {
      vl = -SearchQuiesc(-vlBeta, -vlAlpha);
      pos.UndoMakeMove();
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
  return vlBest == -MATE_VALUE ? pos.nDistance - MATE_VALUE : vlBest;
}

const BOOL NO_NULL = TRUE;

static int SearchFull(int vlAlpha, int vlBeta, int nDepth, BOOL bNoNull = FALSE) {
  int nHashFlag, vl, vlBest;
  int mv, mvBest, mvHash, nNewDepth;
  SortStruct Sort;

  if (nDepth <= 0) {
    return SearchQuiesc(vlAlpha, vlBeta);
  }
  vl = pos.RepStatus();
  if (vl != 0) {
    return pos.RepValue(vl);
  }
  if (pos.nDistance == LIMIT_DEPTH) {
    return pos.Evaluate();
  }
  vl = ProbeHash(vlAlpha, vlBeta, nDepth, mvHash);
  if (vl > -MATE_VALUE) {
    return vl;
  }
  if (!bNoNull && !pos.InCheck() && pos.NullOkay()) {
    pos.NullMove();
    vl = -SearchFull(-vlBeta, 1 - vlBeta, nDepth - NULL_DEPTH - 1, NO_NULL);
    pos.UndoNullMove();
    if (vl >= vlBeta) {
      return vl;
    }
  }
  nHashFlag = HASH_ALPHA;
  vlBest = -MATE_VALUE;
  mvBest = 0;
  Sort.Init(mvHash);
  while ((mv = Sort.Next()) != 0) {
    if (pos.MakeMove(mv)) {
      nNewDepth = pos.InCheck() ? nDepth : nDepth - 1;
      if (vlBest == -MATE_VALUE) {
        vl = -SearchFull(-vlBeta, -vlAlpha, nNewDepth);
      } else {
        vl = -SearchFull(-vlAlpha - 1, -vlAlpha, nNewDepth);
        if (vl > vlAlpha && vl < vlBeta) {
          vl = -SearchFull(-vlBeta, -vlAlpha, nNewDepth);
        }
      }
      pos.UndoMakeMove();
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
    return pos.nDistance - MATE_VALUE;
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
  Sort.Init(Search.mvResult);
  while ((mv = Sort.Next()) != 0) {
    if (pos.MakeMove(mv)) {
      nNewDepth = pos.InCheck() ? nDepth : nDepth - 1;
      if (vlBest == -MATE_VALUE) {
        vl = -SearchFull(-MATE_VALUE, MATE_VALUE, nNewDepth, NO_NULL);
      } else {
        vl = -SearchFull(-vlBest - 1, -vlBest, nNewDepth);
        if (vl > vlBest) {
          vl = -SearchFull(-MATE_VALUE, -vlBest, nNewDepth, NO_NULL);
        }
      }
      pos.UndoMakeMove();
      if (vl > vlBest) {
        vlBest = vl;
        Search.mvResult = mv;
        if (vlBest > -WIN_VALUE && vlBest < WIN_VALUE) {
          vlBest += (std::rand() & RANDOM_MASK) - (std::rand() & RANDOM_MASK);
        }
      }
    }
  }
  RecordHash(HASH_PV, vlBest, nDepth, Search.mvResult);
  SetBestMove(Search.mvResult, nDepth);
  return vlBest;
}

static void SearchMain(void) {
  int i, vl, nGenMoves;
  int mvs[MAX_GEN_MOVES];
  clock_t deadline =
      g_xqwl_sc->t_start + static_cast<clock_t>(g_xqwl_sc->time_limit_ms) * CLOCKS_PER_SEC / 1000;

  std::memset(Search.nHistoryTable, 0, 65536 * sizeof(int));
  std::memset(Search.mvKillers, 0, LIMIT_DEPTH * 2 * sizeof(int));
  std::memset(Search.HashTable, 0, HASH_SIZE * sizeof(HashItem));
  pos.nDistance = 0;

  if (g_xqwl_sc->use_book) {
    Search.mvResult = SearchBook();
    if (Search.mvResult != 0) {
      if (pos.MakeMove(Search.mvResult)) {
        if (pos.RepStatus(3) == 0) {
          pos.UndoMakeMove();
          return;
        }
        pos.UndoMakeMove();
      }
    }
  }

  vl = 0;
  nGenMoves = pos.GenerateMoves(mvs);
  for (i = 0; i < nGenMoves; i++) {
    if (pos.MakeMove(mvs[i])) {
      pos.UndoMakeMove();
      Search.mvResult = mvs[i];
      vl++;
    }
  }
  if (vl == 1) {
    return;
  }

  for (i = 1; i <= LIMIT_DEPTH; i++) {
    vl = SearchRoot(i);
    if (vl > WIN_VALUE || vl < -WIN_VALUE) {
      break;
    }
    if (std::clock() >= deadline) {
      break;
    }
  }
}

/** 在 ``position`` 副本上搜索最佳着法；``tables`` 可跨局复用（开局库等）。 */
inline int XqwlSearchBestMove(PositionStruct &position, XqwlSearchTables &tables, int time_limit_ms,
                              bool use_book = true) {
  if (time_limit_ms < 50) {
    time_limit_ms = 50;
  }
  XqwlSearchCtx ctx{&position, &tables, time_limit_ms, std::clock(), use_book};
  g_xqwl_sc = &ctx;
  tables.mvResult = 0;
  SearchMain();
  const int mv = tables.mvResult;
  g_xqwl_sc = nullptr;
  return mv;
}

#endif // XQWL_SEARCH_INC
