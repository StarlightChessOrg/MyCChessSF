/*
 * Print HalfKAv2_hm ValidBB counts and PS_NB for table generation.
 * Build like pikafish_halfka_dump.cpp (see build_pikafish_halfka_dump.sh).
 */
#include <iostream>

#include "bitboard.h"
#include "types.h"
#include "nnue/features/half_ka_v2_hm.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE::Features;

int main() {
    int cumulative = 0;
    for (Piece pc : HalfKAv2_hm::AllPieces) {
        int n = 0;
        for (Square sq = SQ_A0; sq <= SQ_I9; ++sq)
            if (HalfKAv2_hm::ValidBB[pc] & sq)
                ++n;
        std::cout << int(pc) << " " << n << "\n";
        cumulative += n;
    }
    std::cout << "PS_NB " << cumulative << "\n";
    return 0;
}
