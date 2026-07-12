/*
 * Dump HalfKAv2_hm active indices for a FEN (Pikafish reference for MyCChessSF tests).
 * Standalone: only needs half_ka_v2_hm.h (no Position / movegen link).
 */
#include <cctype>
#include <iostream>
#include <sstream>
#include <string>

#include "nnue/nnue_common.h"
#include "nnue/features/half_ka_v2_hm.h"
#include "types.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE::Features;

namespace {

constexpr auto PSQOffsets = [] {
    int cumulativeOffset = 0;
    MultiArray<u16, PIECE_NB, SQUARE_NB> offsets{};
    for (Piece pc : HalfKAv2_hm::AllPieces)
        for (Square sq = SQ_A0; sq <= SQ_I9; ++sq)
            if (HalfKAv2_hm::ValidBB[pc] & sq)
                offsets[pc][sq] = cumulativeOffset++;
    return offsets;
}();

constexpr auto AttackBucket = [] {
    MultiArray<int, 3, 3, 3> v{};
    for (u8 rook = 0; rook <= 2; ++rook)
        for (u8 knight = 0; knight <= 2; ++knight)
            for (u8 cannon = 0; cannon <= 2; ++cannon)
                v[rook][knight][cannon] = bool(rook) * 2 + bool(knight + cannon);
    return v;
}();

struct FenBoard {
    std::array<Piece, SQUARE_NB> board{};
    int pieceCount[PIECE_NB]{};
    u64 midEncoding[COLOR_NB]{HalfKAv2_hm::BalanceEncoding, HalfKAv2_hm::BalanceEncoding};
    Color sideToMove = WHITE;

    void put_piece(Piece pc, Square sq) {
        board[sq] = pc;
        pieceCount[pc]++;
        midEncoding[color_of(pc)] += HalfKAv2_hm::MidMirrorEncoding[pc][sq];
    }

    template<PieceType Pt>
    int count(Color c) const {
        return pieceCount[make_piece(c, Pt)];
    }

    Square king_square(Color c) const {
        return c == WHITE ? lsb(u64(pieces(KING)))
                          : Square(64 + lsb(u64(pieces(KING) >> 64)));
    }

    Bitboard pieces(PieceType pt) const {
        Bitboard b = 0;
        for (Square sq = SQ_A0; sq <= SQ_I9; ++sq)
            if (type_of(board[sq]) == pt)
                b |= sq;
        return b;
    }

    u64 mid_encoding(Color c) const { return midEncoding[c]; }

    bool set(const std::string& fenStr, std::string& err) {
        board.fill(NO_PIECE);
        std::fill(std::begin(pieceCount), std::end(pieceCount), 0);
        midEncoding[WHITE] = midEncoding[BLACK] = HalfKAv2_hm::BalanceEncoding;

        std::istringstream ss(fenStr);
        int file = FILE_A;
        int rank = RANK_9;
        char token;
        ss >> std::noskipws;

        while (ss >> token) {
            if (std::isspace(static_cast<unsigned char>(token)))
                break;
            if (std::isdigit(static_cast<unsigned char>(token))) {
                file += token - '0';
            } else if (token == '/') {
                --rank;
                file = FILE_A;
            } else {
                constexpr std::string_view PieceToChar(" RACPNBK racpnbk");
                const auto idx = PieceToChar.find(token);
                if (idx == std::string_view::npos) {
                    err = "invalid piece";
                    return false;
                }
                put_piece(Piece(idx), make_square(File(file), Rank(rank)));
                ++file;
            }
        }

        char stm;
        if (!(ss >> stm) || (stm != 'w' && stm != 'b')) {
            err = "invalid side to move";
            return false;
        }
        sideToMove = stm == 'w' ? WHITE : BLACK;
        return true;
    }
};

bool requires_mid_mirror(const FenBoard& pos, Color c) {
    return ((1ULL << 63) & pos.mid_encoding(c) & pos.mid_encoding(~c))
        && (pos.mid_encoding(c) < HalfKAv2_hm::BalanceEncoding
            || (pos.mid_encoding(c) == HalfKAv2_hm::BalanceEncoding
                && pos.mid_encoding(~c) < HalfKAv2_hm::BalanceEncoding));
}

int make_attack_bucket(const FenBoard& pos, Color c) {
    return AttackBucket[pos.count<ROOK>(c)][pos.count<KNIGHT>(c)][pos.count<CANNON>(c)];
}

std::tuple<int, bool, int> make_feature_bucket(Color perspective, const FenBoard& pos) {
    const Square ksq = pos.king_square(perspective);
    const Square oksq = pos.king_square(~perspective);
    auto [king_bucket, mirror] =
        HalfKAv2_hm::KingBuckets[ksq][oksq][requires_mid_mirror(pos, perspective)];
    const auto attack_bucket = make_attack_bucket(pos, perspective);
    const auto bucket = king_bucket * HalfKAv2_hm::AttackBucketNB + attack_bucket;
    return {bucket, mirror, attack_bucket};
}

int make_index(Color perspective, Square s, Piece pc, int bucket, bool mirror) {
    s = Square(HalfKAv2_hm::IndexMap[mirror][perspective == BLACK][s]);
    if (perspective == BLACK)
        pc = ~pc;
    return PSQOffsets[pc][s] + HalfKAv2_hm::PS_NB * bucket;
}

void dump_perspective(const FenBoard& pos, Color perspective) {
    const auto [bucket, mirror, attack_bucket] = make_feature_bucket(perspective, pos);
    std::cout << "perspective=" << (perspective == WHITE ? "w" : "b") << "\n";
    std::cout << "bucket=" << bucket << " mirror=" << mirror << " attack_bucket=" << attack_bucket
              << " mid_mirror=" << requires_mid_mirror(pos, perspective) << "\n";
    std::cout << "king=" << int(pos.king_square(perspective)) << " oking="
              << int(pos.king_square(~perspective)) << "\n";

    std::cout << "indices:";
    bool first = true;
    for (Square sq = SQ_A0; sq <= SQ_I9; ++sq) {
        const Piece pc = pos.board[sq];
        if (pc == NO_PIECE)
            continue;
        if (!(HalfKAv2_hm::ValidBB[pc] & sq))
            continue;
        if (!first)
            std::cout << ' ';
        first = false;
        std::cout << make_index(perspective, sq, pc, bucket, mirror);
    }
    std::cout << "\n";
}

}  // namespace

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty())
            continue;
        FenBoard pos;
        std::string err;
        if (!pos.set(line, err)) {
            std::cout << "ERROR " << err << "\n";
            continue;
        }
        std::cout << "fen=" << line << "\n";
        dump_perspective(pos, pos.sideToMove);
        dump_perspective(pos, ~pos.sideToMove);
        std::cout << "END\n";
    }
    return 0;
}
