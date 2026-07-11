#pragma once

#include <string>

struct PositionStruct;

/** Standard xiangqi FEN from ``PositionStruct`` (red on rank 0 / top row). */
std::string xqwl_position_to_fen(const PositionStruct &pst);
