from mycchess_sf.bridge_codec import (
    bridge4_to_iccs,
    iccs_to_bridge4,
    iccs_to_computer_token,
    move_id_to_iccs,
)


def test_bridge4_roundtrip() -> None:
    iccs = "77-77"
    code = iccs_to_bridge4(iccs)
    assert code == "7777"
    assert bridge4_to_iccs(code) == iccs
    assert iccs_to_computer_token(iccs) == code


def test_move_id_decode() -> None:
    assert move_id_to_iccs(7071) == "70-71"
