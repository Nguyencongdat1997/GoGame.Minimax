def encode_game_state(state, stone_type):
    txt_state = ''.join([str(y) for x in state for y in x])
    txt_state += str(stone_type)
    return txt_state