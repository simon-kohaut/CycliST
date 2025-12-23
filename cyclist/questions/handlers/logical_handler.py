
def logical_and_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    return inputs[0] and inputs[1]

def logical_or_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    return inputs[0] or inputs[1]

def logical_not_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    return not inputs[0]