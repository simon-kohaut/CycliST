# Handlers for answering questions. Each handler receives the scene structure
# that was output from Blender, the node, and a list of values that were output
# from each of the node's inputs; the handler should return the computed output
# value from this node.

# list of indices
def scene_handler(scene_struct, inputs, side_inputs):
    # Just return all objects in the scene
    return list(range(len(scene_struct['objects'])))



# exception
def except_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    return sorted(list(set(inputs[0]) - set(inputs[1])))
# union
def union_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    return sorted(list(set(inputs[0]) | set(inputs[1])))

#intersection
def intersect_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    return sorted(list(set(inputs[0]) & set(inputs[1])))

#inclusion
def include_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    
    if inputs[1] is None:
        return "__INVALID__"
    return inputs[0] in inputs[1]


#existential
def exist_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    return len(inputs[0]) > 0

def make_cyclic_action_exist_handler(attribute):

    def cyclic_action_exist_handler(scene_struct, inputs, side_inputs):
        assert len(inputs) == 1
        assert len(side_inputs) == 0
        attr = attribute + "_period"
        return scene_struct['objects'][inputs[0]][attr] > 0
    return cyclic_action_exist_handler

#count
def count_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    return len(inputs[0])


# return the index of the unique one. [[6]] -> 6
def unique_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    if len(inputs[0]) != 1:
        return '__INVALID__'
    return inputs[0][0]

