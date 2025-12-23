"""
realtionship, we have to create our own verison for handling relationship_seq
rel_seq : [rel_1, ... , rel_{frames}]
"""

#this returns the objects from desired relation. For instance, ["right"][inputs[0]] = 2 , i.e. index of the object on the right of
#index inputs[0]

def relate_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    assert len(side_inputs) == 1
    relation = side_inputs[0]
    return scene_struct['relationships'][relation][inputs[0]]




def relate_existential_handler(scene_struct, inputs, side_inputs):
    """
    gets and object and a relation
    returns all objects that once relate to the initial object with the given relation
    """
    assert len(inputs) == 1
    assert len(side_inputs) == 1
    relation = side_inputs[0]
    result = set()
    
    if 'relationships' in scene_struct: #TODO why is this necessary?
        for frame in scene_struct['relationships'][relation]:
            for object_tuple in scene_struct['relationships'][relation][frame]:
                if object_tuple[1] == inputs[0]:
                    result.add(object_tuple[0])
    
    return list(result)

def relate_universal_handler(scene_struct, inputs, side_inputs):
    """
    gets and object and a relation
    returns all objects that relate to the initial object with the given relation all the time.
    """
    assert len(inputs) == 1
    assert len(side_inputs) == 1
    relation = side_inputs[0]

    result = set(list(range(0, len(scene_struct['objects']))))

    if 'relationships' in scene_struct: #TODO why is this necessary?
        for frame in scene_struct['relationships'][relation]:
            intermediate_set = set()
            for object_tuple in scene_struct['relationships'][relation][frame]:
                if object_tuple[1] == inputs[0]:
                    intermediate_set.add(object_tuple[0])
            result = result.intersection(intermediate_set) # use interesection of all objects per frame to get the ones present in all frames

        return list(result)