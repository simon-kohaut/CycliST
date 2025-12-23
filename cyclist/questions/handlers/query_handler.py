

def make_query_handler(attribute):
    # instantiate a query handler given an attribute
    
    def query_handler(scene_struct, inputs, side_inputs):
        # return the value of a queried attribute
        # e.g. what is the 'size' of obj 2  - large
        
        assert len(inputs) == 1
        assert len(side_inputs) == 0
        idx = int(inputs[0])
        obj = scene_struct['objects'][idx]
        assert attribute in obj
        val = obj[attribute]
    
        # we cannot ask the color of an object which is changing color
        if attribute == 'color' and 'cycles' in obj and 'recolor' in obj['cycles']:
            return '__INVALID__'
    
        # we cannot ask the size of an object which is changing size
        if attribute =='size' and 'cycles' in obj and 'resize' in obj['cycles']:
            return '__INVALID__'
        
        if type(val) == list and len(val) != 1:
            return '__INVALID__'
        elif type(val) == list and len(val) == 1:
            return val[0]
        else:
            return val
    return query_handler


def query_color_initial(scene_struct, inputs, side_inputs):
    # queries for the initial color of an object
    
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    idx = inputs[0]
    return scene_struct['objects'][idx]['color']

def query_size_initial(scene_struct, inputs, side_inputs):
    # queries for the initial size of an object
    
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    idx = inputs[0]
    return scene_struct['objects'][idx]['size']

def query_color_final(scene_struct, inputs, side_inputs):
    # queries for the color an object is changing into
    
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    idx = inputs[0]
    obj = scene_struct['objects'][idx]
    if 'cycles' in obj and 'recolor' in obj['cycles']:
        return obj['cycles']['recolor']['states'][1]['color']
    else:
        return '__INVALID__'

def query_size_final(scene_struct, inputs, side_inputs):
    # queries for the size an object is changing into
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    idx = inputs[0]
    obj = scene_struct['objects'][idx]
    if 'cycles' in obj and 'resize' in obj['cycles']:
        if obj['cycles']['resize']['states'][1]['factor'] < 1:
            return "small"
        else:
            return "large"
    else:
        return '__INVALID__'


#QUERY Motion/Oribt/Linear period
def query_motion_period_handler(scene_struct, inputs, side_inputs):
    #queries for the motion period of a moving object
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    idx = inputs[0]
    obj = scene_struct['objects'][idx]

    if 'orbit' in scene_struct['objects'][idx]['cycles']:
        return obj['cycles']['orbit']['period']
    if 'linear' in scene_struct['objects'][idx]['cycles']:
        return obj['cycles']['linear']['period']


def query_orbit_period_handler(scene_struct, inputs, side_inputs):
    # queries for the orbit period of a moving object
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    idx = inputs[0]
    obj = scene_struct['objects'][idx]
    if 'orbit' in scene_struct['objects'][idx]['cycles']:
        return obj['cycles']['orbit']['period']
    else:
        return '__INVALID__'
    
    
def query_linear_period_handler(scene_struct, inputs, side_inputs):
    # queries for the orbit period of a moving object 
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    idx = inputs[0]
    obj = scene_struct['objects'][idx]
    if 'linear' in scene_struct['objects'][idx]['cycles']:
        return obj['cycles']['linear']['period']
    else:
        return '__INVALID__'


#QUERY Oribt/Linear passes
def query_orbit_passes_handler(scene_struct, inputs, side_inputs):
    #queries for the orbit passes of a moving object

    assert len(inputs) == 1
    assert len(side_inputs) == 0
    idx = inputs[0]
    obj = scene_struct['objects'][idx]
    if 'orbit' in scene_struct['objects'][idx]['cycles']:

        total_frames = scene_struct['fps']*scene_struct['duration']
        num_orbits = total_frames / obj['cycles']['orbit']['period']
        #check if the number of orbits is an integer
        if num_orbits - int(num_orbits) == 0:
            return int(num_orbits)
        else:
            return '__INVALID__'
    else:
        return '__INVALID__'


def query_linear_passes_handler(scene_struct, inputs, side_inputs):
    #queries for the linear passes of a moving object
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    idx = inputs[0]
    obj = scene_struct['objects'][idx]
    if 'linear' in scene_struct['objects'][idx]['cycles']:

        total_frames = scene_struct['fps']*scene_struct['duration']
        num_linear = total_frames / obj['cycles']['linear']['period']
        #check if the number of orbits is an integer
        if num_linear - int(num_linear) == 0:
            return int(num_linear)
        else:
            return '__INVALID__'
    else:
        return '__INVALID__'