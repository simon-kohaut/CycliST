
def make_filter_handler(attribute):
    """
    creates a filter handler that filters objects based on a given attribute or cycle.
    """
    def filter_handler(scene_struct, inputs, side_inputs):
        assert len(inputs) == 1
        assert len(side_inputs) == 1
        value = side_inputs[0]
        output = []
        
        # Map attributes to their corresponding cycle keys
        cycle_mapping = ['orbit','motion', 'linear', 'recolor','resize','rotate']
        
        if attribute in cycle_mapping:
            is_false = (value == 'False')
            
            for idx in inputs[0]:
                obj = scene_struct['objects'][idx]
                if 'cycles' not in obj:
                    continue
                    
                has_cycle = attribute in obj['cycles']
                
                # For motion, check if either linear OR orbit cycles exist
                if attribute == 'motion':
                    has_cycle = 'linear' in obj['cycles'] or 'orbit' in obj['cycles']
                
                # Add to output based on filter condition
                if (is_false and not has_cycle) or (not is_false and has_cycle):
                    output.append(idx)
        
        else:
            # Handle regular attribute filtering
            for idx in inputs[0]:
                atr = scene_struct['objects'][idx][attribute]
                if value == atr:
                    output.append(idx)
        
        return output
    return filter_handler


#filter handler for non-static objects

def make_filter_existential_handler(attribute):
    """
    creates a filter handler that filters objects based on the attribute
    and the value given in side_inputs.
    The attribute can be 'color', 'size', 'mesh', 'material', 'objectcategory'
    """

    def filter_existential_handler(scene_struct, inputs, side_inputs):
        nonlocal attribute
        assert len(inputs) == 1
        assert len(side_inputs) == 1
        value = side_inputs[0]
        output = []
        if attribute == 'color':
            for idx in inputs[0]:
                atr = set()
                atr.add(scene_struct['objects'][idx]['color'])

                if 'cycles' in scene_struct['objects'][idx]:
                    if 'change_color' in scene_struct['objects'][idx]:
                        atr.add(True)
                    else:
                        atr.add(False)
                if value in atr : #
                    output.append(idx)
                    
        elif attribute == 'size':
            for idx in inputs[0]:
                atr = scene_struct['objects'][idx]['size']
                enlarge = False
                if 'cycles' in scene_struct['objects'][idx]:
                    if 'enlarge' in scene_struct['objects'][idx]:
                        enlarge = True
                # enlarge = scene_struct['objects'][idx]['enlarge']
                if value == atr or enlarge == True : #
                    output.append(idx)
        else:
            if attribute == 'mesh':
                attribute = 'mesh' #TODO
            for idx in inputs[0]:
                atr = scene_struct['objects'][idx][attribute]
                if value == atr:  #
                    output.append(idx)
        return output
    return filter_existential_handler

def make_filter_universal_handler(attribute):
    def filter_universal_handler(scene_struct, inputs, side_inputs):
        """
        checks if the attribute is present and no change cycle affecting the attribute is present
        """
        
        assert len(inputs) == 1
        assert len(side_inputs) == 1
        value = side_inputs[0]
        output = []
        if attribute == 'color':
            for idx in inputs[0]:
                atr = scene_struct['objects'][idx]['color']
                if value == atr and scene_struct['objects'][idx]['change_color'] is None:
                    output.append(idx)
        elif attribute == 'size':
            for idx in inputs[0]:
                atr = scene_struct['objects'][idx]['size']
                if value == atr and scene_struct['objects'][idx]['enlarge'] is None: #
                    output.append(idx)
        else:
            for idx in inputs[0]:
                atr = scene_struct['objects'][idx][attribute]
                if value == atr:  #
                    output.append(idx)
        return output
    return filter_universal_handler