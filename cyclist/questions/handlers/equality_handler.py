
def equal_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    return inputs[0] == inputs[1]


def obtain_destination_color(obj):
    # the first color in the cycle state should match the initial color
    assert(obj['cycles']['recolor']['states'][0]['color'] == obj['color'])
    # the second is the color it transitions into
    return obj['cycles']['recolor']['states'][1]['color']


# 
def equal_color_existential_handler(scene_struct, inputs, side_inputs):
    """
    checks if the two objects have the same color at one point in the video
    """    
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    obj1 = scene_struct['objects'][inputs[0]]
    obj2 = scene_struct['objects'][inputs[1]]
    match = False
    
    # if both objects have the same (starting) color
    if obj1['color'] == obj2['color'] :
        match = True
    #if both objects dont have a color change and the colors are also not the same stop here
    elif obj1['change_color'] is None and obj2['change_color'] is None:
        match = False
        
    # they dont have the same starting color but at least on color is changing
    else:
        # One color is changing and its changing into the same color as the other object
        if obj1['change_color'] and obj2['change_color'] is None:
            obj1_change_color = obtain_destination_color(obj1)
            if obj2['color'] == obj1_change_color:
                match = True
                
        elif obj1['change_color'] is None and obj2['change_color']:
            obj2_change_color = obtain_destination_color(obj2)
            if obj1['color'] == obj2_change_color:
                match = True
                
        # both colors are changing and they dont have the same starting color
        else:
            obj1_change_color = obtain_destination_color(obj1)
            obj2_change_color = obtain_destination_color(obj2)
            
            #both colors change into the same color - they will match at the end of the video independent of the frequency
            if obj1_change_color == obj2_change_color:
                match = True
            
            #the colors change into the same starting color but inversly. We cannot make a statement here as they color transitions are hard
            if obj1_change_color == obj2['color'] or obj2_change_color == obj1['color']:
                return '__INVALID__'

    return match


def equal_size_existential_handler(scene_struct, inputs, side_inputs):
    # TODO update
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    obj1 = scene_struct['objects'][inputs[0]]
    obj2 = scene_struct['objects'][inputs[1]]

    match = False    
    # if both objects have the same (starting) size
    if obj1['size'] == obj2['size'] :
        match = True
    #if both objects dont have a size change and the sizes are also not the same stop here
    elif obj1['enlarge'] is None and obj2['enlarge'] is None:
        match = False
        
    # they dont have the same starting size but at least on object is enlarging/shrinking
    else:
        # One size is changing
        if obj1['enlarge'] and obj2['enlarge'] is None:
            match = True
                
        elif obj1['enlarge'] is None and obj2['enlarge']:
            match = True
                
        # both objects enlarge/shrink and they start with different sizes
        else:
            return '__INVALID__'
    return match


def equal_color_universal_handler(scene_struct, inputs, side_inputs):
    """
    checks if the two objects have the same color throughout the video
    This is true when both objects have the same color and are static
    OR if they both start with the same color, change into the same color and have the same period.
    """
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    obj1 = scene_struct['objects'][inputs[0]]
    obj2 = scene_struct['objects'][inputs[1]]
    
    # if both objects have the same (starting) color
    if obj1['color'] == obj2['color']:
        
        # if both objects are static
        if obj1['change_color'] is None and obj2['change_color'] is None:
            return True
        #if both objects are color changing
        elif obj1['change_color'] and obj2['change_color']:
            obj1_change_color = obtain_destination_color(obj1)
            obj2_change_color = obtain_destination_color(obj2)
            
            if obj1_change_color == obj2_change_color and obj1['cycles']['recolor']['period'] == obj2['cycles']['recolor']['period']:
                return True
                
    return False





def equal_size_universal_handler(scene_struct, inputs, side_inputs):
    """
    checks if the two objects have equal size throughout the video
    """
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    obj1 = scene_struct['objects'][inputs[0]]
    obj2 = scene_struct['objects'][inputs[1]]
    
    if obj1['size'] == obj2['size']:
        # if both objects are static
        if obj1['enlarge'] is None and obj2['enlarge'] is None:
            return True
        #if both objects are changing size with the same period
        elif obj1['enlarge'] and obj2['enlarge']:
            if obj1['cycles']['resize']['period'] == obj2['cycles']['resize']['period']:
                return True
    
    return False                    





                    
def less_than_handler(scene_struct, inputs, side_inputs):
    """
    returns True if the first input is less than the second, false otherwise
    """
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    return inputs[0] < inputs[1]


def greater_than_handler(scene_struct, inputs, side_inputs):
    """
    returns True if the first input is greater than the second, false otherwise
    """
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    return inputs[0] > inputs[1]


def period_match(start_frame1, p1, start_frame2, p2, total_frames):
    key_frames1 = [i for i in range(start_frame1, total_frames+1, p1)]
    key_frames2 = [i for i in range(start_frame2, total_frames+1, p2)]
    match = False
    for i in key_frames1:
        if i in key_frames2:
            match = True
    return match