
# functions to obtain objects that share a similar attribute as an input object

# obj with same static attribute or cycle period
def make_same_static_attr_handler(attribute):
    def same_static_attr_handler(scene_struct, inputs, side_inputs):
        cache_key = '_same_%s' % attribute
        if cache_key not in scene_struct:
            cache = {}
            for i, obj1 in enumerate(scene_struct['objects']):
                same = []
                for j, obj2 in enumerate(scene_struct['objects']):
                    if i != j and obj1[attribute] == obj2[attribute]:
                        same.append(j)
                cache[i] = same
            scene_struct[cache_key] = cache

        cache = scene_struct[cache_key]
        assert len(inputs) == 1
        assert len(side_inputs) == 0
        return cache[inputs[0]]
    return same_static_attr_handler


# object with same non-static attribute (color,size) that is 
def make_same_attr_universal_handler(attribute):
    def same_attr_universal_handler(scene_struct, inputs, side_inputs):
        """
        receives an object and returns all object that share the instantiated property with the object.
        """
        
        if attribute == 'orbit':
            return []
        else:
            cache_key = '_same_%s' % attribute
            if cache_key not in scene_struct:
                cache = {}
                for i, obj1 in enumerate(scene_struct['objects']):
                    same = []
                    for j, obj2 in enumerate(scene_struct['objects']):
                        if attribute == 'color':
                            if i != j and (obj1[attribute] == obj2[attribute] or obj1['change_color']==obj2[attribute] or
                                           obj1[attribute] == obj2['change_color'] or obj1['change_color'] == obj2['change_color']):
                                same.append(j)
                        elif attribute == 'size':
                            if i != j and (obj1[attribute] == obj2[attribute] or obj1['enlarge']  or obj2['enlarge']):
                                same.append(j)
                        else:
                            if i != j and obj1[attribute] == obj2[attribute]:
                                same.append(j)
                    cache[i] = same
                scene_struct[cache_key] = cache
            cache = scene_struct[cache_key]
            assert len(inputs) == 1
            assert len(side_inputs) == 0
            return cache[inputs[0]]
    return same_attr_universal_handler