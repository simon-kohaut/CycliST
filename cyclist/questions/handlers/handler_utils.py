



def print_scene_objects(scene_struct):
    for oidx in range(len(scene_struct['objects'])):
        print(oidx, scene_struct['objects'][oidx]['color'],
              scene_struct['objects'][oidx]['size'],
              scene_struct['objects'][oidx]['material'],
              scene_struct['objects'][oidx]['mesh'],
              scene_struct['objects'][oidx]['cycles'].keys() if 'cycles' in scene_struct['objects'][oidx] else {})



