# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import json, os, math
import numpy as np
from collections import defaultdict
from handlers.set_handler import *
from handlers.relate_handler import *
from handlers.filter_handler import *
from handlers.query_handler import *
from handlers.equality_handler import *
from handlers.logical_handler import *
from handlers.same_handler import *

"""
Utilities for working with function program representations of questions.

Some of the metadata about what question node types are available etc are stored
in a JSON metadata file.
"""



# Register all of the answering handlers here.
# TODO maybe this would be cleaner with a function decorator that takes
# care of registration? Not sure. Also what if we want to reuse the same engine
# for different sets of node types?
execute_handlers = {
  # scene handler returns all objects in the scene
  'scene': scene_handler, # DONE

  # filters object by attribute and cycle
  #Existential filters
  #TODO Filters are used in the unique unrollment. How should i use them here?
  'filter_color': make_filter_existential_handler(attribute='color'),
  'filter_shape': make_filter_existential_handler(attribute='mesh'),
  'filter_material': make_filter_existential_handler(attribute='material'),
  'filter_size': make_filter_existential_handler(attribute='size'),
  
  #Existential filters
  'filter_color_universal': make_filter_universal_handler(attribute='color'),
  'filter_size_universal': make_filter_universal_handler(attribute='size'),
  
  'filter_objectcategory': make_filter_handler('objectcategory'), # WHAT IS THIS?

  # filter objects for cycles cycles
  'filter_enlarge': make_filter_handler('resize'),
  'filter_orbit': make_filter_handler('orbit'),
  'filter_linear':make_filter_handler('linear'),
  'filter_motion':make_filter_handler('motion'),
  'filter_rotate':make_filter_handler('rotate'),
  'filter_change_color': make_filter_handler('recolor'),


  #filter cycle period
  'filter_enlarge_period': make_filter_handler('enlarge_period'), # not implemented
  'filter_motion_period': make_filter_handler('orbit_period'), # not implemented
  'filter_linear_period': make_filter_handler('linear_period'), # not implemented
  'filter_color_period': make_filter_handler('color_period'), # not implemented

  # relate
  'relate_existential': relate_existential_handler,
  'relate_universal': relate_universal_handler,


  # set operations
  'except': except_handler, # set difference
  'union': union_handler, # set union
  'intersect': intersect_handler, # set intersection
  'include':include_handler, #set inclusion
  'unique': unique_handler, # check if the input is unique, i.e. has only one element
  'count': count_handler, #count the number of objects in a list
  'exist': exist_handler, # check if the input list is not empty

  #logic
  'logical_and':logical_and_handler,
  'logical_or':logical_or_handler,
  'logical_not':logical_or_handler,


  #query attributes
  'query_color': make_query_handler('color'), #asks for the color of an object, INVALID if color changes
  'query_size': make_query_handler('size'), #asks for the size of an object, INVALID if size changes
  'query_shape': make_query_handler('shape'), #asks for the shape of an object
  'query_material': make_query_handler('material'), #asks for the material of an object

  #query special cyclic properties
  'query_color_initial':query_color_initial, #ask about the initial color of an object which is changing color
  'query_color_final': query_color_final, # ask about the color of an object it is changing into
  'query_size_initial': query_size_initial, #ask about the initial size of an object which is changing size
  'query_size_final': query_size_final, # ask about the size of an object it is changing into
  
  'query_orbit': make_query_handler('orbit'), # asks about an object which is orbited


  #query cyclic periods and passes
  #TODO do the first two work? Are they used anywhere?
  'query_enlarge_period': make_query_handler('enlarge_period'), #ask about the <cycle> period of an object in frames
  'query_color_period': make_query_handler('color_period'),
  'query_motion_period': query_motion_period_handler,
  'query_linear_period': query_linear_period_handler,
  'query_orbit_period': query_orbit_period_handler,
  'query_linear_passes':query_linear_passes_handler,
  'query_orbit_passes': query_orbit_passes_handler, # asks about the number of cycles an object has gone through in a video


  # equalities for static attributes
  'equal_shape': equal_handler,
  'equal_material': equal_handler,
  'equal_object': equal_handler,


  #equalities at different points in time with overlapping periods
  'equal_color_existential': equal_color_existential_handler,
  'equal_color_universal': equal_color_universal_handler,
  'equal_size_existential': equal_size_existential_handler,
  'equal_size_universal': equal_size_universal_handler,

  #equal cyclic
  'equal_enlarge': equal_handler, #Never used
  'equal_change_color': equal_handler, #Never used
  'equal_enlarge_period': equal_handler, #Never used
  'equal_motion_period': equal_handler, #Never used
  'equal_color_period': equal_handler, #Never used

  # (in)equalities for integers
  'less_than': less_than_handler,
  'greater_than': greater_than_handler,
  'equal_integer': equal_handler,


  # same for cyclic attributes
  'same_color': make_same_attr_universal_handler('color'),
  'same_size': make_same_attr_universal_handler('size'),
  # same for static attributes
  'same_shape': make_same_static_attr_handler('shape'),
  'same_material': make_same_static_attr_handler('material'),

  # TODO same cycle periods?
  'same_enlarge_period': make_same_static_attr_handler('enlarge_period'),
  'same_motion_period': make_same_static_attr_handler('motion_period'),
  'same_color_period': make_same_static_attr_handler('color_period'),

  # We can check this by filtering for cycles and then checking if Exists is true
  # #action exist
  # 'enlarge_exist':make_cyclic_action_exist_handler('enlarge'),
  # 'rotate_exist':make_cyclic_action_exist_handler('rotate'),
  # 'linear_exist':make_cyclic_action_exist_handler('linear'),
  # 'color_exist':make_cyclic_action_exist_handler('color'),
  # 'orbit_eixst':make_cyclic_action_exist_handler('orbit'),
}


def answer_question(question, metadata, scene_struct, all_outputs=False,
                    cache_outputs=True):
    """
    Use structured scene information to answer a structured question. Most of the
    heavy lifting is done by the execute handlers defined above.

    We cache node outputs in the node itself; this gives a nontrivial speedup
    when we want to answer many questions that share nodes on the same scene
    (such as during question-generation DFS). This will NOT work if the same
    nodes are executed on different scenes.
    
    @param cache_outputs: Boolean, wherever to store intermediate node outputs
    @param all outputs: Boolean indicating if only the last or all outputs should be returned
    """
    all_input_types, all_output_types = [], []
    node_outputs = []
    for node in question['nodes']:
        
        # if the node output is already computed, use it
        if cache_outputs and '_output' in node: #
            node_output = node['_output']
        #else compute it
        else: 
            node_type = node['type']
            
            #sanity check if node type has a known function
            msg = 'Could not find handler for "%s"' % node_type
            assert node_type in execute_handlers, msg
            
            # obtain outputs for the node given inputs and side inputs
            handler = execute_handlers[node_type]
            node_inputs = [node_outputs[idx] for idx in node['inputs']]
            side_inputs = node.get('side_inputs', [])
            node_output = handler(scene_struct, node_inputs, side_inputs)
            
            if cache_outputs:
                node['_output'] = node_output
            # if verbose:
            #     print(node['type'], "IN:",node_inputs, "+", side_inputs," = ", node_output)
        node_outputs.append(node_output)
        if node_output == '__INVALID__':
            break

    if all_outputs:
        return node_outputs
    else:
        return node_outputs[-1]


def insert_scene_node(nodes, idx):
    # First make a shallow-ish copy of the input
    new_nodes = []
    for node in nodes:
        new_node = {
          'type': node['type'],
          'inputs': node['inputs'],
        }
        if 'side_inputs' in node:
            new_node['side_inputs'] = node['side_inputs']
        new_nodes.append(new_node)

    # Replace the specified index with a scene node
    new_nodes[idx] = {'type': 'scene', 'inputs': []}

    # Search backwards from the last node to see which nodes are actually used
    output_used = [False] * len(new_nodes)
    idxs_to_check = [len(new_nodes) - 1]
    while idxs_to_check:
        cur_idx = idxs_to_check.pop()
        output_used[cur_idx] = True
        idxs_to_check.extend(new_nodes[cur_idx]['inputs'])

    # Iterate through nodes, keeping only those whose output is used;
    # at the same time build up a mapping from old idxs to new idxs
    old_idx_to_new_idx = {}
    new_nodes_trimmed = []
    for old_idx, node in enumerate(new_nodes):
        if output_used[old_idx]:
            new_idx = len(new_nodes_trimmed)
            new_nodes_trimmed.append(node)
            old_idx_to_new_idx[old_idx] = new_idx

    # Finally go through the list of trimmed nodes and change the inputs
    for node in new_nodes_trimmed:
        new_inputs = []
        for old_idx in node['inputs']:
            new_inputs.append(old_idx_to_new_idx[old_idx])
        node['inputs'] = new_inputs

    return new_nodes_trimmed


def is_degenerate(question, metadata, scene_struct, answer=None, verbose=False):
    """
    A question is degenerate if replacing any of its relate nodes with a scene
    node results in a question with the same answer.
    """
    if answer is None:
        answer = answer_question(question, metadata, scene_struct)

    for idx, node in enumerate(question['nodes']):
        if node['type'] == 'relate':
            new_question = {
              'nodes': insert_scene_node(question['nodes'], idx)
            }
            new_answer = answer_question(new_question, metadata, scene_struct)
            if verbose:
                print('here is truncated question:')
                for i, n in enumerate(new_question['nodes']):
                    name = n['type']
                    if 'side_inputs' in n:
                        name = '%s[%s]' % (name, n['side_inputs'][0])
                    print(i, name, n['_output'])
                print('new answer is: ', new_answer)

            if new_answer == answer:
                return True

    return False