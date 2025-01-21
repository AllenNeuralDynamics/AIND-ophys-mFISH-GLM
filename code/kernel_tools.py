import numpy as np
from copy import copy


# def define_kernels():
#     # TODO: expose this to the user
#     kernels = {
#         'intercept':    {'feature':'intercept',   'type':'continuous',    'length':0,     'offset':0,     'num_weights':None, 'dropout':True, 'text': 'constant value'},
#         'hits':         {'feature':'hit',         'type':'discrete',      'length':1.5,   'offset':0,    'num_weights':None, 'dropout':True, 'text': 'lick to image change'},
#         'misses':       {'feature':'miss',        'type':'discrete',      'length':1.5,   'offset':0,    'num_weights':None, 'dropout':True, 'text': 'no lick to image change'},
#         'passive_change':   {'feature':'passive_change','type':'discrete','length':1.5,   'offset':0,    'num_weights':None, 'dropout':True, 'text': 'passive session image change'},
#         #'hits':         {'feature':'hit',         'type':'discrete',      'length':.75,   'offset':0,    'num_weights':None, 'dropout':True, 'text': 'lick to image change'},
#         #'misses':       {'feature':'miss',        'type':'discrete',      'length':.75,   'offset':0,    'num_weights':None, 'dropout':True, 'text': 'no lick to image change'},
#         #'passive_change':   {'feature':'passive_change','type':'discrete','length':.75,   'offset':0,    'num_weights':None, 'dropout':True, 'text': 'passive session image change'},
#         #'post-hits':    {'feature':'hit',         'type':'discrete',      'length':1.5,   'offset':0.75,    'num_weights':None, 'dropout':True, 'text': 'lick to image change'},
#         #'post-misses':  {'feature':'miss',        'type':'discrete',      'length':1.5,   'offset':0.75,    'num_weights':None, 'dropout':True, 'text': 'no lick to image change'},
#         #'post-passive_change': {'feature':'passive_change','type':'discrete','length':1.5,   'offset':0.75,    'num_weights':None, 'dropout':True, 'text': 'passive session image change'},
#         'omissions':        {'feature':'omissions',   'type':'discrete',  'length':1.5,      'offset':0,     'num_weights':None, 'dropout':True, 'text': 'image was omitted'},
#         #'omissions':        {'feature':'omissions',   'type':'discrete',  'length':0.75,      'offset':0,     'num_weights':None, 'dropout':True, 'text': 'image was omitted'},
#         #'post-omissions':   {'feature':'omissions',   'type':'discrete',  'length':2.25,   'offset':0.75,  'num_weights':None, 'dropout':True, 'text': 'images after omission'},
#         'each-image':   {'feature':'each-image',  'type':'discrete',      'length':0.75,  'offset':0,     'num_weights':None, 'dropout':True, 'text': 'image presentation'},
#         'running':      {'feature':'running',     'type':'continuous',    'length':2,     'offset':-1,    'num_weights':None, 'dropout':True, 'text': 'normalized running speed'},
#         # 'pupil':        {'feature':'pupil',       'type':'continuous',    'length':2,     'offset':-1,    'num_weights':None, 'dropout':True, 'text': 'Z-scored pupil diameter'},
#         'licks':        {'feature':'licks',       'type':'discrete',      'length':2,     'offset':-1,    'num_weights':None, 'dropout':True, 'text': 'mouse lick'},
#         #'false_alarms':     {'feature':'false_alarm',   'type':'discrete','length':5.5,   'offset':-1,    'num_weights':None, 'dropout':True, 'text': 'lick on catch trials'},
#         #'correct_rejects':  {'feature':'correct_reject','type':'discrete','length':5.5,   'offset':-1,    'num_weights':None, 'dropout':True, 'text': 'no lick on catch trials'},
#         #'time':         {'feature':'time',        'type':'continuous',    'length':0,     'offset':0,    'num_weights':None,  'dropout':True, 'text': 'linear ramp from 0 to 1'},
#         #'beh_model':    {'feature':'beh_model',   'type':'continuous',    'length':.5,    'offset':-.25, 'num_weights':None,  'dropout':True, 'text': 'behavioral model weights'},
#         #'lick_bouts':   {'feature':'lick_bouts',  'type':'discrete',      'length':4,     'offset':-2,   'num_weights':None,  'dropout':True, 'text': 'lick bout'},
#         #'lick_model':   {'feature':'lick_model',  'type':'continuous',    'length':2,     'offset':-1,   'num_weights':None,  'dropout':True, 'text': 'lick probability from video'},
#         #'groom_model':  {'feature':'groom_model', 'type':'continuous',    'length':2,     'offset':-1,   'num_weights':None,  'dropout':True, 'text': 'groom probability from video'},
#     }
#     ## add face motion energy PCs
#     # for PC in range(5):
#     #     kernels['face_motion_PC_{}'.format(PC)] = {'feature':'face_motion_PC_{}'.format(PC), 'type':'continuous', 'length':2, 'offset':-1, 'dropout':True, 'text':'PCA from face motion videos'}
#     return kernels


def process_kernels(kernels, run_params, bod):
    kernels = replace_kernels(kernels, bod)
    dropouts = define_dropouts(kernels, run_params)
    run_params['kernels'] = kernels
    run_params['dropouts'] = dropouts
    return run_params


def replace_kernels(kernels, bod):
    '''
        Replaces the 'each-image' kernel with each individual image (not omissions), with the same parameters
    '''
    if ('each-image' in kernels) & ('any-image' in kernels):
        raise Exception('Including both each-image and any-image kernels makes the model unstable')
    if 'each-image' in kernels:
        specs = kernels.pop('each-image')
        stimulus_presentations = bod.stimulus_presentations[~bod.stimulus_presentations.image_name.isna()]
        image_names = np.setdiff1d(stimulus_presentations['image_name'].unique(), 'omitted')
        for index, val in enumerate(image_names):
            kernels[val] = copy(specs)
            kernels[val]['feature'] = val
    if 'each-image_change' in kernels:
        specs = kernels.pop('each-image_change')
        stimulus_presentations = bod.stimulus_presentations[~bod.stimulus_presentations.image_name.isna()]
        image_names = np.sort(stimulus_presentations['image_name'].unique())
        for index, val in enumerate(image_names):
            kernels[val] = copy(specs)
            kernels[val]['feature'] = val
    if 'beh_model' in kernels:
        specs = kernels.pop('beh_model')
        weight_names = ['bias','task0','omissions1','timing1D']
        for index, val in enumerate(weight_names):
            kernels['model_' + str(val)] = copy(specs)
            kernels['model_' + str(val)]['feature'] = 'model_' + str(val)
    return kernels


def define_dropouts(kernels, run_params):
    '''
        Creates a dropout dictionary. Each key is the label for the dropout, and the value is a list of kernels to include
        Creates a dropout for each kernel by removing just that kernel.
        Creates a single-dropout for each kernel by removing all but that kernel
        Also defines nested models
    '''

    # Define full model
    dropouts = {'Full': {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}}

    # if run_params['version_type'] in ['production','standard']:
        # Remove each kernel one-by-one
    for kernel in [kernel for kernel in kernels.keys() if kernels[kernel]['dropout']]:
        dropouts[kernel]={'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
        dropouts[kernel]['kernels'].remove(kernel)
        dropouts[kernel]['dropped_kernels'].append(kernel)

    # Define the nested_models
    dropout_definitions={
        'all-images':           [key for key in list(kernels.keys()) if (key.startswith('im')) & (key[2:].isnumeric())],
        'task':                 [key for key in list(kernels.keys()) if any(keyword in key for keyword in ['hits', 'misses', 'passive_change'])],
        'behavioral':           ['running','pupil','licks'],
        }
    
    if 'post-omissions' in kernels:
        dropout_definitions['all-omissions'] = ['omissions','post-omissions']
    if 'post-hits' in kernels:
        dropout_definitions['all-hits'] =            ['hits','post-hits']
        dropout_definitions['all-misses'] =          ['misses','post-misses']
        dropout_definitions['all-passive_change'] =  ['passive_change','post-passive_change']
        dropout_definitions['post-task'] =           ['post-hits','post-misses','post-passive_change']
        dropout_definitions['task'] =                ['hits','misses','passive_change']
        dropout_definitions['all-task'] =            ['hits','misses','passive_change','post-hits','post-misses','post-passive_change']

    # Add all face_motion_energy individual kernels to behavioral, and as a group model
    # Number of PCs is variable, so we have to treat it differently
    if 'face_motion_PC_0' in kernels:
        dropout_definitions['face_motion_energy'] = [kernel for kernel in list(kernels.keys()) if kernel.startswith('face_motion')] 
        dropout_definitions['behavioral']=dropout_definitions['behavioral'] + dropout_definitions['face_motion_energy']   
    
    # For each nested model, move the appropriate kernels to the dropped_kernel list
    for dropout_name in dropout_definitions:
        dropouts = set_up_dropouts(dropouts, kernels, dropout_name, dropout_definitions[dropout_name])
    
    # Adds single kernel dropouts:
    # if run_params['version_type'] == 'production':
    for drop in [drop for drop in dropouts.keys()]:
        if (drop != 'Full') & (drop != 'intercept'):
            # Make a list of kernels by taking the difference between the kernels in 
            # the full model, and those in the dropout specified by this kernel.
            # This formulation lets us do single kernel dropouts for things like beh_model,
            # or all-images

            kernels = set(dropouts['Full']['kernels'])-set(dropouts[drop]['kernels'])
            kernels.add('intercept') # We always include the intercept
            dropped_kernels = set(dropouts['Full']['kernels']) - kernels
            dropouts['single-'+drop] = {'kernels':list(kernels),'dropped_kernels':list(dropped_kernels),'is_single':True} 
   
    # Check to make sure no kernels got lost in the mix 
    for drop in dropouts.keys():
        assert len(dropouts[drop]['kernels']) + len(dropouts[drop]['dropped_kernels']) == len(dropouts['Full']['kernels']), 'bad length'

    return dropouts

    
def set_up_dropouts(dropouts, kernels, dropout_name, kernel_list):
    '''
        Helper function to define dropouts.
        dropouts,       dictionary of dropout models
        kernels,        dictionary of expanded kernel names
        dropout_name,   name of dropout to be defined
        kernel_list,    list of kernels to be dropped from this nested model
    '''

    dropouts[dropout_name] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}

    for k in kernel_list:
        if k in kernels:
            dropouts[dropout_name]['kernels'].remove(k)
            dropouts[dropout_name]['dropped_kernels'].append(k)
    return dropouts