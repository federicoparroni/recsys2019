import os

def options(options, labels, title='', selected_item_prefix=None, selected_item_suffix=None, enable_all=False, custom_exit_label=''):
    """
    Display a multiple-choices menu with activable options. Each option can be on or off.
    Return a list containing the enabled indices.
    options (list): list of element to choose
    labels (list): list of string to display as menu options
    title (str): menu title
    selected_item_prefix (str): prefix to display for the enabled options
    selected_item_suffix (str): suffix to display for the enabled options

    """
    assert isinstance(options, list)
    assert isinstance(labels, list)
    assert len(options) == len(labels)
    selected_item_prefix = selected_item_prefix or '✓ '
    selected_item_suffix = selected_item_suffix or ''
    spacer_prefix = ' ' * len(selected_item_prefix)
    spacer_suffix = ' ' * len(selected_item_suffix)

    ITEMS_COUNT = len(labels)
    VALID_INP = [str(j) for j in range(ITEMS_COUNT)] + ['x']
    enabled = [enable_all] * ITEMS_COUNT

    inp = ''
    while inp != 'x':
        clear()
        if title != '':
            print(title)
        for i, opt in enumerate(labels):
            prefix = selected_item_prefix if enabled[i] else spacer_prefix
            suffix = selected_item_suffix if enabled[i] else spacer_suffix
            print(f'({i}) {prefix}{opt}{suffix}')
        exit_label = custom_exit_label if custom_exit_label != '' else 'Exit' 
        print(f'(x) {exit_label}')
        print()
        
        inp = input()
        if inp in VALID_INP:
            if inp != 'x':
                # inp is a number, enable / disable one option
                idx = int(inp)
                enabled[idx] = not enabled[idx]
    
    return [x for i,x in enumerate(options) if enabled[i]]


def single_choice(title, labels, callbacks=None, exitable=False):
    """
    Display a choice to the user. The corresponding callback will be called in case of
    affermative or negative answers.
    :param title: text to display (e.g.: 'What is your favorite color?' )
    :param labels: list of possibile choices to display (e.g.: ['red','green','blue'])
    :param callbacks: optional list of callback functions or values to be called/returned in the same order of labels
    :param exitable: whether to exit from the menu without choosing any option.
    Return the callback result if specified or the chosen option as string if the callback is a value,
        or None if exited without choosing
    """
    assert isinstance(labels, list)

    callbacks = callbacks or []
    num_callbacks = len(callbacks)
    num_labels = len(labels)
    if num_callbacks < num_labels:
        callbacks.extend([ t for t in labels[num_callbacks:] ])

    print()
    print(title)
    valid_inp = []
    for i in range(num_labels):
        index = str(i+1)
        valid_inp.append(index)
        print(f'({index}) {labels[i]}')
    if exitable:
        print('(x) Exit')
        valid_inp.append('x')

    while(True):
        inp = input()
        if inp in valid_inp:
            if inp == 'x':
                return None
            else:
                idx = int(inp)-1
                fnc = callbacks[idx]
                return fnc() if callable(fnc) else callbacks[idx]
        else:
            print('Wrong choice buddy ;) Retry:')
    


def yesno_choice(title, callback_yes=None, callback_no=None):
    """
    Display a choice to the user. The corresponding callback will be called in case of
    affermative or negative answers.
    :param title: text to display (e.g.: 'Do you want to go to Copenaghen?' )
    :param callback_yes: callback function to be called in case of 'y' answer
    :param callback_no: callback function to be called in case of 'n' answer
    Return the callback result
    """
    
    print()
    print(f'{title} (y/n)')
    valid_inp = ['y','n']
    
    while(True):
        inp = input()
        if inp in valid_inp:
            if inp == 'y':
                if callable(callback_yes):
                    return callback_yes()
                else:
                    return 'y'
            elif inp == 'n':
                if callable(callback_no):
                    return callback_no()
                else:
                    return 'n'
        else:
            print('Wrong choice buddy ;) Retry:')
    

def clear():
    os.system('clear')


def mode_selection(exitable=False):
    """ Quick menu for mode selection. Return 'full', 'local' or 'small'. """
    return single_choice('Choose a mode:', ['full','local','small'], exitable=exitable)


def cluster_selection(exitable=False):
    """ Quick menu for cluster selection. Return the string name of the cluster """
    dir = 'dataset/preprocessed'
    folders = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]
    return single_choice('Choose a cluster:', folders, exitable=exitable)

def checkpoint_selection(checkpoints_dir='saved_models'):
    model_checkpoints = os.listdir(checkpoints_dir)
    checkpoint_path = single_choice('Choose the model checkpoint:', model_checkpoints)

    return os.path.join('saved_models', checkpoint_path)
