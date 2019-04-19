import os

def options(options, title='', item_prefixes=[], item_suffices=[], exitable=True, custom_exit_label=''):
    """
    Display a multiple-choices menu with activable options. Each option can be on or off.
    Return the user input
    """
    assert isinstance(options, list)
    assert isinstance(item_prefixes, list)
    assert isinstance(item_suffices, list)

    clear()
    if title != '':
        print(title)
    for i in range(len(options)):
        prefix = item_prefixes[i] if len(item_prefixes) == len(options) else ''
        suffix = item_suffices[i] if len(item_suffices) == len(options) else ''
        print(f'({i}) {prefix}{options[i]}{suffix}')
    if exitable:
        exit_label = custom_exit_label if custom_exit_label != '' else 'Exit' 
        print(f'(x) {exit_label}')
    print()
    return input()


def single_choice(title, labels, callbacks, exitable=False):
    """
    Display a choice to the user. The corresponding callback will be called in case of
    affermative or negative answers.
    :param title: text to display (e.g.: 'What is your favorite color?' )
    :param labels: list of possibile choices to display (e.g.: ['red','green','blue'])
    :param callbacks: list of callback functions to be called in the same order of labels
    :param exitable: whether to exit from the menu without choosing any option.
    Return the callback result, or None if exited without choosing
    """
    assert isinstance(labels, list)
    assert isinstance(callbacks, list)
    assert len(labels) == len(callbacks)
    
    print()
    print(title)
    valid_inp = []
    for i in range(len(labels)):
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
                return fnc() if callable(fnc) else None
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
    return single_choice('Choose a mode:', ['full','local','small'], [lambda: 'full', lambda: 'local', lambda: 'small'], exitable=exitable)