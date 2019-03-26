import os

def options(options, title='', item_prefixes=[], item_suffices=[], exitable=True, custom_exit_label=''):
    """
    Display a multiple-choices menu.
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


def clear():
    os.system('clear')