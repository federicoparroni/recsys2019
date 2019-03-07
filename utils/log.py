# BLACK: "\033[30m"
# RED: "\033[31m"
# GREEN: "\033[32m"
# YELLOW: "\033[33m"
# BLUE: "\033[34m"
# MAGENTA: "\033[35m"
# CYAN: "\033[36m"
# LIGHTGRAY: "\033[37m"
# GRAY: "\033[90m"
# LIGHTRED: "\033[91m"
# LIGHTGREEN: "\033[92m"
# LIGHTYELLOW: "\033[93m"
# LIGHTBLUE: "\033[94m"
# LIGHTMAGENTA: "\033[95m"
# LIGHTCYAN: "\033[96m"
# WHITE: '\033[97m'
# ENDC = '\033[0m'
# BOLD = '\033[1m'
# UNDERLINE = '\033[4m'

def info(string, end='\n'):
    """
    Print a log message in _BLUE
    """
    print('{}{}{}'.format('\033[34m',string,'\033[0m'), end=end)

def success(string, end='\n'):
    """
    Print a log message in _GREEN
    """
    print('{}{}{}'.format('\033[32m',string,'\033[0m'), end=end)
    
def warning(string, end='\n'):
    """
    Print a log message in _YELLOW
    """
    print('{}{}{}'.format('\033[93m',string,'\033[0m'), end=end)

def error(string, end='\n'):
    """
    Print a log message in _RED
    """
    print('{}{}{}'.format('\033[91m',string,'\033[0m'), end=end)

def progressbar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to print a progress bar

    Parameters
    ----------
    iteration:  int, current iteration
    total:      int, total iterations
    prefix:     str, prefix string
    suffix:     suffix string
    decimals:   int (optional), positive number of decimals in percent complete
    length:     int (optional), length of bar (in characters)
    fill:       int (optional), fill character
    """
    percent = ("{:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('%s [%s] %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:  # print new line on complete
        print('')
