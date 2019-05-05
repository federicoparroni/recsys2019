import numpy as np

def logarithmic(x, max_value, min_value=0):
    """ Scale x (in the range [min_value, max_value]) to be in a logarithmic scale between 0 and 1 """
    x = np.interp(x, [min_value, max_value], [0,1])
    log2 = np.log(2)
    return np.log(1 + x) / log2


# if __name__ == "__main__":
    
#     test1 = np.array([0,1,10,20,30,40,50])
#     print(logarithmic(test1, max_value=50))
#     print()
#     test2 = np.array([-20,1,10,20,30,40,50])
#     print(logarithmic(test2, min_value=-20, max_value=50))
