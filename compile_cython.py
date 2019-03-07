from distutils.core import setup
from Cython.Build import cythonize
#run
# python compile_cython.py build_ext --inplace


if __name__ == '__main__':
    setup(name='evaluate_C', ext_modules= cythonize('evaluate_C.pyx'))