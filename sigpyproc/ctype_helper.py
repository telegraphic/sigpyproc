import os
import glob
import sysconfig
import ctypes as C

THIS_DIRPATH   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIRPATH = os.path.abspath(os.path.join(THIS_DIRPATH, '..'))

def load_lib(libname):
    """ Load a shared library .so file in same directory as this module

    args:
        libname (str): name of library.so to load.

    Returns a ctypes.CDLL object
    """
    try:
    	lib = C.CDLL(os.path.join(PARENT_DIRPATH, libname))
    except OSError:
    	libname = libname.split(".so")[0] + sysconfig.get_config_var('EXT_SUFFIX') # PEP 3149
    	lib = C.CDLL(os.path.join(PARENT_DIRPATH, libname))
    return lib