# ==============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import sys
from importlib.abc import MetaPathFinder, Loader
import importlib


class MyLoader(Loader):
    def module_repr(self, my_module):
        return repr(my_module)

    def load_module(self, fullname):
        old_name = fullname
        top_level_name = fullname.split(".")[0]
        # Change the name to be new package name
        if top_level_name == 'snpe' or fullname == 'snpe':
            fullname = fullname.replace("snpe", "qti.aisw", 1)
        my_module = importlib.import_module(fullname)
        # update/add import of both old and new name to point to the new module just imported
        sys.modules[old_name] = my_module
        sys.modules[fullname] = my_module
        return my_module

class MyImport(MetaPathFinder):
    def find_module(self, fullname, path=None):
        top_level_name = fullname.split(".")[0]
        if top_level_name == 'snpe' or fullname == 'snpe':
            return MyLoader()
        return None

# overwrite top-level asnpe import
sys.modules[__name__] = __import__('qti.aisw')

# overwrite all submodule imports.
sys.meta_path.insert(0, MyImport())
