
###############
# This file replace the AttrDict package that was broken since python 3.10
###############

from collections import UserDict
# from attrdict import AttrDict

class AttrDict(UserDict):
    def __getattr__(self, key):
        return self.__getitem__(key)
    def __setattr__(self, key, value):
        if key == "data":
            return super().__setattr__(key, value)
        return self.__setitem__(key, value)