# ==============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from abc import abstractmethod, ABCMeta


class RelayImporter(object):
    __metaclass__ = ABCMeta

    def __init__(self, args):
        self.mod = None
        self.params = None

    @abstractmethod
    def convert_to_relay(self, input_model_path, **kwargs):
        """
        :param input_model_path: String representing path to source model
        :param kwargs:
        :return: Relay Module, Relay Params, [Expr to Output Names Dict]
        """
        raise NotImplementedError("convert_to_relay not implemented for {}".format(str(self.__class__.__name__)))
