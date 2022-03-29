# ==============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from .snpe_udo_config import *
from qti.aisw.converters.common.custom_ops.op_factory import *


class UDOFactory(CustomOpFactory, metaclass=ABCMeta):
    # global variables
    package_resolver = dict()
    op_collection = CustomOpCollection()

    def __init__(self):
        super(UDOFactory, self).__init__()

    @staticmethod
    def create_op(op_type, inputs, outputs, *args, **kwargs):
        return SnpeUdoCustomOp(op_type,
                               input_tensors=inputs,
                               output_tensors=outputs,
                               *args,
                               **kwargs)

    @classmethod
    def get_package_name(cls, op_type):
        # In SNPE, each package name and op name pair is unique. So the
        # the package resolver is a one to one mapping of op name - > package name
        for node_type, package_name in cls.package_resolver.items():
            if op_type.lower() == node_type.lower():
                return package_name
        raise TypeError("Op type: {} was not registered with any known packages".format(op_type))

    def parse_config(self, config_path, model, converter_type, **kwargs):
        """
        Parses a user provided json config into a udo op object. The config is expected to
        contain information about a user's operation as well as a package containing the op
        definition. See sample config in <examples> for more info. A UdoOp object is created from
        the parsed information and added to a UdoCollection object. Note that if no operator
        defined in the config spec, a udo op will not be created.

         :param config_path: The file path to the user's json config file
         :param model: The model containing the op(s) defined in the config spec.
         :param converter_type: The converter type from which the config was passed.
         """
        # Import config
        with open(config_path, 'r') as json_config:
            config_vars = json.load(json_config)

        for udo_package_name, udo_package_dict in config_vars.items():
            new_udo_package = UdoPackage(udo_package_dict['UDO_PACKAGE_NAME'])
            udo_package_info = UdoPackageInfo.from_dict(udo_package_dict)
            new_udo_package.add_package_info(udo_package_info)
            if model:
                for operator in udo_package_info.operators:
                    # Create UDO object and add to UDO collection
                    try:
                        udo_ops = self.create_ops_from_operator(operator, model=model,
                                                                converter_type=converter_type,
                                                                **kwargs)
                        self.op_collection[udo_ops[0].op_type] = udo_ops
                        for udo_op in udo_ops:
                            if udo_op.op_type in self.package_resolver:
                                if self.package_resolver[udo_op.op_type] != new_udo_package.name:
                                    raise ValueError("Attempted to register the same op with "
                                                     "name:{} across "
                                                     " the two different packages:{} vs {}".
                                                     format(udo_op.op_type,
                                                            self.package_resolver[udo_op.op_type],
                                                            new_udo_package.name))
                            self.package_resolver[udo_op.op_type] = new_udo_package.name
                    except CustomOpNotFoundError:  # if an op is not found then it is skipped
                        log_warning("Custom Op: {} was defined in the config but was not found in "
                                    "the model".
                                    format(operator.type_name))
                    except Exception as e:
                        raise e


OpFactory = UDOFactory
