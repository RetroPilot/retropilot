# ==============================================================================
#
#  Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.custom_ops.utils.config_helpers import CustomOpNotFoundError
from abc import abstractmethod, ABCMeta
import copy

# global variables
package_resolver = dict()


class CustomOpFactory(object):
    __metaclass__ = ABCMeta
    """
    The Factory object which manages all custom op creation and instantiation.
    """
    op_collection = dict()

    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def create_op(op_type, inputs: list, outputs: list, *args, **kwargs):
        """
        This method creates a generic CustomOp. Note that the CustomOp is an abstract class,
        so the Op returned here
        is intended to be implemented.

        :param op_type: The type of the CustomOp object
        :param inputs: A list of objects describing the inputs
        :param outputs: A list of objects describing the outputs
        :param args:  Optional arguments to the Op constructor
        :return: A CustomOp object
        """

    @classmethod
    def create_ops_from_operator(cls, operator, converter_type, model=None, **kwargs):
        """
        Creates multiples ops from a single Operator object, using a list of src_ops in the model
        that match the operator spec.
        :param operator: The operator to be used
        :param model: The framework model
        :param converter_type: The given converter type
        :return:
        """
        nodes = cls.get_src_ops(str(operator.type_name).lower(), model, converter_type)
        resolved_ops = []

        for node in nodes:
            resolved_ops.append(cls.create_op_from_operator(operator, node, model,
                                                            converter_type, **kwargs))

        return resolved_ops

    @classmethod
    def create_op_from_operator(cls, operator, node, model, converter_type,
                                **kwargs):
        """
        This method creates a CustomOp from an Operator object, based on the operator itself,
        the src_op node, the model containing said Op and the converter type.

        :param operator: An Operator object
        :param node:     The node as defined in the framework Model, needed if the model is not
        provided.
        :param model: The framework model, needed if the node is not provided.
        :param converter_type: The converter type to which the config containing the operator
        definition was provided.
        :return: A well-defined CustomOp based on the converter type selected, the default return
        value is a CustomOnnx Op.
        """
        input_tensor_info = copy.deepcopy(operator.input)  # pass by value
        output_tensor_info = copy.deepcopy(operator.output)
        param_info = operator.param if hasattr(operator, 'param') else []
        converter_type = str(converter_type).lower()
        if not node:
            raise RuntimeError(code_to_message.get_error_message("ERROR_CANNOT_CREATE_CUSTOM_OP"))
        if converter_type == 'onnx':
            return cls.__create_onnx_op(node, input_tensor_info, output_tensor_info, param_info,
                                        model)
        elif converter_type == 'caffe':
            return cls.__create_caffe_op(node, input_tensor_info, output_tensor_info,
                                         param_info, caffe_net=kwargs['caffe_net'])
        elif converter_type == 'tf':
            return cls.__create_tf_op(node, input_tensor_info, output_tensor_info, param_info,
                                      model)
        elif converter_type == 'caffe2':
            return cls.__create_caffe2_op(node, input_tensor_info, output_tensor_info, param_info,
                                          model)

    @classmethod
    def get_src_ops(cls, op_type, model, src_type):
        if str(src_type).lower() == 'onnx':
            return cls.get_onnx_src_ops(op_type, model)
        elif str(src_type).lower() == 'caffe':
            return cls.get_caffe_src_ops(op_type, model)
        elif str(src_type).lower() == 'caffe2':
            raise NotImplementedError
        elif str(src_type).lower() == 'tf':
            return cls.get_tf_src_ops(op_type, model)

    @staticmethod
    def update_tensor_infos_with_src_op_names(tensor_infos, src_op_names):
        """
        Changes the tensor infos that will be ingested by a CustomOp to match an actual instance
        of a src_op. Note that ordering of both iterables must match.
        :param tensor_infos: A list of input or output tensor infos
        :param src_op_names: A list of names for each input or output tensors
        :return:
        """
        new_tensor_infos = []
        num_of_repeated_tensors = len(
            [tensor_info for tensor_info in tensor_infos if tensor_info.repeated])
        if num_of_repeated_tensors > 1:
            raise ValueError(
                "There can be at most one repeated tensor in a Custom Op spec, found: {} repeated "
                "tensors in {}".format(num_of_repeated_tensors, src_op_names))

        if tensor_infos[0].repeated:
            # find the tensor that is static, as that will not be replicated. note that this
            # requires all static tensors be placed after repeated tensors, which is a reasonable
            # assumption since the first input is almost always data.
            static_tensor_info = [tensor_info for tensor_info in tensor_infos if tensor_info.static]
            if static_tensor_info:
                variadic_tensor_info_len = len(src_op_names) - len(static_tensor_info)
            else:
                variadic_tensor_info_len = len(src_op_names)

            for j in range(variadic_tensor_info_len):
                # creates a copy of the tensor_info to be replicated, so that each new
                # tensor_info in the list will have a different address in memory
                __tensor_info = copy.deepcopy(tensor_infos[0])
                # once we duplicate a tensor it is no longer repeated
                __tensor_info.repeated = False
                new_tensor_infos.append(__tensor_info)

            new_tensor_infos = new_tensor_infos + static_tensor_info

            if len(new_tensor_infos) != len(src_op_names):
                raise Exception(
                    code_to_message.get_error_message("ERROR_CANNOT_RESOLVE_VARIADIC_OPERATION")
                    (tensor_infos[0].name))
        else:
            new_tensor_infos = tensor_infos

        for i, name in enumerate(src_op_names):
            new_tensor_infos[i].name = name

        return new_tensor_infos

    @staticmethod
    def __create_onnx_op(src_op, input_tensor_info, output_tensor_info, param_info, model=None):
        """
        :return: A well-defined Custom Onnx Op object
        """
        from qti.aisw.converters.onnx.custom_op_definition import CustomOnnxOp
        output_tensor_info = CustomOpFactory.update_tensor_infos_with_src_op_names(
            output_tensor_info, src_op.output)
        input_tensor_info = CustomOpFactory.filter_optional_input_tensors(input_tensor_info,
                                                                          src_op.input,
                                                                          src_op.name)
        input_tensor_info = CustomOpFactory.update_tensor_infos_with_src_op_names(input_tensor_info,
                                                                                  src_op.input)
        return CustomOnnxOp(src_op, input_tensor_info, output_tensor_info, param_info, model)

    @staticmethod
    def __create_caffe2_op(src_op, input_tensor_info, output_tensor_info, param_info, model=None):
        """
        :return: TO-DO
        """
        raise NotImplementedError

    @staticmethod
    def __create_tf_op(src_op, input_tensor_info, output_tensor_info, param_info, model=None):
        """
       :return: A well-defined TfCustomOp object
       """
        from qti.aisw.converters.tensorflow.layers.custom import CustomTfOp
        output_tensor_info = CustomOpFactory. \
            update_tensor_infos_with_src_op_names(output_tensor_info, [output.name for output in
                                                                       src_op.outputs])

        input_tensor_names = [input.name for input in src_op.inputs]
        _param_info = copy.deepcopy(param_info)  # to avoid changing original param_info
        input_tensor_info = CustomOpFactory.filter_optional_input_tensors(input_tensor_info,
                                                                          input_tensor_names,
                                                                          op_name=src_op.name)

        # loop is to support the case where an input is mis-classified as a param. This loop
        # finds that param in src_op.op_def.input_args and designates it as static, implying that
        # it has data which would be retrieved either from a constant/identity op.
        for i, name in enumerate(input_tensor_names):
            # if any tensor infos are repeated, the logic to identify mis-classified params would
            # not be valid. once a repeated tensor info is identified, the code below will repeat
            # the first repeated tensor and cause an early exit.
            if input_tensor_info[i].repeated:
                input_tensor_info[i:] = CustomOpFactory. \
                    update_tensor_infos_with_src_op_names(input_tensor_info[i:],
                                                          input_tensor_names[i:])
                break
            try:
                CustomOpFactory.update_tensor_infos_with_src_op_names([input_tensor_info[i]],
                                                                      [name])
            except IndexError:
                input_param = None
                for info in _param_info:
                    # iterates through the list of input args, and updates if the input index and
                    # the names match any input_arg
                    for j, input_arg in enumerate(src_op.op_def.input_arg):
                        if input_arg.name == info.name and i == j:
                            input_param = input_arg
                            info.name = name
                            info.static = True
                            break
                if not input_param:
                    raise RuntimeError(
                        "Input with name: {} is not registered in operator config".format(name))
        return CustomTfOp(src_op, input_tensor_info, output_tensor_info, _param_info, model)

    @staticmethod
    def __create_caffe_op(src_op, input_tensor_info, output_tensor_info, param_info,
                          caffe_net=None):
        """
       :return: creates a well-defined Caffe Custom Op object
       """
        from qti.aisw.converters.caffe.custom_op_definition import CustomCaffeOp
        CustomOpFactory.update_tensor_infos_with_src_op_names(output_tensor_info, src_op.top)
        input_tensor_info = CustomOpFactory.filter_optional_input_tensors(input_tensor_info,
                                                                          src_op.bottom,
                                                                          op_name=src_op.name)
        CustomOpFactory.update_tensor_infos_with_src_op_names(input_tensor_info, src_op.bottom)
        return CustomCaffeOp(src_op, input_tensor_info, output_tensor_info, param_info, caffe_net)

    @staticmethod
    def get_onnx_src_ops(op_type, model):
        """
        Gets an onnx node from a model using its op_name

        :param op_type: The name of the onnx op
        :param model: The ModelProto object
        :return: the nodes if present, or a TypeError otherwise
        """
        nodes = []
        found = False
        for node in model.graph.node:
            if str(node.op_type).lower() == str(op_type).lower():
                nodes.append(node)
                found = True
        if not found:
            log_debug(code_to_message.get_debugging_message("DEBUG_CUSTOM_OP_NOT_FOUND")
                      (op_type, ""))
            raise CustomOpNotFoundError
        log_debug(code_to_message.get_debugging_message("DEBUG_CUSTOM_OPS_FOUND")
                  (len(nodes), op_type))
        return nodes

    @staticmethod
    def get_tf_src_ops(op_type, model):
        """
        Gets a TF node from a model using its op_type

        :param op_type: The type of the TF op
        :param model: A TF converter ModelLoader object
        :return: the nodes if present, or a TypeError otherwise
        """
        nodes = []
        found = False
        for node in model.session.graph.get_operations():
            if str(node.type).lower() == op_type.lower():
                nodes.append(node)
                found = True
        if not found:
            log_debug(code_to_message.get_debugging_message("DEBUG_CUSTOM_OP_NOT_FOUND")
                      (op_type, ""))
            raise CustomOpNotFoundError
        log_debug(code_to_message.get_debugging_message("DEBUG_CUSTOM_OPS_FOUND")
                  (len(nodes), op_type))
        return nodes

    @staticmethod
    def get_caffe_src_ops(op_type, model):
        """
        Gets a caffe node from a model using its op_type

        :param op_type: The type of the caffe op
        :param model: A repeated composite container of caffe objects
        :return: the nodes if present, or a TypeError otherwise
        """
        layers = []
        found = False
        for layer in model:
            if str(layer.type).lower() == op_type:
                layers.append(layer)
                found = True
        if not found:
            log_debug(code_to_message.get_debugging_message("DEBUG_CUSTOM_OP_NOT_FOUND")
                      (op_type, ""))
            raise CustomOpNotFoundError
        log_debug(code_to_message.get_debugging_message("DEBUG_CUSTOM_OPS_FOUND")
                  (len(layers), op_type))
        return layers

    @abstractmethod
    def parse_config(self, config_path, model, converter_type, **kwargs):
        """
        Parses a user provided json config into a custom op object. The config is expected to
        contain information about a user's operation as well as a package containing the op
        definition. See sample config in <examples> for more info. A CustomOp object is created
        from the parsed information and added to a OpCollection object. Note that if no operator
        defined in the config spec, a custom op will not be created.

         :param config_path: The file path to the user's config file
         :param model: The model containing the op(s) defined in the config spec.
         :param converter_type: The converter type from which the config was passed.
         """

    @staticmethod
    def filter_optional_input_tensors(input_tensor_info, input_names, op_name=""):
        """
        Removes optional input tensors that are specified in the config but not in the op
        instance, provided the input tensor is not static and has a pre-specified default value.
        :param input_tensor_info: The full list of input tensor infos
        :param input_names: The input names from the op instance
        :param op_name: The name of the operation
        :return: A modified input tensor info list with optional tensors removed

        :raises KeyError if the tensor info is not present in the tensor info is neither static or
                          has no default value
        """
        input_tensor_len = len(input_names)
        if input_tensor_len < len(input_tensor_info):
            input_tensor_info_copy = copy.deepcopy(input_tensor_info)
            for i, tensor_info in enumerate(input_tensor_info_copy[input_tensor_len:]):
                if tensor_info.static:
                    continue
                elif tensor_info.default_value is not None:
                    input_tensor_info.pop(i + input_tensor_len)
                else:
                    raise KeyError("Required input tensor: {} was found in the spec "
                                   "but not in the op instance for: {}".format(tensor_info.name,
                                                                               op_name))
        return input_tensor_info
