# ==============================================================================
#
#  Copyright (c) 2018-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from abc import ABCMeta

from qti.aisw.converters.common.converter_ir import translation, op_adapter
from qti.aisw.converters.onnx.op_schema import OpSchemaBase, OpSchemaDict, OP_SCHEMA_REGISTRY
from .util import *


OnnxTranslations = translation.TranslationBank()


class OnnxTranslationBase(translation.ConversionTranslationBase):
    # onnx specific translation method keys
    ADD_INPUT = "ADD_INPUT"
    SUPPORTED_VERSION = "SUPPORTED_VERSION"

    def __init__(self):
        translation.ConversionTranslationBase.__init__(self)
        self.register_method(self.SUPPORTED_VERSION, self.get_supported_version)
        # list of dictionary-style class that maps {version:op_schema_list}
        self._op_schemas = []

    @staticmethod
    def fetch_constant_op(name, graph, *, prunable=True, quantizable=True, dtype=None, fail_if_dynamic=True,
                          fail_if_not_found=False):
        """
        Gets ConstantOp object for given tensor name if static
        :param name: the name of the tensor to look up
        :param graph: the IROpgraph instance
        :param prunable: determines if ConstantOp associated with provided name will be consumed by its
                         consumer node and if so it will later be pruned
        :param quantizable: if the ConstantOp associated with name is quantizable
        :param dtype: the data type of the tensor to be fetched from the weights list
        :param fail_if_dynamic: flag if True will raise ValueError if given tensor name is dynamic
        :param fail_if_not_found: flag if True will raise ValueError if given tensor name is not found
        :return: the ConstantOp associated with the provided name if found, None otherwise
        :raises ValueError if buffer is found for provided name but is not produced by a ConstantOp
        """
        op = None
        if not graph.has_buffer(name) and graph.weights.has(name):
            buf_value = graph.weights.fetch(name, prunable=prunable, dtype=dtype)
            op = op_adapter.ConstantOp(name, buf_value, quantizable=quantizable)
        elif graph.has_buffer(name):
            op = graph.get_producer_op(name)
            if op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                op.quantizable = quantizable
                # Constant Op translation adds the associated tensor to the weight map, use that to mark
                # prunability.
                if name in graph.weights.weight_map:
                    graph.weights.weight_map[name].consumed = prunable
            elif fail_if_dynamic:
                raise ValueError("Dynamic value for tensor name: {}, is not supported.".format(name))
            else:
                return None
        elif fail_if_not_found:
            raise ValueError("Input tensor: {} not found in the graph.".format(name))
        return op

    def add_src_op_info(self, node_name, src_op, graph):
        graph.add_src_op_info(node_name,
                              [i for i in src_op.input],
                              [o for o in src_op.output])

    def extract_parameters(self, src_op, graph):
        raise NotImplementedError("extract_parameters for {} not "
                                  "implemented ".format(str(self.__class__.__name__)))

    def extract_input_names(self, src_op, graph):
        return list(map(str, src_op.input))

    def extract_output_names(self, src_op, graph):
        return list(map(str, src_op.output))

    def get_supported_version(self, op_type):
        try:
            versions = []
            for schema_dict in self._op_schemas:
                # There may be more than one op schema associated with a given translation
                # This loop ensures that the right schema is selected
                if schema_dict.op_name == op_type:
                    versions = list(map(int, schema_dict.get_schemas().keys()))
            if not versions:
                raise RuntimeError(code_to_message.get_error_message
                                   ("ERROR_OP_SCHEMA_NOT_FOUND")(op_type))
            return versions
        except Exception as e:
            raise RuntimeError(code_to_message.get_error_message
                               ("ERROR_GET_SUPPORTED_VERSION")(op_type, str(e)))

    def register_op_schema(self, name, versions, unsupported_attrs=None):
        """
           Wraps Onnx's internal schema definition into a condensed op_schema_dict internal object (OpSchemaDict)
           which contains individual op_schema(s)(OpSchemaBase) that tie supported attributes,
           number of inputs and outputs to the appropriate op version

           :param name: The type of op to be registered
           :param versions : list of versions of the op to be registered. Note the versions must be available in
                             the Onnx spec.
           :param unsupported_attrs: A list of lists of unsupported attrs, which are in the Onnx spec
                                    for an op version but are not supported by the translation

           registers the resulting op_schema dictionary with the translation, as well as with a
           global schema registry

        """
        op_schema_idx = 0
        if unsupported_attrs:
            while len(unsupported_attrs) < len(versions):
                unsupported_attrs.append(unsupported_attrs[0])
        else:
            unsupported_attrs = [[] for _ in range(len(versions))]

        for i, version in enumerate(versions):
            try:
                # Note: get_schema uses version as maxInclusiveVersion and returns the schema
                # with the biggest version, which is not greater than specified version in
                # specified domain
                schema = defs.get_schema(name, version, '')
                op_schema = OpSchemaBase()
                op_schema.populate_op_schema(schema, unsupported_attrs[i])

                # if a schema dictionary already exists, then a new version is added. Otherwise,
                # a new op schema dictionary is created and the new schema is added
                schema_dicts = [schema_dict for schema_dict in self._op_schemas
                                if schema_dict.op_name == name]
                if schema_dicts:
                    schema_dicts[0].add_schema(op_schema, version)
                else:
                    op_schema_idx = len(self._op_schemas) - 1 if self._op_schemas else 0
                    schema_dict = OpSchemaDict(name)
                    schema_dict.add_schema(op_schema, version)
                    self._op_schemas.append(schema_dict)
            except RuntimeError as e:
                # Only warn user here since even though their onnx installation doesnt have all
                # the ops(and the different versions) we support, their model might not contain
                # that Op. If it does, it will be caught at conversion time later. Note: need to
                # use print here instead of log_warning since Ops are registered at module import
                #

                print(code_to_message.get_warning_message("WARNING_OP_NOT_SUPPORTED_BY_ONNX")(name,
                                                                                              version,
                                                                                              str(e)))

                # add a dummy op schema dict so functions can still be registered
                self._op_schemas.append(OpSchemaDict(name))

        OP_SCHEMA_REGISTRY[name.lower()] = self._op_schemas[op_schema_idx]
        return self._op_schemas[op_schema_idx]

    def op_schema(self, version: str = None, op_type: str = None):
        values = []

        # If version is provided, then all registered schemas matching that version are returned
        if version is not None:
            schema_versions = []
            for schema_dict in self._op_schemas:
                # There may be more than one op schema associated with a given translation
                # This loop ensures that the right schema is selected
                if op_type and not schema_dict.op_name == op_type:
                    continue
                schema_versions.append(schema_dict.get_schemas(version))

            if not schema_versions:
                raise RuntimeError(code_to_message.get_error_message
                                   ("ERROR_OP_SCHEMA_VERSION_NOT_FOUND")(str(version),
                                                                         op_type))
            return schema_versions
        else:
            # if no explicit version is requested, then we can retrieve the op_schemas associated
            # with this translation. If there is more than op schema dictionary registered then an
            # error is returned if no op type is requested. Otherwise, the latest schema is returned
            # for that op type. The latest is also returned if only one schema is registered.
            if not self._op_schemas:
                raise ValueError("No op schemas were registered for this translation: "
                                 "{}".format(self.__class__.__name__))
            elif len(self._op_schemas) == 1:
                values = list(self._op_schemas[0].get_schemas().values())
            elif len(self._op_schemas) > 1:
                if not op_type:
                    # internal error
                    raise AttributeError("Op type attribute must be provided when translation"
                                         " has more than one op schema registered")
                else:
                    schema_dicts = [schema_dict for schema_dict in self._op_schemas
                                    if schema_dict.op_name == op_type]
                    if not schema_dicts:
                        raise RuntimeError(code_to_message.get_error_message
                                           ("ERROR_OP_SCHEMA_NOT_FOUND")(op_type))
                    values = list(schema_dicts[0].get_schemas().values())
            return values[-1]


class ElementwiseBinaryTranslationBase(OnnxTranslationBase, metaclass=ABCMeta):
    """
    Additional BaseClass for elementWiseBinary Ops(mul, prod, div and sub) since they need add_op to handle constant Op
    addition to graph
    """
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.input_names = []

    def add_op(self, src_op, graph):
        op = self.extract_parameters(src_op, graph)
        self.input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)

        if op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            self.add_src_op_info(op.name, src_op, graph)
            return graph.add(op, [], output_names)

        for input_name in self.input_names:
            const_op = self.fetch_constant_op(input_name, graph, prunable=False, fail_if_dynamic=False)
            # Add fetched constant op to graph, if it doesn't exist
            if const_op is not None:
                if not graph.has_buffer(input_name):
                    const_node = graph.add(const_op, [], input_name)
                    graph.add_src_op_info(const_op.name, None, const_node.output_names[0])

        # Add elementwise src op info
        self.add_src_op_info(op.name, src_op, graph)

        return graph.add(op, self.input_names, output_names)

    def extract_input_names(self, src_op, graph):
        return [input_name for input_name in src_op.input]


# -----------------------------------------------------------------
# Converter translations
# Note: ONNX doesn't have input op(s) but we create one for the IR
# -----------------------------------------------------------------
class OnnxInputTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_input_names(self, src_op, graph):
        raise NotImplementedError("extract_input_names() for {} not implemented ".format(str(self.__class__.__name__)))

    def extract_output_names(self, src_op, graph):
        raise NotImplementedError("extract_output_names() for {} not implemented ".format(str(self.__class__.__name__)))

    def add_op(self, src_op, graph, **kwargs):
        raise NotImplementedError("add_op() for {} not implemented. Call add_input_op() instead."
                                  .format(str(self.__class__.__name__)))

    def add_input_op(self, input_, graph, **kwargs):
        name = str(input_.name)
        tensor_shape = input_.type.tensor_type.shape
        shape = [int(dim.dim_value) for dim in tensor_shape.dim]
        neg_idx = [idx for idx in range(len(shape)) if shape[idx] <= 0]

        if neg_idx:
            raise RuntimeError('Negative/placeholder dimensions is not supported.'
                               'Expected shape: {} > 0\nNote: Dynamic input batch_size not supported. '
                               'Use --input_dim command to provide a static batch value'.format(shape))

        return graph.add_input(name, shape)


OnnxTranslations.register_translation(OnnxInputTranslation(),
                                      converter_type('input', 'onnx'),
                                      op_adapter.InputOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Dropout and other Noops
# ------------------------------------------------------------------------------
class OnnxNoopTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Dropout', [1, 6, 7, 10])

    def extract_parameters(self, src_op, graph):
        return op_adapter.NoopOp(src_op.name)

    def extract_output_names(self, src_op, graph):
        return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxNoopTranslation(),
                                      converter_type('Dropout', 'onnx'),
                                      op_adapter.NoopOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Class OpVersionInfo
# ------------------------------------------------------------------------------
# Returns name and version information about an op from a particular model
class OpVersionInfo:
    def __init__(self):
        self.model_opset_version = 0

    @staticmethod
    def update_schema_registry(src_op_type, op_version):
        """ Updates the schema registry so that get_op_schema(src_op_type) will always return the appropriate schema
            for the global model opset version """
        op_schema_dict = OP_SCHEMA_REGISTRY[src_op_type.lower()]
        op_schema_keys = list(op_schema_dict.get_schemas().keys())
        if op_schema_keys[-1] != str(op_version):
           op_schema_dict.reorder_op_schemas(str(op_version))

    def validate_op_ver(self, src_op, supported_version):
        """

        :param src_op: The op from the Onnx framework
        :param supported_version: The version of the op supported by the Onnx Converter
        :return: a warning if the opset_version for the source op does not match any version supported
                 by the converter
                 updates the schema registry if the src_op version is supported, so that any schema calls (self.op_schema()
                 or get_op_schema) will return the src_op_version.
        """

        # This uses the model version to extract the associated opset version for a given op
        # For example:
        # The scenarios are described below:
        # supported_version = [1, 6, 7]
        # Model_opset_version = 3,    Model_opset_version = 7,   Model_opset_version = 7,    Model_opset_version = 9
        # current_op_version = 1,     current_op_version = 7,    current_op_version = 1      current_op_version = 8
        #                                                        returns a warning for       returns a warning for
        #                                                        onnx installation support   converter support
        try:
            current_op_version = int(defs.C.get_schema(src_op.op_type, self.model_opset_version, '').since_version)
            if current_op_version not in supported_version:
                log_warning(code_to_message.get_warning_message("WARNING_OP_VERSION_NOT_SUPPORTED")
                            (src_op.op_type, list(map(int, supported_version)), [current_op_version]))
            else:
                if self.model_opset_version != current_op_version and self.model_opset_version in supported_version:
                    log_warning(code_to_message.get_warning_message("WARNING_OP_VERSION_NOT_SUPPORTED_BY_ONNX")
                                (src_op.op_type, self.model_opset_version, current_op_version))
                self.update_schema_registry(src_op.op_type, current_op_version)
        except RuntimeError as e:
            # Throw an error here since model contains an op or a max_version that is not part of the current onnx
            # installation.
            # Note: re-raising error since the onnx error message is not very informative
            raise RuntimeError(code_to_message.get_error_message("ERROR_OP_NOT_SUPPORTED_BY_ONNX")(src_op.op_type,
                               self.model_opset_version, str(e)))

    def set_global_op_ver(self, model):
        """ Sets the highest global op version supported by the model"""
        # Get the global opset version
        if len(model.opset_import) > 1:
            log_warning(code_to_message.get_warning_message("WARNING_OPSET_VERSION"))

        for opset in model.opset_import:
            if opset.version > self.model_opset_version:
                self.model_opset_version = opset.version
