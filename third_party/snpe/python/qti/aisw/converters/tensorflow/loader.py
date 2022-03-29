# =============================================================================
#
#  Copyright (c) 2016-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os

import tensorflow as tf
import qti.aisw.converters.common.utils.code_to_message as code_to_message
from qti.aisw.converters.tensorflow import tf_compat_v1
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.tensorflow.util import ConverterError
from qti.aisw.converters.tensorflow.util import GraphHelper
from qti.aisw.converters.tensorflow.util import GraphPrinter
from qti.aisw.converters.tensorflow.util import VisitableGraph
from qti.aisw.converters.tensorflow.util import is_tf2

class Model(object):
    class Input(object):
        INPUT_TYPE_DEFAULT = "default"

        def __init__(self, name, shape):
            self.name = name  # str
            self.shape = shape  # list[int]

    def __init__(self, graph_def, session, inputs, out_nodes_names, saved_model_tag="", saved_model_signature_key=""):
        """
        :type graph_def: tensorflow.GraphDef
        :type session: tensorflow.Session
        :type inputs: list[Model.Input]
        :type out_nodes_names: list[str]
        :type saved_model_tag: (optional) str
        :type saved_model_signature_key: (optional) str
        """
        self.graph_def = graph_def
        self.session = session
        self.inputs = inputs
        self.out_nodes_names = out_nodes_names
        self.saved_model_tag = saved_model_tag
        self.saved_model_signature_key = saved_model_signature_key


class ModelLoader(object):
    def __init__(self, session, in_model_path, in_nodes, in_dims, out_node,
                saved_model_tag="", saved_model_signature_key=""):
        """
        :type session: tf.Session
        :type in_model_path: string describing location of input model
        :type in_nodes: string describing input node name
        :type in_dims: List of Input dimensions for all inputs
        :type out_node: string describing output node name
        :type saved_model_tag: (optional) string to identify metagraph from savedmodel
        :type saved_model_signature_key: (optional) string to identify inputs and outputs from savedmodel
        """
        self.model = self.load(in_model_path, in_nodes, in_dims, out_node, session,
                                saved_model_tag, saved_model_signature_key)

    def prepare_savedmodel_functions(self, session):
        self.sess = session
        if is_tf2() :
            from tensorflow.python.saved_model.load import load as saved_model_load
            from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

            self.savedmodel_path_check = tf.saved_model.contains_saved_model
            self.savedmodel_load = lambda path, tags=None: saved_model_load(path, tags=tags)
            self.savedmodel_get_signature = lambda imported_model: imported_model.signatures
            self.savedmodel_get_version = lambda imported_model: imported_model.tensorflow_version

            def savedmodel_get_inputs_outputs(func):
                # get the defined inputs and outputs names from model. ignore tf.resource data type ones
                inputs = [tensor.name for tensor in func.inputs if tensor.dtype != tf.dtypes.resource]
                outputs = [tensor.name for tensor in func.outputs if tensor.dtype != tf.dtypes.resource]
                return inputs, outputs
            self.savedmodel_get_inputs_outputs = savedmodel_get_inputs_outputs

            def savedmodel_convert_variables_to_constants(func, inputs, outputs):
                frozen_concrete_func = convert_variables_to_constants_v2(func, lower_control_flow=False)
                graph_def = frozen_concrete_func.graph.as_graph_def()

                # replace outputs name by the name defined in savedmodel
                name_mapping = tf.nest.pack_sequence_as(
                        func.graph.structured_outputs,
                        frozen_concrete_func.graph.structured_outputs)
                name_map = {}
                for key in name_mapping:
                    name_map[name_mapping[key].name.lstrip('^').split(':')[0]] = key
                for node in graph_def.node:
                    if node.name in name_map:
                        old_name = node.name
                        new_name = name_map.get(node.name)
                        node.name = new_name
                        log_info(code_to_message.get_progress_message('INFO_TF_CHANGE_NODE_NAME')(old_name, new_name))
                        # replace related node's input
                        for _node in graph_def.node:
                            for idx, input_name in enumerate(_node.input):
                                if input_name == old_name:
                                    _node.input[idx] = new_name
                return graph_def
            self.savedmodel_convert_variables_to_constants = savedmodel_convert_variables_to_constants
            self.disable_eager_execution = tf_compat_v1.disable_eager_execution

        else:
            from tensorflow.python.saved_model.loader import load as saved_model_load
            from tensorflow.python.framework.graph_util import convert_variables_to_constants

            self.savedmodel_path_check = tf.saved_model.loader.maybe_saved_model_directory
            self.savedmodel_load = lambda path, tags=None: saved_model_load(self.sess, tags, path)
            self.savedmodel_get_signature = lambda imported_model: imported_model.signature_def
            self.savedmodel_get_version = lambda imported_model: imported_model.meta_info_def.tensorflow_version

            def savedmodel_get_inputs_outputs(func):
                # get the defined inputs and outputs names from model. ignore tf.resource data type ones
                inputs = [tensor.name.split(':')[0] for _, tensor in func.inputs.items() if tensor.dtype != tf.resource.as_datatype_enum]
                outputs = [tensor.name.split(':')[0] for _, tensor in func.outputs.items() if tensor.dtype != tf.resource.as_datatype_enum]
                return inputs, outputs
            self.savedmodel_get_inputs_outputs = savedmodel_get_inputs_outputs

            def savedmodel_convert_variables_to_constants(func, inputs, outputs):
                graph_def = convert_variables_to_constants(self.sess, self.sess.graph.as_graph_def(add_shapes=True), outputs)
                return graph_def
            self.savedmodel_convert_variables_to_constants = savedmodel_convert_variables_to_constants
            self.disable_eager_execution = lambda : None

    def get_model(self):
        return self.model

    def load(self, graph_path, input_nodes_names, input_nodes_shapes, out_node_names,
             session, saved_model_tag="", saved_model_signature_key=""):
        """
        Loads the Tensorflow Graph into the specified Session's Graph and builds a Model instance
        with all the relevant information for a ModelConverter to use during conversion.
        :type graph_path: str
        :type input_nodes_names: list[str]
        :type input_nodes_shapes: list[str]
        :type out_node_names: list[str]
        :type session: tensorflow.Session
        :type saved_model_tag: (optional) str
        :type saved_model_signature_key: (optional) str
        :rtype: Model
        """
        if len(input_nodes_names) != len(input_nodes_shapes):
            raise ConverterError(code_to_message.get_error_message(
                                     'ERROR_TF_INPUT_NODE_SHAPE_DIMS_MISMATCH'))

        graph_def = self.__import_graph(graph_path, session, out_node_names, saved_model_tag, saved_model_signature_key)
        with session.graph.as_default():
            inputs = []
            for name, shape in zip(input_nodes_names, input_nodes_shapes):
                self.__assert_node_in_graph(graph_def, name)
                input_tensor = session.graph.get_tensor_by_name(GraphHelper.indexed_tensor_name(name))

                batched_shape = []
                try:
                    tensor_shape = input_tensor.get_shape().as_list()
                    input_shape = list(map(int, shape.split(',')))
                    if len(input_shape) != len(tensor_shape):
                        raise ConverterError(code_to_message.get_error_message('ERROR_TF_INPUT_NODE_SHAPE_DIMS_MISMATCH'))
                    batched_shape = [1] * len(tensor_shape)
                    batched_shape[-len(input_shape):] = input_shape
                except ValueError:
                    pass

                if len(batched_shape) == 0:
                    try:
                        batched_shape = list(map(int, shape.split(',')))
                    except ValueError:
                        raise ConverterError(code_to_message.get_error_message('ERROR_TF_INVALID_INPUT_DIMS')(shape))

                inputs.append(Model.Input(name, batched_shape))

            visitable_graph = VisitableGraph(self.__get_graph_operations(graph_def, session.graph))
            visitable_graph.accept(GraphPrinter())

            return Model(graph_def, session, inputs, out_node_names, saved_model_tag, saved_model_signature_key)

    @classmethod
    def __get_graph_operations(cls, graph_def, graph):
        ops = [graph.get_operation_by_name(node.name) for node in graph_def.node]
        return ops

    @classmethod
    def __import_graph(cls, graph_path, session, out_nodes_names, saved_model_tag, saved_model_signature_key):
        """
        :type graph_path: str
        :type session: tensorflow.Session
        :type out_nodes_names: list[str]
        :rtype: tf.GraphDef
        """
        if not os.path.exists(graph_path):
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_GRAPH_FILE_DOES_NOT_EXIST')(graph_path))

        graph_path = os.path.abspath(graph_path)
        # SavedModel
        if os.path.isdir(graph_path):
            cls.prepare_savedmodel_functions(cls, session)
            if cls.savedmodel_path_check(graph_path):
                graph_def = cls.__import_from_savedmodel(graph_path, saved_model_tag, saved_model_signature_key)
            else:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_GRAPH_PATH_CANT_BE_RECOGNIZED')(graph_path))
        # Frozen Graph (pb)
        elif graph_path.endswith('.pb'):
            graph_def = cls.__import_from_frozen_graph(graph_path)
        # CheckPoint + MetaGraph
        elif graph_path.endswith('.meta'):
            checkpoint = graph_path.split('.meta')[0]
            graph_def = cls.__import_from_meta_graph(graph_path, session, checkpoint, out_nodes_names)
        else:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_GRAPH_PATH_CANT_BE_RECOGNIZED')(graph_path))

        if len(graph_def.node) == 0:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_NODES_NOT_FOUND_IN_GRAPH'))

        with session.graph.as_default():
            tf_compat_v1.import_graph_def(graph_def, name="")
        return graph_def

    @classmethod
    def __import_from_savedmodel(cls, savedmodel_path, tag, signature_key):
        tags = [tag]
        try:
            imported_model = cls.savedmodel_load(savedmodel_path, tags=tags)
        except:
            log_warning(code_to_message.get_warning_message('WARNING_TF_USE_FIRST_META_GRAPH')(tags))
            imported_model = cls.savedmodel_load(savedmodel_path)
        imported_signatures = cls.savedmodel_get_signature(imported_model)
        imported_signatures_keys = imported_signatures.keys()
        imported_model_version = cls.savedmodel_get_version(imported_model)

        if tf.__version__.split('.')[0] != imported_model_version[0]:
            log_warning(code_to_message.get_warning_message('WARNING_TF_MODEL_VERSION_DOES_NOT_MATCHED')(imported_model_version, tf.__version__))

        if len(imported_signatures_keys) == 0:
            raise ConverterError(code_to_message.get_error_message(
                                    'ERROR_TF_SIGNATURES_EMPTY_IN_SAVEDMODEL')(savedmodel_path))

        if signature_key not in imported_signatures_keys:
            input_signature_key = signature_key
            signature_key = list(imported_signatures_keys)[0]
            log_warning(code_to_message.get_warning_message('WARNING_TF_USE_FIRST_SIGNATURE_KEY')(input_signature_key, signature_key))

        func = imported_signatures[signature_key]
        inputs, outputs = cls.savedmodel_get_inputs_outputs(func)
        log_debug(code_to_message.get_progress_message('INFO_INPUT_OUTPUT_FROM_SAVEDMODEL')(inputs, outputs))

        graph_def = cls.savedmodel_convert_variables_to_constants(func, inputs, outputs)
        cls.disable_eager_execution()
        return graph_def

    @classmethod
    def __import_from_frozen_graph(cls, graph_path):
        graph_def = tf_compat_v1.GraphDef()
        with open(graph_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        return graph_def

    @classmethod
    def __import_from_meta_graph(cls, meta_graph_path, session, graph_path, out_nodes_names):
        """
        :type meta_graph_path: str
        :type graph_path: str
        :type out_nodes_names: list[str]
        :rtype: tensorflow.GraphDef
        """
        session = tf_compat_v1.Session(graph=tf_compat_v1.Graph())
        with session.graph.as_default():
            try:
                saver = tf_compat_v1.train.import_meta_graph(meta_graph_path)
            except AssertionError as e:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_CANNOT_IMPORT_GRAPH_FROM_META')(e.message))

            if saver is None:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_GRAPH_META_EMPTY'))
            saver.restore(session, graph_path)

        graph_def = session.graph.as_graph_def(add_shapes=True)
        return cls.__freeze_graph(session, graph_def, out_nodes_names)

    @classmethod
    def __freeze_graph(cls, session, graph_def, out_nodes_names):
        for node_name in out_nodes_names:
            cls.__assert_node_in_graph(graph_def, node_name)
        frozen = tf_compat_v1.graph_util.convert_variables_to_constants(session, graph_def, out_nodes_names)
        return frozen

    @classmethod
    def __assert_node_in_graph(cls, graph_def, node_name):
        if node_name not in [node.name for node in graph_def.node]:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_NODE_NOT_FOUND_IN_GRAPH')(node_name))
