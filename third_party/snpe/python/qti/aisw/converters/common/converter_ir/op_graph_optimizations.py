# ==============================================================================
#
#  Copyright (c) 2019-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import copy
import json
import numpy
from operator import mul
from functools import reduce


from qti.aisw.converters.common.converter_ir import translation, op_adapter, op_graph
from qti.aisw.converters.common.converter_ir.op_graph import InputEncodings
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker, CaffeAxisOrder, OnnxAxisOrder
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper
from qti.aisw.converters.common.utils import code_to_message, translation_utils


# ------------------------------
#   Module Level enum/Functions
# ------------------------------
INJECT_CAST_FOR_GATHER = "INJECT_CAST_FOR_GATHER"
REMOVE_NOOP = "REMOVE_NOOP"
REMOVE_CAST_NOOP = "REMOVE_CAST_NOOP"
REMOVE_DISCONNECTED = "REMOVE_DISCONNECTED"
MATCH_CHANNELSHUFFLE = "MATCH_CHANNELSHUFFLE"
MATCH_GELU = "MATCH_GELU"
MATCH_HARDSWISH = "MATCH_HARDSWISH"
MATCH_LAYERNORM = "MATCH_LAYERNORM"
MATCH_CAFFE_SSD_TO_TF = "MATCH_CAFFE_SSD_TO_TF"
MATCH_SPACETODEPTH = "MATCH_SPACETODEPTH"
SQUASH_BATCHNORM = "SQUASH_BATCHNORM"
SQUASH_SCALE = "SQUASH_SCALE"
SQUASH_BOX_DECODER = "SQUASH_BOX_DECODER"
SQUASH_SUM = "SQUASH_SUM"
SQUASH_PROD = "SQUASH_PROD"
SQUASH_DIV = "SQUASH_DIV"
SQUASH_SUB = "SQUASH_SUB"
SQUASH_PAD = "SQUASH_PAD"
FOLD_CONCATS = "FOLD_CONCATS"
AXES_TO_SPATIAL_FIRST_ORDER = "AXES_TO_SPATIAL_FIRST_ORDER"
ADD_QPARAMS = "ADD_QPARAMS"
CHAIN_ELTWISE_OPS = "CHAIN_ELTWISE_OPS"
ADJUST_NMS_FEATURE_DIMS = "ADJUST_NMS_FEATURE_DIMS"
EXTRACT_COLOR_TRANSFROM = "EXTRACT_COLOR_TRANSFROM"
OPTIMIZE_NEG = "OPTIMIZE_NEG"
PREPROCESS_ROI_POOL_INPUTS = "PREPROCESS_ROI_POOL_INPUTS"
PREPROCESS_LSTM_OPS = "PREPROCESS_LSTM_OPS"
UNROLL_LSTM_TIME_STEPS = "UNROLL_LSTM_TIME_STEPS"
MERGE_LOW_LEVEL_OPS_TO_LAYERS = "MERGE_LOW_LEVEL_OPS_TO_LAYERS"
REMOVE_QUANT_NODES = "REMOVE_QUANT_NODES"
SQUASH_QUANT_NODES = "SQUASH_QUANT_NODES"
ALIGN_MATMUL_RANKS = "ALIGN_MATMUL_RANKS"
PREPARE_INPUTS_AS_PARAMS = "PREPARE_INPUTS_AS_PARAMS"
HANDLE_GATHER_NEGATIVE_INDICES = "HANDLE_GATHER_NEGATIVE_INDICES"
PREPARE_BIASES = "PREPARE_BIASES"
supported_opt_list = [SQUASH_SCALE, SQUASH_PROD, SQUASH_DIV, SQUASH_SUM, SQUASH_SUB, SQUASH_BATCHNORM, FOLD_CONCATS,
                      MATCH_CHANNELSHUFFLE, MATCH_GELU, MATCH_HARDSWISH, MATCH_LAYERNORM, AXES_TO_SPATIAL_FIRST_ORDER,
                      REMOVE_NOOP, REMOVE_CAST_NOOP, ADD_QPARAMS, CHAIN_ELTWISE_OPS,
                      ADJUST_NMS_FEATURE_DIMS, EXTRACT_COLOR_TRANSFROM, OPTIMIZE_NEG, MATCH_SPACETODEPTH,
                      PREPROCESS_ROI_POOL_INPUTS, PREPROCESS_LSTM_OPS, UNROLL_LSTM_TIME_STEPS, SQUASH_PAD,
                      MERGE_LOW_LEVEL_OPS_TO_LAYERS, INJECT_CAST_FOR_GATHER, REMOVE_QUANT_NODES, SQUASH_QUANT_NODES,
                      ALIGN_MATMUL_RANKS, PREPARE_INPUTS_AS_PARAMS, HANDLE_GATHER_NEGATIVE_INDICES, PREPARE_BIASES]

format_to_permute_order = {'NSC': AxisTracker.AxisFormat.NSC_TO_NCS,
                           'BTF': AxisTracker.AxisFormat.BTF_TO_TBF}
format_to_format = {'NSC': AxisTracker.AxisFormat.NCS, 'BTF': AxisTracker.AxisFormat.TBF}
OptimizationTranslations = translation.TranslationBank()


class IROptimizations(object):
    class ArgParser(ArgParserWrapper):
        def __init__(self, **kwargs):
            super(IROptimizations.ArgParser, self).__init__(**kwargs)
            self.add_optional_argument("--dumpIR", action="store_true",
                                       help=argparse.SUPPRESS,
                                       default=False)
            self.add_optional_argument("--disable_batchnorm_folding",
                                       default=False,
                                       action="store_true")
            self.add_optional_argument("--squash_box_decoder",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--match_caffe_ssd_to_tf",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--disable_chaining_eltwise_ops",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--adjust_nms_features_dims",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--extract_color_transform",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--preprocess_roi_pool_inputs",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--perform_axes_to_spatial_first_order",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--preprocess_lstm_ops",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--unroll_lstm_time_steps",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--force_prune_cast_ops",
                                       default=True,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--inject_cast_for_gather",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--use_convert_quantization_nodes",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--align_matmul_ranks",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--prepare_inputs_as_params",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--handle_gather_negative_indices",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")

    def __init__(self, args):
        self.dump_ir_graph = args.dumpIR
        self.enable_batchnorm_folding = not args.disable_batchnorm_folding
        self.squash_box_decoder = args.squash_box_decoder
        self.match_caffe_ssd_to_tf = args.match_caffe_ssd_to_tf
        self.chain_eltwise_ops = not args.disable_chaining_eltwise_ops
        self.adjust_nms_features_dims = args.adjust_nms_features_dims
        self.extract_color_transform = args.extract_color_transform
        self.perform_axes_to_spatial_first_order = args.perform_axes_to_spatial_first_order
        self.preprocess_roi_pool_inputs = args.preprocess_roi_pool_inputs
        self.preprocess_lstm_ops = args.preprocess_lstm_ops
        self.unroll_lstm_time_steps = args.unroll_lstm_time_steps
        self.force_prune_cast_ops = args.force_prune_cast_ops
        self.inject_cast_for_gather = args.inject_cast_for_gather
        self.use_convert_quantization_nodes = args.use_convert_quantization_nodes
        self.align_matmul_ranks = args.align_matmul_ranks
        self.prepare_inputs_as_params = args.prepare_inputs_as_params
        self.handle_gather_negative_indices = args.handle_gather_negative_indices

    def optimize(self, graph):
        # apply graph transformations
        log_debug2("Applying graph Optimizations...")

        # Dump the IR for debug before or after an optimization using graph.dump_json(<filename>)
        if self.dump_ir_graph:
            log_info("Dumping IR graph before all optimizations as IRGraph_before_optimizations.json")
            graph.dump_json("IRGraph_before_optimizations.json")

        # First attempt to match and fold quant nodes, then remove any remaining
        if graph.keep_quant_nodes:
            if self.use_convert_quantization_nodes:
                OptimizationTranslations.apply_method_to_graph(SQUASH_QUANT_NODES, graph, fail_if_no_method=False)
        else:
            OptimizationTranslations.apply_method_to_all_ops(REMOVE_QUANT_NODES, graph, fail_if_no_method=False)

        if graph.user_quantization_overrides:
            self.populate_quantization_params(graph)

        # TODO Remove this preparation once backends are able to consume optional bias tensors
        # prepares bias tensors from frontends for consumption by optimizations and backends
        OptimizationTranslations.apply_method_to_all_ops(PREPARE_BIASES, graph, fail_if_no_method=False)

        # this optimization needs to be run first before any other optimizations
        OptimizationTranslations.apply_method_to_graph(MERGE_LOW_LEVEL_OPS_TO_LAYERS, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_PAD, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(FOLD_CONCATS, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_CHANNELSHUFFLE, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_GELU, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_HARDSWISH, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_LAYERNORM, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_SPACETODEPTH, graph, fail_if_no_method=False)

        # Element-wise squashing optimizations. This shall be done after matching larger sequences as they single-op
        # squashing into previous layer
        OptimizationTranslations.apply_method_to_graph(SQUASH_SCALE, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_PROD, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_DIV, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_SUM, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_SUB, graph, fail_if_no_method=False)

        if self.enable_batchnorm_folding:
            OptimizationTranslations.apply_method_to_graph(SQUASH_BATCHNORM, graph, fail_if_no_method=False)
        if self.squash_box_decoder:
            OptimizationTranslations.apply_method_to_graph(SQUASH_BOX_DECODER, graph, fail_if_no_method=False)
        if self.match_caffe_ssd_to_tf:
            OptimizationTranslations.apply_method_to_graph(MATCH_CAFFE_SSD_TO_TF, graph, fail_if_no_method=False)
        if self.adjust_nms_features_dims:
            OptimizationTranslations.apply_method_to_graph(ADJUST_NMS_FEATURE_DIMS, graph, fail_if_no_method=False)
        if self.extract_color_transform:
            OptimizationTranslations.apply_method_to_graph(EXTRACT_COLOR_TRANSFROM, graph, fail_if_no_method=False)

        # ------------------------------------------------------------------------------
        #   PRE-PROCESSING
        # TODO: Move once optimizations are split into backend specific sections
        # ------------------------------------------------------------------------------
        # pre-process roi inputs
        if self.preprocess_roi_pool_inputs:
            OptimizationTranslations.apply_method_to_graph(PREPROCESS_ROI_POOL_INPUTS, graph, fail_if_no_method=False)

        # Performs pruning of cast Ops that are noop, if force_prune is set then all cast ops are pruned
        # TODO: remove separate noop call for casts when Cast supported by all backends
        OptimizationTranslations.apply_method_to_all_ops(REMOVE_CAST_NOOP, graph, force_prune=self.force_prune_cast_ops,
                                                         fail_if_no_method=False)

        # transition to NSC
        if self.perform_axes_to_spatial_first_order:
            OptimizationTranslations.apply_method_to_all_ops(AXES_TO_SPATIAL_FIRST_ORDER, graph)

        # Remove nodes disconnected from the main graph
        if graph.output_names:
            remove_disconnected_nodes(graph)

        # Performs an expansion on eltwise ops with > 2 inputs which should occur after all optimizations are attempted
        if self.chain_eltwise_ops:
            OptimizationTranslations.apply_method_to_graph(CHAIN_ELTWISE_OPS, graph, fail_if_no_method=False)

        # Optimize negations which typically apply to binary eltwise operations, hence adding after the optional
        # chaining step.
        OptimizationTranslations.apply_method_to_graph(OPTIMIZE_NEG, graph, fail_if_no_method=False)

        # remove NOOPs, which may include trivial permutes at this point
        # This may happen because some ops result in constant attributes that are absorbed by the layers
        OptimizationTranslations.apply_method_to_all_ops(REMOVE_NOOP, graph, fail_if_no_method=False)

        # Ensure matmul dims are handled/squashed as needed.
        if self.align_matmul_ranks:
            OptimizationTranslations.apply_method_to_all_ops(ALIGN_MATMUL_RANKS, graph, fail_if_no_method=False)

        # add op-specific quantization encodings to QParams Record.
        OptimizationTranslations.apply_method_to_all_ops(ADD_QPARAMS, graph, fail_if_no_method=False)

        # Apply unrolling to LSTM op
        # Note this optimization needs to happen before any LSTM pre-processing
        if self.unroll_lstm_time_steps:
            OptimizationTranslations.apply_method_to_graph(UNROLL_LSTM_TIME_STEPS, graph, fail_if_no_method=False)

        # Apply pre-processing to LSTM inputs after the time steps have been unrolled if applicable
        if self.preprocess_lstm_ops:
            OptimizationTranslations.apply_method_to_graph(PREPROCESS_LSTM_OPS, graph, fail_if_no_method=False)

        # Pre-processing of gather indices input
        if self.handle_gather_negative_indices:
            OptimizationTranslations.apply_method_to_all_ops(HANDLE_GATHER_NEGATIVE_INDICES, graph,
                                                             fail_if_no_method=False)

        # TODO Remove optimization once casts are properly removed in optimization stage
        if self.inject_cast_for_gather:
            OptimizationTranslations.apply_method_to_all_ops(INJECT_CAST_FOR_GATHER, graph, fail_if_no_method=False)

        # Prepares inputs in converter IR as parameters, as needed
        if self.prepare_inputs_as_params:
            OptimizationTranslations.apply_method_to_all_ops(PREPARE_INPUTS_AS_PARAMS, graph, fail_if_no_method=False)

        if self.dump_ir_graph:
            log_info("Dumping IR graph after all optimizations as IRGraph_after_optimizations.json")
            graph.dump_json("IRGraph_after_optimizations.json")

        # re-evaluate graph macs and params_count given optimization might have added/removed certain ops
        graph.reeval_macs_params()

        return graph

    def populate_quantization_params(self, ir_graph):

        def _extract_encoding_dict(name, enc):
            is_symmetric = enc['is_symmetric'].lower() == "true" if 'is_symmetric' in enc else False
            offset = int(-abs(enc['offset']) if 'offset' in enc else 0)

            # User quantization overrides may specify only scale/offset/bitwidth and then min/max can be calculated
            bitwidth_max = (2 ** (int(enc['bitwidth']) - 1))
            if all(key not in enc for key in ['min', 'max']) \
                    and all(key in enc for key in ['scale']):
                if is_symmetric:
                    enc['min'] = (-bitwidth_max + 1) * enc['scale']
                    enc['max'] = (bitwidth_max - 1) * enc['scale']
                else:
                    enc['min'] = offset * enc['scale']
                    enc['max'] = (((2 ** enc['bitwidth']) - 1) + offset) * enc['scale']

            # Symmetric weights should have 0 offset overridden with -bitwidth_max, or already be equal to -bitwidth_max
            if is_symmetric:
                if offset == 0:
                    offset = -bitwidth_max
                else:
                    if offset != -bitwidth_max:
                        raise ValueError("Invalid offset overridden for symmetric encodings got {}, expected {}."
                                         .format(offset, -bitwidth_max))

            # Offsets are sometimes stored as positive values, but converters require negative
            # Everything is optional except bw. Default to 0s if not provided.
            return {"name": name,
                    "min": float(enc["min"] if 'min' in enc else 0.0),
                    "max": float(enc["max"] if 'max' in enc else 0.0),
                    "bw": int(enc['bitwidth']),
                    "offset": offset,
                    "scale": float(enc['scale'] if 'scale' in enc else 0.0),
                    "is_symmetric": is_symmetric,
                    "overridden": True}

        def _adjust_bias_encoding(ir_graph):
            # The bias encoding in ir_graph.quantization_params corresponds to BiasAdd node as weights, we need to alter the name
            # 'weights' with 'bias' and add it to the params_encodings of the conv, deconv, matmul or fc node prior to the BiasAdd
            # so that the quantizer can get the bias encoding properly.
            for node in ir_graph.list_nodes():
                if node.op.hasattr('bias_op_name'):
                    _bias_op_name = node.op.bias_op_name

                    if _bias_op_name and _bias_op_name in ir_graph.quantization_params:
                        param_encodings = ir_graph.get_layer_quantization_param(_bias_op_name)[op_graph.QuantParams.PARAM_ENCODINGS]
                        if len(param_encodings) > 0:
                           _bias_encoding = param_encodings[0]
                           _bias_encoding['name'] = 'bias' # alter name 'weights' with 'bias'
                           ir_graph.add_quantization_params(node.op.name, param_encodings=_bias_encoding)

        q = ir_graph.user_quantization_overrides
        acts = q['activation_encodings']
        params = q['param_encodings']
        encoding_count = 0

        # Graph inputs are special cases because they aren't owned by a node until IR conversion
        inputs = ir_graph.get_input_nodes_to_graph()
        for i in inputs:
            n = i.op.name
            if n in acts:
                encoding_count += 1
                ir_graph.add_quantization_params(n, output_encodings=[_extract_encoding_dict(n, acts[n][0])])

        # Walk through the original source framework op->input mapping to find the weights
        for op_name, op in ir_graph.src_graph_op_info.items():
            param_encs = []

            inputs = op['inputs']
            node = None
            if op_name in ir_graph.nodes_by_name:
                node = ir_graph.nodes_by_name[op_name]
            if inputs:
                for idx, i in enumerate(inputs):
                    if i in params:
                        encoding_count += 1
                        # If this encoding name is bias op name, the name should be set be "bias"
                        if node != None and node.op.hasattr('bias_op_name') and node.op.bias_op_name == i:
                            param_encs.append(_extract_encoding_dict('bias', params[i][0]))
                        else:
                            param_encs.append(_extract_encoding_dict('weights', params[i][0]))

                ir_graph.add_quantization_params(op_name, param_encodings=param_encs)

        # adjust the bias encoding for 'matmul', 'fully_connected', 'convolution', 'deconvolution' ops.
        _adjust_bias_encoding(ir_graph)

        # Walk through the activations and lookup in the IR graph since folding, squashing, pruning
        # may have moved the activation names to new ops.
        for act in acts:
            act_encs = []
            if ir_graph.has_buffer(act):
                op = ir_graph.get_producer_op(act)
                encoding_count += 1
                act_encs.append(_extract_encoding_dict(act, acts[act][0]))
                ir_graph.add_quantization_params(op.name, output_encodings=act_encs)

        log_info('Processed '+ str(encoding_count)+' quantization encodings')


class OptimizationTranslationBase(translation.Translation):
    """
    This class is to be used to perform graph optimizations such as: folding, squashing,pruning, etc. Additionally,
    it is also used to perform axis tracking and by default implements to spatial first order function
    (NCHW to NHWC, or TBF to BTF). Use this base class to get the default function and call register_method to add a new
    optimization. For eg: The OptimizeBatchnormTranslation overloads the axes_to_spatial_first_order to handle weights
    as well as adds a squash_batchnorm function and registers the method in the __init__ function.
    """
    def __init__(self):
        translation.Translation.__init__(self)
        self.register_method(AXES_TO_SPATIAL_FIRST_ORDER, self.axes_to_spatial_first_order)
        self.register_method(MERGE_LOW_LEVEL_OPS_TO_LAYERS, self.merge_low_level_ops_to_layers)

    def axes_to_spatial_first_order(self, node, graph):
        """
        Performs axis permutations(as needed) to get a spatial first order.

        Note: The eltwise_...() function that gets called re-populates the node's buffer "axis_format" and "shape" from
        source framework to the destination for certain ranks. If an overload of this function is done for a child class
        and this eltwise_...() function is not called make sure to understand and implement these changes to avoid
        conversion errors.

        :param node: an OpNode object to optimize from the IR graph
        :param graph: an IROpgraph object

        """
        AxisTracker.eltwise_to_spatial_first_order(node, graph)

    def merge_low_level_ops_to_layers(self, graph):
        """"
        When overloaded in the child class, it is implemented to merge to the low level ops to layers.

        """
        pass


# ------------------------------------------------------------------------------------------------------------------
#   Graph Optimizations
# ------------------------------------------------------------------------------------------------------------------
def register_graph_optimization(graph_optimization_method):
    """
    For anything decorated with register in this module, the class along with its op_type is registered in
    a TranslationBank
    :param graph: a concrete class for a given optimization
    """
    return graph_optimization_method


@register_graph_optimization
def remove_disconnected_nodes(graph):
    """Removes nodes with all its outputs unconsumed from the graph."""
    all_ops = set(graph.nodes_in_order)
    connected_ops = set()
    queue = []
    graph_output_nodes = graph.get_output_nodes_of_graph()

    if graph_output_nodes:
        queue.extend(graph_output_nodes)
        # Find nodes from Output to Input Op
        while queue:
            node = queue.pop(0)
            connected_ops.add(node)

            # Add input nodes for the node
            node_inputs = graph.get_op_input_nodes(node)
            new_nodes = [node_ for node_ in node_inputs if (node_ not in connected_ops and node_ not in queue)]
            queue.extend(new_nodes)

    else:
        # Ensure input nodes have consumers before adding them to queue
        input_nodes = graph.get_input_nodes_to_graph()
        input_nodes = [node for node in input_nodes if graph.get_buffer(node.output_names[0]).consumers]
        queue.extend(input_nodes)
        # Find nodes from Input Op to outputs
        while queue:
            node = queue.pop(0)
            connected_ops.add(node)

            # Add input nodes for the node, this will add the Constant input Ops that will be otherwise missed
            node_inputs = graph.get_op_input_nodes(node)
            new_nodes = [node for node in node_inputs if node not in connected_ops]
            for new_node in new_nodes:
                queue.insert(0, new_node)

            # Extend the queue with output nodes
            node_outputs = graph.get_op_output_nodes(node)
            new_nodes = [node for node in node_outputs if node not in queue]
            queue.extend(new_nodes)

    disconnected_nodes = all_ops - connected_ops
    prunable_node_names = [node.op.name for node in disconnected_nodes]
    if disconnected_nodes:
        log_debug("Pruning Disconnected nodes {}".format(prunable_node_names))

    for node in disconnected_nodes:
        try:
            graph.prune(node, force_remove=True)
        except Exception as e:
            log_error("Cannot find node {}".format(node.op.name))
            raise e

    if not graph.list_nodes():
        raise ValueError("After pruning disconnected nodes, this model is empty.")

    return graph


# ------------------------------
# Util used for common squashing
# ------------------------------
def squash_node_into_nn_node(graph, matched_node_list):
    """
    Squashes a node into an NN node. This can be done by accounting for the node's operation in arithmetic adjustments
    to the NN node's weights and biases. Intended use is for Elementwise or Scale ops that follow an NN op.
    :param graph: The IROpGraph object
    :param matched_node_list: the list of nodes that contain elementwise or scale ops, have a constant input, and are
                              preceded by a node that contains an NN op
    """

    OPS_HAVING_BIAS_SUM = [
        op_adapter.ElementwiseSumOp.TRANSLATION_KEY
    ]
    OPS_HAVING_BIAS_SUB = [
        op_adapter.ElementwiseSubOp.TRANSLATION_KEY
    ]

    OPS_HAVING_WEIGHTS_PRODUCT = [
        op_adapter.ElementwiseProductOp.TRANSLATION_KEY
    ]
    OPS_HAVING_WEIGHTS_DIV = [
        op_adapter.ElementwiseDivOp.TRANSLATION_KEY
    ]
    OPS_HAVING_WEIGHTS_AND_BIASES = [
        op_adapter.ScaleOp.TRANSLATION_KEY
    ]

    for node_tuple in matched_node_list:

        # collect previous and current op information
        node = node_tuple[0]
        node_type = node.op.type
        nn_buf, nn_op, const_op = None, None, None
        for name in node.input_names:
            input_buf = graph.get_buffer(name)
            input_op = graph.get_producer_op(name)
            if hasattr(input_op, "weights") or hasattr(input_op, "bias"):
                # temp fix to avoid squashing of eltwise Ops into Matmul
                # TODO: Remove once Matmul Opdef is updated to support bias attribute
                if input_op.type == op_adapter.MatMulOp.TRANSLATION_KEY:
                    return
                nn_buf = input_buf
                nn_op = input_op
            elif input_op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                const_op = input_op

        if nn_op is None:
            raise ValueError("Failed to retrieve NN op to squash {} node {} into.".format(node_type, node.op.name))

        # Fails to find const_op in ScaleOp case because weights and biases are attributes and not inputs
        if const_op is None and node_type != op_adapter.ScaleOp.TRANSLATION_KEY:
            raise ValueError("Failed to retrieve const op to squash {} node {} into.".format(node_type, node.op.name))

        if nn_buf.axis_format == AxisTracker.AxisFormat.NCS:
            if nn_op.hasattr("weights") and len(nn_op.weights.shape) == 4:
                # weights are not yet transposed as that happens in axes_to_spatial_first later,
                # so we need to transpose for broadcasting to handle non-square kernel and then revert
                if nn_op.type in [op_adapter.ConvolutionOp.TRANSLATION_KEY,
                                  op_adapter.DepthwiseConvolutionOp.TRANSLATION_KEY]:
                    nn_op.weights = numpy.transpose(nn_op.weights, AxisTracker.AxisFormat.OIHW_TO_HWIO)
                elif nn_op.type == op_adapter.DeconvolutionOp.TRANSLATION_KEY:
                    nn_op.weights = numpy.transpose(nn_op.weights, AxisTracker.AxisFormat.IOHW_TO_HWIO)
            if const_op is not None and len(const_op.tensor.shape) == 4:
                const_op.tensor = numpy.transpose(const_op.tensor, AxisTracker.AxisFormat.NCS_TO_NSC)

        # separate conditionals according to which arithmetic operation needs to happen
        if node_type in OPS_HAVING_BIAS_SUM:
            scale_bias = const_op.tensor
            nn_op.bias = numpy.atleast_1d((nn_op.bias + scale_bias).squeeze())
        elif node_type in OPS_HAVING_BIAS_SUB:
            scale_bias = const_op.tensor
            nn_op.bias = numpy.atleast_1d((nn_op.bias - scale_bias).squeeze())
        elif node_type in OPS_HAVING_WEIGHTS_PRODUCT:
            scale_weights = const_op.tensor
            nn_op.weights = nn_op.weights * scale_weights
            nn_op.bias = numpy.atleast_1d((nn_op.bias * scale_weights).squeeze())
        elif node_type in OPS_HAVING_WEIGHTS_DIV:
            scale_weights = const_op.tensor
            nn_op.weights = nn_op.weights / scale_weights
            nn_op.bias = numpy.atleast_1d((nn_op.bias / scale_weights).squeeze())
        elif node_type in OPS_HAVING_WEIGHTS_AND_BIASES:
            scale_weights = node.op.weights
            scale_bias = node.op.bias
            nn_op.weights = nn_op.weights * scale_weights
            nn_op.bias = numpy.atleast_1d(((nn_op.bias * scale_weights) + scale_bias).squeeze())
        else:
            raise ValueError("Squashing {} node {} into {} node {} unsupported.".format(node_type, node.op.name,
                                                                                        nn_op.type, nn_op.name))

        if nn_buf.axis_format == AxisTracker.AxisFormat.NCS:
            if nn_op.hasattr("weights") and len(nn_op.weights.shape) == 4:
                if nn_op.type in [op_adapter.ConvolutionOp.TRANSLATION_KEY,
                                  op_adapter.DepthwiseConvolutionOp.TRANSLATION_KEY]:
                    nn_op.weights = numpy.transpose(nn_op.weights, AxisTracker.AxisFormat.HWIO_TO_OIHW)
                elif nn_op.type == op_adapter.DeconvolutionOp.TRANSLATION_KEY:
                    nn_op.weights = numpy.transpose(nn_op.weights, AxisTracker.AxisFormat.HWIO_TO_IOHW)
            if const_op is not None and len(const_op.tensor.shape) == 4:
                const_op.tensor = numpy.transpose(const_op.tensor, AxisTracker.AxisFormat.NSC_TO_NCS)

        log_debug2(code_to_message.get_debugging_message("DEBUG_SQUASH_INTO_NN_NODE")
                   (node_type, node.op.name, nn_op.type, nn_op.name))
        graph.squash(node, input_name=nn_buf.name)


def validate_eltwise_pattern(graph, nodes_tuple, mode):
    """
    Common function to validate if pattern is squashable
    :param graph: the IROpGraph
    :param nodes_tuple: the matched list of nodes
    :param mode: either bias or weight. Use to determine if squashing is
                 eltwise[add|sub] or eltwise[prod|div] respectively.
    :return:
    """

    OPS_HAVING_WEIGHTS_AND_BIASES_AS_INPUTS = [
        op_adapter.ConvolutionOp.TRANSLATION_KEY,
        op_adapter.DeconvolutionOp.TRANSLATION_KEY,
        op_adapter.DepthwiseConvolutionOp.TRANSLATION_KEY
    ]

    node = nodes_tuple[0]
    nn_buf, nn_op, const_op = None, None, None
    for name in node.input_names:
        input_op = graph.get_buffer(name).producer.op
        # verify that one of the inputs is constant and the other input is produced by nn_type op(BN, FC, Conv/Deconv)
        if input_op.type in OPS_HAVING_WEIGHTS_AND_BIASES_AS_INPUTS:
            # Squashing elementwise operations into these nodes is handled in their respective optimizations classes
            return False
        elif input_op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            const_op = input_op
        elif (mode == "weights" and hasattr(input_op, "weights") and hasattr(input_op, "bias")) or \
                (mode == "bias" and hasattr(input_op, "bias")):
            if len(graph.get_buffer(name).consumers) != 1:
                # Unable to squash into nn_op which has more than one consumer
                return False
            nn_op = input_op
            nn_buf = graph.get_buffer(name)

    # For mode:weights
    #      Only valid to squash if the nn_op has act output that are broadcastable with the scale weights AND
    #      the scale weights are same rank with nn_op bias and broadcastable
    # For mode:bias
    #      Only valid if the nn_op has a bias with the same rank as const_op and broadcastable
    if nn_op is not None and const_op is not None:
        const_shape = const_op.tensor.shape
        const_shape_squeezed = numpy.atleast_1d(const_op.tensor.squeeze()).shape
        if mode == 'bias':
            if len(const_shape_squeezed) == len(nn_op.bias.shape) and \
                    translation_utils.broadcastable(nn_op.bias.shape, const_shape_squeezed):
                return True
        elif mode == 'weights':
            nn_buf_shape = nn_buf.get_buf_dims()
            axis_order = graph.src_axis_order
            input_ir_shapes = [axis_order.permute_shape_to_ir(nn_buf_shape),
                               axis_order.permute_shape_to_ir(const_shape)]
            # Note: verify with the ir shapes for inputs since this is done pre axis-tracking
            if translation_utils.broadcastable(*input_ir_shapes) and \
                    (len(const_shape_squeezed) == len(nn_op.bias.shape) and
                     translation_utils.broadcastable(nn_op.bias.shape, const_shape_squeezed)):
                return True
    return False


def add_or_broadcast_bias(node, graph, output_depth):
    weights_buffer = graph.get_buffer(node.input_names[1])
    if len(node.input_names) < 3:
        bias_tensor = numpy.zeros([output_depth], dtype=numpy.float32)
        bias_op_name = node.op.name + "_bias"
        bias_op = op_adapter.ConstantOp(bias_op_name, tensor=bias_tensor.copy())
        conv_idx = graph.list_nodes().index(node)
        graph.add(bias_op, [], [bias_op_name], axis_formats=[AxisTracker.AxisFormat.ANY], idx=conv_idx)
        graph.get_buffer(bias_op_name).consumers.add(node)
        node.input_names.append(bias_op_name)
    else:
        bias_buffer = graph.get_buffer(node.input_names[2])
        # Represents case where broadcasting biases is required
        if bias_buffer.shape[0] < output_depth:
            bias_const_node = bias_buffer.producer
            if len(bias_const_node.op.tensor) != 1:
                raise ValueError("Unable to broadcast bias tensor for node {}".format(node.op.name))
            bias_const_node.op.tensor = numpy.repeat(bias_const_node.op.tensor, weights_buffer.shape[3])
            bias_buffer.shape = list(bias_const_node.op.tensor.shape)


def squash_eltwise_into_conv(graph, conv_node):
    conv_output_buffer = graph.get_buffer(conv_node.output_names[0])
    eltwise_node = list(conv_output_buffer.consumers)[0]
    # Find and assign the const_op from eltwise_node's input_names
    const_op = None
    for name in eltwise_node.input_names:
        input_op = graph.get_producer_op(name)
        if input_op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            const_op = input_op

    # Ensure the constant operation has the proper squash shape based on source axis order
    const_tensor = const_op.tensor
    if conv_output_buffer.axis_format == AxisTracker.AxisFormat.NCS and len(const_op.tensor.shape) == 4:
        const_tensor = numpy.transpose(const_tensor, AxisTracker.AxisFormat.NCS_TO_NSC)

    bias_producer = graph.get_buffer(conv_node.input_names[2]).producer
    bias_tensor = bias_producer.op.tensor

    # Apply the const_node's tensor to the conv_node's bias according to type of elementwise operation
    if eltwise_node.op.type == op_adapter.ElementwiseSumOp.TRANSLATION_KEY:
        bias_tensor = numpy.atleast_1d((bias_tensor + const_tensor).squeeze())
    elif eltwise_node.op.type == op_adapter.ElementwiseSubOp.TRANSLATION_KEY:
        bias_tensor = numpy.atleast_1d((bias_tensor - const_tensor).squeeze())
    else:
        # Only ElementwiseProduct/DivOp require static weights, so extract the static weights only in these cases
        weights_producer = graph.get_buffer(conv_node.input_names[1]).producer
        weights_tensor = weights_producer.op.tensor
        if eltwise_node.op.type == op_adapter.ElementwiseProductOp.TRANSLATION_KEY:
            weights_tensor = weights_tensor * const_tensor
            bias_tensor = numpy.atleast_1d((bias_tensor * const_tensor).squeeze())
        elif eltwise_node.op.type == op_adapter.ElementwiseDivOp.TRANSLATION_KEY:
            weights_tensor = weights_tensor / const_tensor
            bias_tensor = numpy.atleast_1d((bias_tensor / const_tensor).squeeze())
        weights_producer.op.tensor = weights_tensor

    # Reincorporate the new bias and squash the elementwise operation
    bias_producer.op.tensor = bias_tensor
    log_debug2(code_to_message.get_debugging_message("DEBUG_SQUASH_INTO_NN_NODE")
               (eltwise_node, eltwise_node.op.name, conv_node.op.type, conv_node.op.name))
    graph.squash(eltwise_node, input_name=conv_output_buffer.name)


def validate_conv_eltwise_pattern(graph, conv_node, eltwise_type):
    conv_node_output_buffer = graph.get_buffer(conv_node.output_names[0])
    if len(conv_node_output_buffer.consumers) != 1 or \
            list(conv_node_output_buffer.consumers)[0].op.type != eltwise_type:
        return False

    eltwise_node = list(conv_node_output_buffer.consumers)[0]

    # Find the constant op from input_names of the eltwise_node
    const_op = None
    for name in eltwise_node.input_names:
        input_op = graph.get_producer_op(name)
        if input_op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            const_op = input_op

    # Constant op was not found, so we cannot squash this elementwise operation
    if const_op is None:
        return False

    # Scalar products are able to be squashed into convolution weights
    if eltwise_node.op.type == op_adapter.ElementwiseProductOp.TRANSLATION_KEY:
        return len(const_op.tensor.shape) == 1

    const_shape_squeezed = numpy.atleast_1d(const_op.tensor.squeeze()).shape
    bias_shape = graph.get_buffer(conv_node.input_names[2]).shape
    # Const shape and bias shape should have the same rank and be broadcastable
    return len(const_shape_squeezed) == len(bias_shape) and \
        translation_utils.broadcastable(bias_shape, const_shape_squeezed)


def prepare_conv_inputs_as_params(graph, conv_node):
    weights_buffer = graph.get_buffer(conv_node.input_names[1])
    weights_node = weights_buffer.producer
    bias_buffer = graph.get_buffer(conv_node.input_names[2])
    bias_node = bias_buffer.producer
    if weights_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and \
            bias_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
        conv_node.op.weights = weights_node.op.tensor
        conv_node.op.bias = bias_node.op.tensor
        # Remove the weights/bias inputs from the IR graph
        graph.remove_node_as_consumer(conv_node, weights_buffer.name)
        graph.remove_node_as_consumer(conv_node, bias_buffer.name)
        conv_node.input_names = [conv_node.input_names[0]]


# ------------------------------------------------------------------------------------------------------------------
#   Translations
#   Note: each Optimization Concrete class has at a minimum 1 optimize function. i.e axes_to_spatial_first_order(..)
#         if more is needed for a given op, it needs to register that method_key and implement a function for it.
# ------------------------------------------------------------------------------------------------------------------
def register_layer_optimization(layer_translation):
    """
    For anything decorated with register in this module, the class along with its op_type is registered in
    a TranslationBank
    :param optimization_translation: a concrete class for a given optimization
    """
    OptimizationTranslations.register_translation(layer_translation(), layer_translation().op_type)
    return layer_translation


@register_layer_optimization
class OptimizeInputTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.InputOp.TRANSLATION_KEY
        self.register_method(EXTRACT_COLOR_TRANSFROM, self.extract_color_transform)

    @staticmethod
    def extract_color_transform(graph):
        """ Optional Optimization to create separate Op to handle color transformation pre-processing for network
            inputs
        """
        def validate_transformation(nodes_tuple):
            node_ = nodes_tuple[0]
            if node_.op.input_encoding_in != node_.op.input_encoding_out and \
                    node_.op.input_encoding_in not in [InputEncodings.TIME_SERIES, InputEncodings.OTHER]:
                return True
            return False

        sequence = [("input", (), ())]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_transformation)

        for node_tuple in matched_node_list:
            input_node = node_tuple[0]
            # adjust shape for input as that will be the expected shape after transformation
            color_transform_name = input_node.output_names[0] + "_post_transform"
            color_transform_output_shape = input_node.op.shape
            input_buf = graph.get_buffer(input_node.output_names[0])
            b, h, w, c = graph.src_axis_order.extract_spatial_dims(input_node.op.shape)
            if input_node.op.input_encoding_in in (InputEncodings.NV21, InputEncodings.NV12):
                # determine expected shape for yuv_(nv21|nv12)(width * height * 3 / 2)
                shape = int(h * w * (3 / 2))
                input_node.op.shape = [input_node.op.shape[0], shape]
                input_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                b, h, w, c = graph.src_axis_order.extract_spatial_dims(input_node.op.shape)
                input_node.op.shape = graph.src_axis_order.format_spatial_output_shape(b, h, w, 4)
            input_buf.set_buf_dims(input_node.op.shape)
            color_transform_op = op_adapter.ColorTransformOp(color_transform_name,
                                                             color_transform_output_shape,
                                                             input_encoding_in=input_node.op.input_encoding_in,
                                                             input_encoding_out=input_node.op.input_encoding_out)
            graph.inject(color_transform_op, input_name=input_node.output_names[0],
                         output_name=color_transform_name)
            log_debug2(code_to_message.get_debugging_message("DEBUG_COLOR_TRANSFORM_EXTRACTION")
                       (input_node.op.name, input_node.op.shape, input_node.op.input_encoding_in))

    def axes_to_spatial_first_order(self, node, graph):
        buf = graph.get_buffer(node.output_names[0])
        if buf.axis_format == AxisTracker.AxisFormat.NCS:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
            buf.axis_format = AxisTracker.AxisFormat.NSC
            node.op.shape = buf.shape
        elif buf.axis_format == AxisTracker.AxisFormat.TBF:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.TBF_TO_BTF)
            buf.axis_format = AxisTracker.AxisFormat.BTF
            node.op.shape = buf.shape
        elif buf.axis_format == AxisTracker.AxisFormat.OIHW:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.OIHW_TO_HWIO)
            buf.axis_format = AxisTracker.AxisFormat.HWIO
            node.op.shape = buf.shape


@register_layer_optimization
class OptimizeArgMaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ArgMaxOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            # If keep dims = 0 we must permute as it will remove dimensions
            if not node.op.keep_dims:
                graph.inject_implicit_permute(input_name, node.op.name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                AxisTracker.eltwise_to_spatial_first_order(node, graph)
                axis_map = graph.src_axis_order.permute_sequence[input_buf.rank() - 1]
                node.op.axis = axis_map[node.op.axis]


@register_layer_optimization
class OptimizeArgMinTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ArgMinOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            # If keep dims = 0 we must permute as it will remove dimensions
            if not node.op.keep_dims:
                graph.inject_implicit_permute(input_name, node.op.name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                AxisTracker.eltwise_to_spatial_first_order(node, graph)
                axis_map = graph.src_axis_order.permute_sequence[input_buf.rank() - 1]
                node.op.axis = axis_map[node.op.axis]


@register_layer_optimization
class OptimizeBatchnormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.BatchnormOp.TRANSLATION_KEY
        self.register_method(SQUASH_BATCHNORM, self.squash_batchnorm)

    def axes_to_spatial_first_order(self, node, graph):
        input_buf = graph.get_input_buffers(node)[0]
        if input_buf.rank() == 4:
            AxisTracker.image_to_spatial_first_order(node, graph)
        elif input_buf.rank() == 2 or input_buf.rank() == 3:
            output_buf = graph.get_output_buffers(node)[0]
            output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            if input_buf.rank() == 3:
                # add custom permute for 3D use-case. This input use-case is added for batchnorm-1D
                permute_order = [0, 2, 1]  # channel must be last
                AxisTracker.enforce_input_type(graph, node.input_names[0], node.op.name,
                                               AxisTracker.AxisFormat.NONTRIVIAL, permute_order)
                output_buf.shape = AxisTracker.permute_shape(output_buf.shape, permute_order)
        else:
            raise ValueError(code_to_message.get_error_message("ERROR_BATCHNORM_DIM_UNSUPPORTED")(input_buf.rank()))

    def merge_low_level_ops_to_layers(self, graph):
        def validate(nodes_tuple_):
            prod_node = nodes_tuple_[1]
            prod_node_input_op = graph.get_producer_op(prod_node.input_names[0])
            # previous must not be a Batchnorm and previous node must be a nn_node for sequence to match batchnorm
            if prod_node_input_op.type == op_adapter.BatchnormOp.TRANSLATION_KEY or \
                    not hasattr(prod_node_input_op, "weights") or not hasattr(prod_node_input_op, "bias"):
                return False

            mul_const_ip_node_ = nodes_tuple_[0]
            add_const_ip_node_ = nodes_tuple_[2]
            # batchnorm nodes require 1D weights/biases
            mul_const_ip_ = numpy.atleast_1d(mul_const_ip_node_.op.tensor.squeeze())
            add_const_ip_ = numpy.atleast_1d(add_const_ip_node_.op.tensor.squeeze())
            if len(mul_const_ip_.shape) != 1 or len(add_const_ip_.shape) != 1:
                return False
            return True

        sequence = [
                    ("constant", (), ()),
                    ("elementwise_product",
                        ("MATCH_BUFS_AT_INDEX", [("constant", 1)]),
                        ()),
                    ("constant", (), ()),
                    ("elementwise_sum",
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_product", 0),
                                                 ("constant", 1)]), ())
                   ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate)
        for nodes_tuple in matched_node_list:
            mul_const_ip_node = nodes_tuple[0]
            mul_node = nodes_tuple[1]
            add_const_ip_node = nodes_tuple[2]
            add_node = nodes_tuple[3]

            # batchnorm nodes require 1D weights/biases
            mul_const_ip = numpy.atleast_1d(mul_const_ip_node.op.tensor.squeeze())
            add_const_ip = numpy.atleast_1d(add_const_ip_node.op.tensor.squeeze())

            # Squashes the add node
            add_input_buffer = graph.get_input_buffers(add_node)[0]
            graph.squash(add_node, input_name=add_input_buffer.name)

            # Remove mul_node as consumer of const node's buffer
            graph.get_buffer(mul_const_ip_node.output_names[0]).consumers.remove(mul_node)
            # Remove const node from mul_node's input names
            mul_node.input_names.remove(mul_const_ip_node.output_names[0])

            batchnorm_op = op_adapter.BatchnormOp(None, weights=mul_const_ip, bias=add_const_ip)
            batchnorm_op.name = graph.naming_policy.get_op_name(batchnorm_op)
            batchnorm_op.weights = numpy.atleast_1d(batchnorm_op.weights.squeeze())
            batchnorm_op.bias = numpy.atleast_1d(batchnorm_op.bias.squeeze())
            graph.replace(mul_node.op, batchnorm_op)

    @staticmethod
    def squash_batchnorm(graph):
        def validate(nodes_tuple):
            bn_node_ = next(iter(graph.get_output_buffers(nodes_tuple[0])[0].consumers))
            bn_input_buffer_ = graph.get_input_buffers(bn_node_)[0]
            if bn_node_.op.compute_statistics:
                log_debug("InstanceNorm layer {} cannot be squashed", bn_node_.op.name)
                return False
            return bn_node_.op.type == op_adapter.BatchnormOp.TRANSLATION_KEY and bn_input_buffer_.rank() == 4

        sequences = [[("convolution",
                       ("MATCH_BUFS_AT_INDEX", [("constant", 1),
                                                ("constant", 2)]),
                       ("MATCH_NUM_BUFS", [("batchnorm", "ALL")]))],
                     [("depthwise_convolution",
                       ("MATCH_BUFS_AT_INDEX", [("constant", 1),
                                                ("constant", 2)]),
                       ("MATCH_NUM_BUFS", [("batchnorm", "ALL")]))]]

        for sequence in sequences:
            for node_tuple in graph.get_matched_nodes(sequence, validator=validate):
                # sanity check
                log_assert(len(node_tuple) == len(sequence),
                           "Pattern matching for squash batchnorm returned extra nodes. Got {} nodes, Expected {}.",
                           len(node_tuple), len(sequence))

                conv_node = node_tuple[0]
                bn_node = next(iter(graph.get_output_buffers(conv_node)[0].consumers))
                bn_input_buffer = graph.get_input_buffers(bn_node)[0]

                conv_node_weights_buffer = graph.get_buffer(conv_node.input_names[1])
                conv_node_weights_op = conv_node_weights_buffer.producer.op
                conv_node_weights = conv_node_weights_op.tensor

                # Extract bias from ConstantOp
                conv_node_bias_op = graph.get_buffer(conv_node.input_names[2]).producer.op
                conv_node_bias = conv_node_bias_op.tensor

                if conv_node_weights_buffer.axis_format == AxisTracker.AxisFormat.OIHW:
                    weights = numpy.transpose(conv_node_weights, AxisTracker.AxisFormat.OIHW_TO_HWIO)
                    weights = weights * bn_node.op.weights
                    weights = numpy.transpose(weights, AxisTracker.AxisFormat.HWIO_TO_OIHW)
                else:
                    weights = conv_node_weights * bn_node.op.weights

                conv_node_weights_op.tensor = weights
                conv_node_bias = numpy.atleast_1d(
                    (conv_node_bias * bn_node.op.weights + bn_node.op.bias).squeeze())

                conv_node_bias_op.tensor = conv_node_bias.copy()

                graph.add_quantization_params(conv_node.op.name, bn_params={"gamma": bn_node.op.gamma,
                                                                            "beta": bn_node.op.beta})
                graph.squash(bn_node, input_name=bn_input_buffer.name)
                log_debug2(code_to_message.get_debugging_message("DEBUG_BATCHNORM_SQUASH")(bn_node.op.name,
                                                                                           conv_node.op.type,
                                                                                           conv_node.op.name))


@register_layer_optimization
class OptimizeCastTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CastOp.TRANSLATION_KEY
        self.register_method(REMOVE_CAST_NOOP, self.remove_noop)

    @staticmethod
    def remove_noop(node, graph, force_prune=True):
        # TODO Properly identify and remove casts once datatypes are trackable in IR
        if node.op.from_type == node.op.to_type or force_prune:
            graph.squash(node, input_name=node.input_names[0], squash_into_next=True)


@register_layer_optimization
class OptimizeChannelShuffleTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ChannelShuffleOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeChannelShuffleTranslation, self).axes_to_spatial_first_order(node, graph)
        for buf in graph.get_input_buffers(node):
            log_debug("input {} {} {}", buf.name, buf.axis_format, buf.shape)
        for buf in graph.get_output_buffers(node):
            log_debug("output {} {} {}", buf.name, buf.axis_format, buf.shape)


@register_layer_optimization
class OptimizeColorTransformTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ColorTransformOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        buf = graph.get_buffer(node.output_names[0])
        if buf.axis_format == AxisTracker.AxisFormat.NCS:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
            buf.axis_format = AxisTracker.AxisFormat.NSC


@register_layer_optimization
class OptimizeConvolutionTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConvolutionOp.TRANSLATION_KEY
        self.register_method(PREPARE_BIASES, self.prepare_biases)
        self.register_method(PREPARE_INPUTS_AS_PARAMS, self.prepare_inputs_as_params)

    def prepare_biases(self, node, graph):
        weights_buffer = graph.get_buffer(node.input_names[1])
        output_depth = graph.src_axis_order.extract_conv_weights_dims(weights_buffer.shape)[-1]
        add_or_broadcast_bias(node, graph, output_depth)

    def prepare_inputs_as_params(self, node, graph):
        prepare_conv_inputs_as_params(graph, node)

    def axes_to_spatial_first_order(self, node, graph):
        if isinstance(graph.src_axis_order, (CaffeAxisOrder, OnnxAxisOrder)):
            input_buffers = graph.get_input_buffers(node)
            input_orders = [buf.axis_format for buf in input_buffers]

            # If the weights input is already NSC, transpose it to OIHW by using a transpose to NCS. Then, to HWIO.
            if input_orders[1] in [AxisTracker.AxisFormat.NSC,
                                   AxisTracker.AxisFormat.NONTRIVIAL]:
                # Inject an implicit permute to NCS, which is actually taking us back to OIHW
                graph.inject_implicit_permute(input_buffers[1].name, node.op.name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                input_buffers[1].axis_format = AxisTracker.AxisFormat.OIHW

                # Must update input_buffers after first injection of implicit permute
                input_buffers = graph.get_input_buffers(node)

                # Inject an implicit permute to HWIO from OIHW
                graph.inject_implicit_permute(input_buffers[1].name, node.op.name, AxisTracker.AxisFormat.HWIO,
                                              AxisTracker.AxisFormat.OIHW_TO_HWIO, [node.op.name])
                input_buffers[1].axis_format = AxisTracker.AxisFormat.HWIO

                # Update input_buffers and input_orders after second injection of implicit permute
                input_buffers = graph.get_input_buffers(node)
                input_orders = [buf.axis_format for buf in input_buffers]

            if any(order in input_orders for order in [AxisTracker.AxisFormat.NSC,
                                                       AxisTracker.AxisFormat.HWIO,
                                                       AxisTracker.AxisFormat.ANY,
                                                       AxisTracker.AxisFormat.NONTRIVIAL]):
                AxisTracker.image_to_spatial_first_order(node, graph)
            else:
                raise ValueError(code_to_message.get_error_message("ERROR_CONVOLUTION_UNEXPECTED_INPUT_ORDER")
                                 (input_orders))


@register_layer_optimization
class OptimizeConcatTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConcatOp.TRANSLATION_KEY
        self.register_method(FOLD_CONCATS, self.fold_concats)

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.eltwise_to_spatial_first_order(node, graph)
        buf = graph.get_buffer(node.output_names[0])
        if buf.axis_format != AxisTracker.AxisFormat.NONTRIVIAL:
            axis_map = graph.src_axis_order.permute_sequence[buf.rank() - 1]
            node.op.axis = axis_map[node.op.axis]

    @staticmethod
    def fold_concats(graph):
        def validate_concat_axis(nodes_tuple):
            concat_node_ = nodes_tuple[0]
            concat_node_input_bufs_ = graph.get_input_buffers(concat_node_)
            for buf_ in concat_node_input_bufs_:
                if buf_.producer.op.type == op_adapter.ConcatOp.TRANSLATION_KEY:
                    prev_concat_node_ = buf_.producer
                    # only fold concats with same axis
                    if prev_concat_node_.op.axis != concat_node_.op.axis:
                        log_debug2("Found concat node({}) with a concat input, but axis does not match for input ({}), "
                                   "{} != {} ", concat_node_.op.name, prev_concat_node_.op.name,
                                   prev_concat_node_.op.axis, concat_node_.op.axis)
                        return False

            return True

        sequence = [
                    ("concatenation",
                     ("FLEXIBLE_NUM_BUFS", [("concatenation", "ANY")]),
                     ()
                     )
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_concat_axis)

        for node_tuple in matched_node_list:
            concat_node = node_tuple[0]
            concat_node_input_bufs = graph.get_input_buffers(concat_node)

            for buf in concat_node_input_bufs:
                if buf.producer.op.type == op_adapter.ConcatOp.TRANSLATION_KEY:
                    prev_concat_buf = buf  # for readability
                    prev_concat_node = prev_concat_buf.producer

                    # remove prev concat as input from current concat and replace with prev concat's input names
                    prev_concat_inputs = prev_concat_node.input_names
                    idx = concat_node.input_names.index(prev_concat_buf.name)
                    concat_node.input_names.remove(prev_concat_buf.name)
                    # extend the inputs in the same index as prev concat
                    concat_node.input_names[idx:idx] = prev_concat_inputs

                    prev_concat_buf.consumers.remove(concat_node)

                    # we can prune the prev concat node if the current concat was the only consumer.
                    if len(prev_concat_buf.consumers) == 0:
                        graph.prune(prev_concat_node)

                    # remove prev concat as consumer for prev concat's input bufs and replace with current concat
                    for input_name in prev_concat_inputs:
                        input_buf = graph.get_buffer(input_name)
                        input_buf.consumers.add(concat_node)

                    log_debug2(code_to_message.get_debugging_message("DEBUG_CONCAT_FOLD")(prev_concat_node.op.name,
                                                                                          concat_node.op.name))


@register_layer_optimization
class OptimizeConstantTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConstantOp.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        output_buf = graph.get_buffer(node.output_names[0])

        # TODO Remove this code once limitations of AxisTracking are resolved
        # If the consumer of this buffer has another input with NSC format, and this buffer is 3D, it needs to be
        # padded with a 1 and have its constant operation permuted
        consumers = list(output_buf.consumers)
        if len(consumers) and output_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
            consumer_has_dimension_mismatch = [False] * len(consumers)
            for i, consumer in enumerate(consumers):
                for input_buffer in graph.get_input_buffers(consumer):
                    if input_buffer.axis_format == AxisTracker.AxisFormat.NSC and len(output_buf.shape) == 3:
                        consumer_has_dimension_mismatch[i] = True
                        break

            if all(consumer_has_dimension_mismatch):
                log_debug("All consumers of {} node {} have 4D-3D rank mismatch in inputs. Updating buffer {}.".format(
                    node.op.type, node.op.name, output_buf.name))
                # Capture tensor and prepare for placement in graph
                const_tensor = output_buf.producer.op.tensor
                const_tensor_shape = [1, *list(const_tensor.shape)]
                const_tensor = numpy.reshape(const_tensor, const_tensor_shape)
                # Modify the graph according to updated shape
                output_buf.producer.op.tensor = const_tensor
                output_buf.shape = const_tensor_shape
                output_buf.axis_format = AxisTracker.AxisFormat.NCS
            elif any(consumer_has_dimension_mismatch):
                # Remove consumers that need to be updated from current graph
                consumers_to_update = [consumer for i, consumer in output_buf.consumers if
                                       consumer_has_dimension_mismatch[i]]
                for consumer in consumers_to_update:
                    consumer.input_names.remove(output_buf.name)
                    output_buf.remove(consumer)
                # Create the new constant tensor
                const_tensor = output_buf.producer.op.tensor
                const_tensor_shape = [1, *list(const_tensor.shape)]
                const_tensor = numpy.reshape(const_tensor, const_tensor_shape)
                # Create the new 4D constant operation
                const_op_name = output_buf.name + "_4d"
                const_op = op_adapter.ConstantOp(const_op_name, const_tensor,
                                                 quantizable=output_buf.producer.op.quantizable)
                # Place the new 4D constant operation in graph
                log_debug("At least one, but not all consumers of buffer {} have 4D-3D dimension mismatch. Creating "
                          "a new constant 4D constant operation named {}.".format(output_buf.name, const_op_name))
                graph.add(const_op, [], [const_op_name], axis_formats=[AxisTracker.AxisFormat.NCS])
                graph.get_buffer(const_op_name).consumers = consumers_to_update
                for consumer in consumers_to_update:
                    consumer.input_names.add(const_op_name)

        # Permute the constant data if necessary
        if output_buf.axis_format == AxisTracker.AxisFormat.NCS:
            node.op.tensor = numpy.ascontiguousarray(numpy.transpose(node.op.tensor, AxisTracker.AxisFormat.NCS_TO_NSC))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
            output_buf.axis_format = AxisTracker.AxisFormat.NSC
        elif output_buf.axis_format == AxisTracker.AxisFormat.TBF:
            node.op.tensor = numpy.ascontiguousarray(numpy.transpose(node.op.tensor, AxisTracker.AxisFormat.TBF_TO_BTF))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.TBF_TO_BTF)
            output_buf.axis_format = AxisTracker.AxisFormat.BTF
        elif output_buf.axis_format == AxisTracker.AxisFormat.OIHW:
            node.op.tensor = numpy.ascontiguousarray(numpy.transpose(node.op.tensor, AxisTracker.AxisFormat.OIHW_TO_HWIO))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.OIHW_TO_HWIO)
            output_buf.axis_format = AxisTracker.AxisFormat.HWIO
        elif output_buf.axis_format == AxisTracker.AxisFormat.IOHW:
            node.op.tensor = numpy.ascontiguousarray(numpy.transpose(node.op.tensor, AxisTracker.AxisFormat.IOHW_TO_HWIO))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.IOHW_TO_HWIO)
            output_buf.axis_format = AxisTracker.AxisFormat.HWIO

    @staticmethod
    def remove_noop(node, graph):
        # Prune this node if it's an input to a weight layer and was used internally
        if getattr(graph, "weights", None) and getattr(graph.weights, "consumed", None) \
                and graph.weights.consumed(node.output_names[0]):
            log_debug(code_to_message.get_debugging_message("DEBUG_CONSTANT_PRUNED")(node.output_names[0]))
            graph.prune(node)


@register_layer_optimization
class OptimizeConvertTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConvertOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeCropTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CropOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        target_buf = None
        if len(node.input_names) > 1:
            target_name = node.input_names[1]
            target_buf = graph.get_buffer(target_name)
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC and (target_buf is None or target_buf.rank() == 4):
            node.op.offsets = AxisTracker.permute_shape(node.op.offsets, AxisTracker.AxisFormat.NCS_TO_NSC)
            node.op.counts = AxisTracker.permute_shape(node.op.counts, AxisTracker.AxisFormat.NCS_TO_NSC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and (target_buf is None or target_buf.rank() == 3):
            node.op.offsets = AxisTracker.permute_shape(node.op.offsets, [1, 2, 0])
            node.op.counts = AxisTracker.permute_shape(node.op.counts, [1, 2, 0])
        elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            node.op.offsets = AxisTracker.permute_shape(node.op.offsets, AxisTracker.AxisFormat.TBF_TO_BTF)
            node.op.counts = AxisTracker.permute_shape(node.op.counts, AxisTracker.AxisFormat.TBF_TO_BTF)
        AxisTracker.eltwise_to_spatial_first_order(node, graph)


@register_layer_optimization
class OptimizeCrossCorrelationTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CrossCorrelationOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeCustomOpTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CustomOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeCustomOpTranslation, self).axes_to_spatial_first_order(node, graph)

        for i, buf in enumerate(graph.get_output_buffers(node)):
            node.op.output_dims[i] = buf.shape


@register_layer_optimization
class OptimizeDeconvolutionTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.DeconvolutionOp.TRANSLATION_KEY
        self.register_method(PREPARE_BIASES, self.prepare_biases)
        self.register_method(PREPARE_INPUTS_AS_PARAMS, self.prepare_inputs_as_params)

    def axes_to_spatial_first_order(self, node, graph):
        if isinstance(graph.src_axis_order, (CaffeAxisOrder, OnnxAxisOrder)):
            input_buffers = graph.get_input_buffers(node)
            input_orders = [buf.axis_format for buf in input_buffers]

            # If the weights input is already NSC, transpose it to IOHW by using a transpose to NCS. Then, to HWIO.
            if input_orders[1] in [AxisTracker.AxisFormat.NSC,
                                   AxisTracker.AxisFormat.NONTRIVIAL]:
                # Inject an implicit permute to NCS, which is actually taking us back to IOHW
                graph.inject_implicit_permute(input_buffers[1].name, node.op.name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                input_buffers[1].axis_format = AxisTracker.AxisFormat.IOHW

                # Must update input_buffers after first injection of implicit permute
                input_buffers = graph.get_input_buffers(node)

                # Inject an implicit permute to HWIO from IOHW
                graph.inject_implicit_permute(input_buffers[1].name, node.op.name, AxisTracker.AxisFormat.HWIO,
                                              AxisTracker.AxisFormat.IOHW_TO_HWIO, [node.op.name])
                input_buffers[1].axis_format = AxisTracker.AxisFormat.HWIO

                # Update input_buffers and input_orders after second injection of implicit permute
                input_buffers = graph.get_input_buffers(node)
                input_orders = [buf.axis_format for buf in input_buffers]

            if any(order in input_orders for order in [AxisTracker.AxisFormat.NSC,
                                                       AxisTracker.AxisFormat.HWIO,
                                                       AxisTracker.AxisFormat.ANY,
                                                       AxisTracker.AxisFormat.NONTRIVIAL]):
                AxisTracker.image_to_spatial_first_order(node, graph)
            else:
                raise ValueError(code_to_message.get_error_message("ERROR_DECONVOLUTION_UNEXPECTED_INPUT_ORDER")
                                 (input_orders))

    def prepare_biases(self, node, graph):
        weights_buffer = graph.get_buffer(node.input_names[1])
        output_depth = graph.src_axis_order.extract_deconv_weights_dims(weights_buffer.shape)[-1] * node.op.groups
        add_or_broadcast_bias(node, graph, output_depth)

    def prepare_inputs_as_params(self, node, graph):
        prepare_conv_inputs_as_params(graph, node)


@register_layer_optimization
class OptimizeDepthwiseConvolutionTranslation(OptimizeConvolutionTranslation):
    def __init__(self):
        OptimizeConvolutionTranslation.__init__(self)
        self.op_type = op_adapter.DepthwiseConvolutionOp.TRANSLATION_KEY

    def prepare_biases(self, node, graph):
        output_depth = node.op.groups
        add_or_broadcast_bias(node, graph, output_depth)


@register_layer_optimization
class OptimizeDetectionOutTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.DetectionOutputOp.TRANSLATION_KEY
        self.register_method(FOLD_CONCATS, self.fold_concats)
        self.register_method(MATCH_CAFFE_SSD_TO_TF, self.caffe_ssd_to_tf)

    @staticmethod
    def fold_concats(graph):
        def process_ssd_priorbox_concat_layer(input_buffers_):
            concatenated_priorbox_data = []
            concatenated_priorbox_cz_data = []
            concatenated_priorbox_variance = []
            scale_factors_ = input_buffers_[0].producer.op.scale_factors
            for input_buffer in input_buffers_:
                priorbox_op = input_buffer.producer.op
                concatenated_priorbox_data.extend(priorbox_op.priorbox_box_output[0])
                concatenated_priorbox_variance.extend(priorbox_op.priorbox_box_output[1])
                concatenated_priorbox_cz_data.extend(priorbox_op.priorbox_box_cz_output)
                if scale_factors_ != priorbox_op.scale_factors:
                    # Currently only support 1 set of scale factor for priorboxes.
                    raise ValueError(code_to_message.get_error_message("ERROR_INVALID_PRIORBOX_VARIANCES")
                                     (scale_factors_, input_buffers_[0].producer.op.name,
                                      priorbox_op.scale_factors, priorbox_op.name))

            return concatenated_priorbox_data + concatenated_priorbox_variance, concatenated_priorbox_cz_data, \
                   scale_factors_

        sequence = [
            ("concatenation",
                ("FLEXIBLE_NUM_BUFS", [("noop", "ALL")]),  # noop here since all priorboxes are mapped to noopOp
                ("MATCH_NUM_BUFS", [("detection_output", "ALL")])
             )
        ]
        matched_node_list = graph.get_matched_nodes(sequence)

        for node_tuple in matched_node_list:
            concat_node = node_tuple[0]
            concat_input_buffers = graph.get_input_buffers(concat_node)
            concat_output_buffer = graph.get_output_buffers(concat_node)[0]
            detection_out_node = concat_output_buffer.consumers.pop()
            priorbox_data, priorbox_cz_data, scale_factors = process_ssd_priorbox_concat_layer(concat_input_buffers)
            detection_out_node.op.priorbox_data = priorbox_data
            detection_out_node.op.priorbox_center_size_data = priorbox_cz_data
            # order determined per caffe/util/bbox_util.cpp
            detection_out_node.op.scale_x = scale_factors[0]
            detection_out_node.op.scale_y = scale_factors[1]
            detection_out_node.op.scale_w = scale_factors[2]
            detection_out_node.op.scale_h = scale_factors[3]

            # remove concat node.
            detection_out_node.input_names.remove(concat_output_buffer.name)
            graph.prune(concat_node)

            # remove priorboxes
            for buf in concat_input_buffers:
                graph.prune(buf.producer)

            log_debug2(code_to_message.get_debugging_message("DEBUG_DETECTIONOUT_FOLDING")(concat_node.op.name,
                                                                                           detection_out_node.op.name))

    @staticmethod
    def caffe_ssd_to_tf(graph):
        sequence = [
            ("detection_output",
                ("MATCH_NUM_BUFS", [("reshape", "ANY"), ("concatenation", "ANY")]),  # flattened scores and boxes
                ()
             )
        ]
        matched_node_list = graph.get_matched_nodes(sequence)

        for node_tuple in matched_node_list:
            detection_out_node = node_tuple[0]
            for input_name in detection_out_node.input_names:
                node = graph.get_producer_node(input_name)
                if node.op.type == op_adapter.ReshapeOp.TRANSLATION_KEY:
                    reshape_node = node
                elif node.op.type == op_adapter.ConcatOp.TRANSLATION_KEY:
                    concat_node = node
                else:
                    raise ValueError(code_to_message.get_error_message("ERROR_DETECTIONOUT_UNKNOWN_INPUTS")
                                     (node.op.type))

            # 0. Verify valid anchors/priorboxes
            log_assert(detection_out_node.op.code_type == op_adapter.DetectionOutputOp.PriorBoxType.CENTER_SIZE,
                       "DetectionOut Op only supports center size code type. Got {}".
                       format(detection_out_node.op.code_type))

            # 1. Pre-process steps
            # Caffe score input is flattened, remove reshape to match shape [batch, num_anchors, num_classes]
            reshape_output_buffer = graph.get_output_buffers(reshape_node)[0]
            detection_out_node.input_names.remove(reshape_output_buffer.name)
            detection_out_node.input_names.insert(0, reshape_node.input_names[0])
            graph.get_buffer(reshape_node.input_names[0]).consumers.add(detection_out_node)

            reshape_output_buffer.consumers.remove(detection_out_node)
            # remove reshape node if applicable.
            if len(reshape_output_buffer.consumers) == 0:
                graph.prune(reshape_node)

            # Caffe boxes(location) data is also flattened. Reshape to [batch, num_boxes, 4]
            concat_output_buffer = graph.get_output_buffers(concat_node)[0]
            concat_buf_shape = concat_output_buffer.shape
            # add reshape node
            reshape_name = concat_node.op.name + "_preprocess_reshape"
            reshape_op = op_adapter.ReshapeOp(reshape_name, output_shape=[concat_buf_shape[0],
                                                                          int(concat_buf_shape[1] / 4),
                                                                          4])
            graph.inject(reshape_op, input_name=concat_node.output_names[0], output_name=reshape_name,
                         consumer_names=detection_out_node.output_names)

            # DetectionOut in IR has priorboxes as param, need to add those to input instead
            detection_out_name = detection_out_node.op.name
            detection_out_node_idx = graph.nodes_in_order.index(detection_out_node)
            prior_box_name = detection_out_name + "_anchors"
            pbox_data = numpy.asarray(detection_out_node.op.priorbox_center_size_data, dtype=numpy.float32)\
                        .reshape(int(len(detection_out_node.op.priorbox_center_size_data)/4), 4)
            prior_box_op = op_adapter.ConstantOp(name=prior_box_name, tensor=pbox_data)
            graph.add(prior_box_op, input_names=[], output_names=[prior_box_name], idx=detection_out_node_idx-1)
            detection_out_node.input_names.append(prior_box_name)

            # Caffe Ssd scales is the reciprocal compared to TF scales
            detection_out_node.op.scale_y = 1 / detection_out_node.op.scale_y
            detection_out_node.op.scale_x = 1 / detection_out_node.op.scale_x
            detection_out_node.op.scale_h = 1 / detection_out_node.op.scale_h
            detection_out_node.op.scale_w = 1 / detection_out_node.op.scale_w

            # 2. Change DetectionOut's single output to multiple. Outputs:
            #    Expected: scores[1, max_num_det], boxes[1, max_num_det, 4], classes[1, max_num_det], num_det[batch],
            #    Caffe Style: 1 output of shape [1, 1, max_num_det, 7]
            #                   7(last dim above): [image_batch, label, confidence, x_min, y_min, x_max, y_max]
            detection_out_buf = graph.get_buffer(detection_out_node.output_names[0])
            boxes_shape = [detection_out_buf.shape[0], detection_out_node.op.keep_top_k, 4]  # [batch, max_num_detections, 4)
            boxes_name = detection_out_name + "_boxes"
            boxes_buf = op_graph.Buffer(boxes_name, boxes_shape, detection_out_node)
            graph.buffers[boxes_name] = boxes_buf

            scores_name = detection_out_name + "_scores"
            scores_buf = op_graph.Buffer(scores_name, boxes_shape[:-1], detection_out_node)
            graph.buffers[scores_name] = scores_buf

            classes_name = detection_out_name + "_classes"
            classes_buf = op_graph.Buffer(classes_name, boxes_shape[:-1], detection_out_node)
            graph.buffers[classes_name] = classes_buf

            num_det_name = detection_out_name + "_num_detections"
            num_det_buf = op_graph.Buffer(num_det_name, [boxes_shape[0]], detection_out_node)
            graph.buffers[num_det_name] = num_det_buf

            del graph.buffers[detection_out_node.output_names[0]]
            detection_out_node.output_names = [boxes_name, scores_name, classes_name, num_det_name]

            log_debug2(code_to_message.get_debugging_message("DEBUG_DETECTIONOUT_CAFFE_TO_TF_STYLE")
                       (detection_out_node.op.name))

@register_layer_optimization
class OptimizeDequantizeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.DequantizeOp.TRANSLATION_KEY
        self.register_method(REMOVE_QUANT_NODES, self.remove_quant_nodes)

    @staticmethod
    def remove_quant_nodes(node, graph):
        graph.squash(node, input_name=node.input_names[0])
        log_debug("Remove dequantize op {}".format(node.op.name))

@register_layer_optimization
class OptimizeElementwiseAndTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseAndOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseDivTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseDivOp.TRANSLATION_KEY
        self.register_method(SQUASH_DIV, self.squash_div)

    @staticmethod
    def squash_div(graph):
        def validate_node(nodes_tuple):
            return validate_eltwise_pattern(graph, nodes_tuple, "weights")

        sequence = [
            (op_adapter.ElementwiseDivOp.TRANSLATION_KEY, (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        squash_node_into_nn_node(graph, matched_node_list)

        def validate_conv_sequences(nodes_tuple):
            return validate_conv_eltwise_pattern(graph, nodes_tuple[0], op_adapter.ElementwiseDivOp.TRANSLATION_KEY)

        sequences = [
            [("convolution",
              ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")]))],
            [("depthwise_convolution",
              ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")]))],
            [("deconvolution",
              ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")]))]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence,
                                                        validator=validate_conv_sequences,
                                                        ignore_constants=True)
            for match in matched_node_list:
                squash_eltwise_into_conv(graph, match[0])


@register_layer_optimization
class OptimizeElementwiseEqualTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseEqualOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseFloorDivTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseFloorDivOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseGreaterTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseGreaterOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseGreaterEqualTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseGreaterEqualOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseLessTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseLessOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseLessEqualTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseLessEqualOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseNotEqualTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseNotEqualOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseMaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseMaxOp.TRANSLATION_KEY
        self.register_method(CHAIN_ELTWISE_OPS, self.chain_eltwise_ops)

    @staticmethod
    def chain_eltwise_ops(graph):
        def validate_node(nodes_tuple):
            return len(nodes_tuple[0].input_names) > 2

        sequence = [
            (op_adapter.ElementwiseMaxOp.TRANSLATION_KEY, (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        translation_utils.chain_matched_eltwise_ops(graph, matched_node_list, op_adapter.ElementwiseMaxOp)


@register_layer_optimization
class OptimizeElementwiseMinTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseMinOp.TRANSLATION_KEY
        self.register_method(CHAIN_ELTWISE_OPS, self.chain_eltwise_ops)

    @staticmethod
    def chain_eltwise_ops(graph):
        def validate_node(nodes_tuple):
            return len(nodes_tuple[0].input_names) > 2

        sequence = [
            (op_adapter.ElementwiseMinOp.TRANSLATION_KEY, (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        translation_utils.chain_matched_eltwise_ops(graph, matched_node_list, op_adapter.ElementwiseMinOp)


@register_layer_optimization
class OptimizeElementwisePowerTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwisePowerOp.TRANSLATION_KEY
        self.register_method(PREPARE_INPUTS_AS_PARAMS, self.prepare_inputs_as_params)

    def prepare_inputs_as_params(self, node, graph):
        exponent_buffer = graph.get_buffer(node.input_names[1])
        exponent_node = exponent_buffer.producer
        if exponent_node.op.type != op_adapter.ConstantOp.TRANSLATION_KEY:
            raise ValueError("Dynamic exponents on node {} are not supported in this backend.".format(node.op.name))
        node.op.power = exponent_node.op.tensor
        graph.remove_node_as_consumer(node, exponent_buffer.name)


@register_layer_optimization
class OptimizeElementwiseProductTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseProductOp.TRANSLATION_KEY
        self.register_method(SQUASH_PROD, self.squash_prod)
        self.register_method(CHAIN_ELTWISE_OPS, self.chain_eltwise_ops)

    @staticmethod
    def squash_prod(graph):
        def validate_node(nodes_tuple):
            return validate_eltwise_pattern(graph, nodes_tuple, "weights")

        sequence = [
            (op_adapter.ElementwiseProductOp.TRANSLATION_KEY, (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        squash_node_into_nn_node(graph, matched_node_list)

        def validate_conv_sequences(nodes_tuple):
            return validate_conv_eltwise_pattern(graph, nodes_tuple[0], op_adapter.ElementwiseProductOp.TRANSLATION_KEY)

        sequences = [
            [("convolution",
              ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]))],
            [("depthwise_convolution",
              ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]))],
            [("deconvolution",
              ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]))]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence,
                                                        validator=validate_conv_sequences,
                                                        ignore_constants=True)
            for match in matched_node_list:
                squash_eltwise_into_conv(graph, match[0])

    @staticmethod
    def chain_eltwise_ops(graph):
        def validate_node(nodes_tuple):
            return len(nodes_tuple[0].input_names) > 2

        sequence = [
            (op_adapter.ElementwiseProductOp.TRANSLATION_KEY, (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        translation_utils.chain_matched_eltwise_ops(graph, matched_node_list, op_adapter.ElementwiseProductOp)


@register_layer_optimization
class OptimizeElementwiseSelectTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseSelectOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseSubTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseSubOp.TRANSLATION_KEY
        self.register_method(SQUASH_SUB, self.squash_sub)

    @staticmethod
    def squash_sub(graph):
        def validate_node(nodes_tuple):
            return validate_eltwise_pattern(graph, nodes_tuple, "bias")

        sequence = [
            (op_adapter.ElementwiseSubOp.TRANSLATION_KEY, (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        squash_node_into_nn_node(graph, matched_node_list)

        def validate_conv_sequences(nodes_tuple):
            return validate_conv_eltwise_pattern(graph, nodes_tuple[0], op_adapter.ElementwiseSubOp.TRANSLATION_KEY)

        sequences = [
            [("convolution",
              ("MATCH_BUFS_AT_INDEX", [("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")]))],
            [("depthwise_convolution",
              ("MATCH_BUFS_AT_INDEX", [("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")]))],
            [("deconvolution",
              ("MATCH_BUFS_AT_INDEX", [("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")]))]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence,
                                                        validator=validate_conv_sequences,
                                                        ignore_constants=True)
            for match in matched_node_list:
                squash_eltwise_into_conv(graph, match[0])


@register_layer_optimization
class OptimizeElementwiseSumTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseSumOp.TRANSLATION_KEY
        self.register_method(SQUASH_SUM, self.squash_sum)
        self.register_method(CHAIN_ELTWISE_OPS, self.chain_eltwise_ops)

    @staticmethod
    def squash_sum(graph):
        def validate_node(nodes_tuple):
            return validate_eltwise_pattern(graph, nodes_tuple, "bias")

        sequence = [
            (op_adapter.ElementwiseSumOp.TRANSLATION_KEY, (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        squash_node_into_nn_node(graph, matched_node_list)

        def validate_conv_sequences(nodes_tuple):
            return validate_conv_eltwise_pattern(graph, nodes_tuple[0], op_adapter.ElementwiseSumOp.TRANSLATION_KEY)

        sequences = [
            [("convolution",
              ("MATCH_BUFS_AT_INDEX", [("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]))],
            [("depthwise_convolution",
              ("MATCH_BUFS_AT_INDEX", [("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]))],
            [("deconvolution",
              ("MATCH_BUFS_AT_INDEX", [("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]))]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence,
                                                        validator=validate_conv_sequences,
                                                        ignore_constants=True)
            for match in matched_node_list:
                squash_eltwise_into_conv(graph, match[0])


    @staticmethod
    def chain_eltwise_ops(graph):
        def validate_node(nodes_tuple):
            return len(nodes_tuple[0].input_names) > 2

        sequence = [
            (op_adapter.ElementwiseSumOp.TRANSLATION_KEY, (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        translation_utils.chain_matched_eltwise_ops(graph, matched_node_list, op_adapter.ElementwiseSumOp)


@register_layer_optimization
class OptimizeElementwiseOrTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseOrOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseUnaryAbsTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryAbsOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseUnaryCeilTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryCeilOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseUnaryExpTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryExpOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseUnaryFloorTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryFloorOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseUnaryLogTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryLogOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseUnaryNegTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryNegOp.TRANSLATION_KEY
        self.register_method(OPTIMIZE_NEG, self.optimize_negation)

    @staticmethod
    def optimize_negation(graph):
        def validate_neg(nodes_tuple):
            for input_name_ in nodes_tuple[0].input_names:
                node_ = graph.get_producer_node(input_name_)
                if node_.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and \
                        all(val == -1 for val in numpy.array(node_.op.tensor).flatten()):
                    return True

            return False

        # Optimization: -1 * A => Neg(A)
        sequences = [
            [
                ("elementwise_product",
                 ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
                 ())
            ]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence, validator=validate_neg)
            for node_tuple in matched_node_list:
                prod_node = node_tuple[0]
                non_const_input_node = None
                const_input_node = None
                for input_name in prod_node.input_names:
                    input_node = graph.get_producer_node(input_name)
                    if input_node.op.type != op_adapter.ConstantOp.TRANSLATION_KEY:
                        non_const_input_node = input_node
                    else:
                        const_input_node = input_node
                const_input_buf = graph.get_buffer(const_input_node.output_names[0])

                # remove const as input to prod, the prod node will then be replaced as Neg
                const_input_buf.consumers.remove(prod_node)
                prod_node.input_names.remove(const_input_node.output_names[0])
                if len(const_input_buf.consumers) == 0:
                    graph.prune(const_input_node)

                neg_op = op_adapter.ElementwiseUnaryNegOp(None)
                neg_op.name = graph.naming_policy.get_op_name(neg_op)
                graph.replace(prod_node.op, neg_op)
                log_debug2("Optimization of -1 * A => Neg(A) complete. Op {} replaced with NegOp"
                           .format(prod_node.op.name))

        # Optimization: A + Neg(B) => A - B
        #               Neg(A) + B => B - A
        #               Neg(A) + Neg(B) => Neg(A) - B
        sequences = [
            [
                ("elementwise_sum",
                 ("FLEXIBLE_NUM_BUFS", [("elementwise_unary_neg", "ANY")]),
                 ())
            ]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence)
            for node_tuple in matched_node_list:
                sum_node = node_tuple[0]
                neg_node_to_prune = None
                for input_name in sum_node.input_names:
                    input_node = graph.get_producer_node(input_name)
                    input_buf = graph.get_buffer(input_name)
                    if input_node.op.type == op_adapter.ElementwiseUnaryNegOp.TRANSLATION_KEY:
                        # if more than consumer of NegOp then we cant remove it hence optimization
                        # is not really relevant.
                        if len(input_buf.consumers) == 1:
                            neg_node_to_prune = input_node

                if neg_node_to_prune is not None:
                    # Update the input and consumer list and remove NegOp from graph
                    neg_idx = sum_node.input_names.index(neg_node_to_prune.output_names[0])
                    sum_input_names = sum_node.input_names[:]
                    neg_input_name = neg_node_to_prune.input_names[0]
                    neg_input_buf = graph.get_buffer(neg_input_name)
                    graph.prune(neg_node_to_prune, force_remove=True)
                    if neg_idx == 0:
                        # got Neg(A) + B, need B - A
                        sum_input_names[0] = sum_input_names[1]
                        sum_input_names[1] = neg_input_name
                    else:
                        # Neg(A) + Neg(B) or A + Neg(B)
                        sum_input_names[neg_idx] = neg_input_name
                    neg_input_buf.consumers.add(sum_node)
                    sum_node.input_names = sum_input_names

                    sub_op = op_adapter.ElementwiseSubOp(None)
                    sub_op.name = graph.naming_policy.get_op_name(sub_op)
                    graph.replace(sum_node.op, sub_op)
                    log_debug2("Optimization of addition to a negative of an op (e.g: A + Neg(B) => A - B) complete. "
                               "Op {} replaced with SubOp"
                               .format(sum_node.op.name))


@register_layer_optimization
class OptimizeElementwiseUnaryNotTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryNotOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseUnaryRoundTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryRoundOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseUnaryRsqrtTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryRsqrtOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseUnarySinTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnarySinOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseUnarySqrtTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnarySqrtOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeErfTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ErfOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeExpandTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ExpandOp.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    @staticmethod
    def remove_noop(node, graph):
        input_name = node.input_names[0]
        input_shape = graph.get_buffer(input_name).shape
        if input_shape == node.op.output_shape:
            graph.squash(node, input_name=input_name)


@register_layer_optimization
class OptimizeFullyConnectedTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.FullyConnectedOp.TRANSLATION_KEY
        self.register_method(SQUASH_BATCHNORM, self.squash_batchnorm)

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.log_axes_to_spatial_first_order(node, graph)
        input_buf = graph.get_input_buffers(node)[0]
        if input_buf.rank() == 4:
            AxisTracker.enforce_input_type(graph, input_buf.name, node.op.name, AxisTracker.AxisFormat.NSC,
                                             AxisTracker.AxisFormat.NCS_TO_NSC)

            # weights expect NCHW order, need to permute
            input_buf = graph.get_input_buffers(node)[0]
            batch, height, width, depth = input_buf.shape
            weights = node.op.weights

            # Assuming FC: W^Tx + b and weights have shape (input_size, output_size)
            input_size = weights.shape[0]
            output_size = weights.shape[1]
            log_assert(input_size == depth * height * width,
                       code_to_message.get_error_message("ERROR_FC_WRONG_INPUT_SIZE")(node.op.name,
                                                                                      (input_size, output_size),
                                                                                      (batch,  height, width, depth)))
            weights.shape = (depth, height, width, output_size)
            weights = numpy.transpose(weights, (3, 1, 2, 0))
            weights = numpy.ascontiguousarray(weights, dtype=numpy.float32)
            weights.shape = (output_size, input_size)
            node.op.weights = weights
        else:
            # again, need to transpose weights for spatial_first order
            weights = node.op.weights
            weights = numpy.ascontiguousarray(numpy.transpose(weights, (1, 0)))
            node.op.weights = weights

        output_buf = graph.get_output_buffers(node)[0]
        output_buf.axis_format = AxisTracker.AxisFormat.FEATURE

    @staticmethod
    def squash_batchnorm(graph):
        def validate(nodes_tuple):
            bn_node_ = next(iter(graph.get_output_buffers(nodes_tuple[0])[0].consumers))
            bn_input_buffer_ = graph.get_input_buffers(bn_node_)[0]
            if bn_node_.op.compute_statistics:
                log_debug("InstanceNorm layer {} cannot be squashed", bn_node_.op.name)
                return False
            return True

        sequence = [
            ("fully_connected",
                (),
                ("MATCH_NUM_BUFS", [("batchnorm", "ALL")])
             )
        ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate)

        for node_tuple in matched_node_list:
            # sanity check
            log_assert(len(node_tuple) == len(sequence),
                       "ERROR: Pattern matching for squash batchnorm returned extra nodes. Got {} nodes, Expected {}.",
                       len(node_tuple), len(sequence))

            fc_node = node_tuple[0]
            bn_node = next(iter(graph.get_output_buffers(fc_node)[0].consumers))
            bn_input_buffer = graph.get_input_buffers(bn_node)[0]
            weights = fc_node.op.weights
            broadcasted_tensor = numpy.zeros(len(bn_node.op.weights), dtype=numpy.float32)
            if fc_node.op.transpose_b == False:
                weight_tensor = numpy.transpose(weights, (1, 0)).copy()
            else:
                weight_tensor = weights.copy()
            broadcasted_tensor = broadcasted_tensor + weight_tensor
            broadcasted_tensor = broadcasted_tensor * bn_node.op.weights
            if fc_node.op.transpose_b == False:
                broadcasted_transpose = numpy.transpose(broadcasted_tensor, (1, 0)).copy()
            else:
                broadcasted_transpose = broadcasted_tensor.copy()
            fc_node.op.weights = broadcasted_transpose
            fc_node.op.bias = fc_node.op.bias * bn_node.op.weights + bn_node.op.bias
            graph.squash(bn_node, input_name=bn_input_buffer.name)
            log_debug2(code_to_message.get_debugging_message("DEBUG_BATCHNORM_SQUASH")(bn_node.op.name,
                                                                                       fc_node.op.type,
                                                                                       fc_node.op.name))


@register_layer_optimization
class OptimizeGatherTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GatherOp.TRANSLATION_KEY
        self.register_method(INJECT_CAST_FOR_GATHER, self.inject_cast_for_gather)
        self.register_method(REMOVE_NOOP, self.remove_noop)
        self.register_method(HANDLE_GATHER_NEGATIVE_INDICES, self.handle_gather_negative_indices)

    def axes_to_spatial_first_order(self, node, graph):
        # Remap the axis if < 0 to the real axis and if needed permute it for NSC
        # In addition, output buffer axis tracking stays the same as input so long
        # as the rank of indices == 1. Otherwise it's non trivial as the rank will change
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        indices_buf = graph.get_input_buffers(node)[1]
        output_buf = graph.get_output_buffers(node)[0]
        if node.op.axis < 0:
            node.op.axis = node.op.axis+input_buf.rank()
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            if indices_buf.rank() > 1:
                graph.inject_implicit_permute(input_name, node.op.name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                axis_map = graph.src_axis_order.permute_sequence[input_buf.rank() - 1]
                node.op.axis = axis_map[node.op.axis]
                output_buf.axis_format = AxisTracker.AxisFormat.NSC
                output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
        else:
            if indices_buf.rank() > 1:
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                output_buf.axis_format = input_buf.axis_format

    def handle_gather_negative_indices(self, node, graph):
        indices_name = node.input_names[1]
        if isinstance(graph.get_producer_op(indices_name), op_adapter.ConstantOp):
            const_op = graph.get_producer_op(indices_name)
            input_data_shape = graph.get_buffer(node.input_names[0]).shape
            with numpy.nditer(const_op.tensor, op_flags=['readwrite']) as it:
                for index in it:
                    if index < 0:
                        index += input_data_shape[node.op.axis]

    # TODO Remove this optimization once casts are properly optimized out in IR
    def inject_cast_for_gather(self, node, graph):
        cast_node_name = node.input_names[1] + "_cast"
        cast_op = op_adapter.CastOp(name=cast_node_name, to_type="int32")
        # check and reuse existing CastOp if already added
        if graph.has_buffer(cast_node_name):
            cast_buffer = graph.buffers[cast_node_name]
            cast_buffer.consumers.add(node)
            input_buffer = graph.buffers[node.input_names[1]]
            input_buffer.consumers.remove(node)
            node.input_names[1] = cast_node_name
        else:
            log_debug("Injecting cast op {} for node {}'s indices input.".format(cast_node_name, node.op.name))
            graph.inject(cast_op, input_name=node.input_names[1], output_name=cast_node_name, consumer_names=[node.op.name])

    @staticmethod
    def remove_noop(node, graph):
        input_buffer = graph.get_input_buffers(node)[0]
        indices_buffer = graph.get_input_buffers(node)[1]
        output_buffer_shape = graph.get_output_buffers(node)[0].shape
        if input_buffer.shape == output_buffer_shape and len(input_buffer.consumers) == 1:
            # this gather has no effect, remove indices first
            if (indices_buffer.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and
                len(indices_buffer.consumers) == 1):
                indices_node = indices_buffer.producer
                graph.prune(indices_node, force_remove=True)
            # then remove gather
            ret = graph.squash(node, input_name=input_buffer.name)
            if ret:
                log_debug("Squash Gather op {} due to Noop. "
                          "Input shape {}".format(node.op.name,
                                                  input_buffer.shape))


@register_layer_optimization
class OptimizeGeluTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GeluOp.TRANSLATION_KEY
        self.register_method(MATCH_GELU, self.match_gelu)

    @staticmethod
    def match_gelu(graph):
        sequence = [
            ("elementwise_div",
             ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("erf", "ALL")])
             ),
            ("erf",
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("erf", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_sum", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY")]),
             ()
             )
        ]
        matched_node_list = graph.get_matched_nodes(sequence, ignore_constants=True)
        for node_tuple in matched_node_list:
            div_node = node_tuple[0]
            # Squash all nodes except the first div in reverse order and the div op will be replaced
            for node in node_tuple[:0:-1]:
                input_names = node.input_names[:]
                # pick squashable input based on whether current node is only consumer and input is not network input
                input_name = [name for name in input_names if (len(graph.get_buffer(name).consumers) == 1 and
                              not isinstance(graph.get_producer_op(name), op_adapter.InputOp))][0]
                input_names.remove(input_name)
                for input_name_ in input_names:
                    # disconnect rest of inputs from node
                    input_buf_ = graph.get_buffer(input_name_)
                    input_buf_.consumers.remove(node)
                    node.input_names.remove(input_name_)
                graph.squash(node, input_name=input_name)

            # disconnect the constant div input
            const_input_buf = [graph.get_buffer(name) for name in div_node.input_names if
                               graph.get_producer_op(name).type == op_adapter.ConstantOp.TRANSLATION_KEY][0]
            const_input_buf.consumers.remove(div_node)
            div_node.input_names.remove(const_input_buf.name)

            # replace the div op with gelu
            div_op = div_node.op
            div_op_name = graph.naming_policy.get_op_name(div_op)
            gelu_op_name = div_op_name + '_gelu'
            gelu_op = op_adapter.GeluOp(gelu_op_name)
            graph.replace(div_op, gelu_op)


@register_layer_optimization
class OptimizeGenerateProposalsOp(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GenerateProposalsOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeGruTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GruOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.eltwise_to_spatial_first_order(node, graph)


@register_layer_optimization
class OptimizeIdentityTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.IdentityOp.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        output_buf = graph.get_output_buffers(node)[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf.shape = input_buf.shape
        output_buf.axis_format = input_buf.axis_format

    @staticmethod
    def remove_noop(node, graph):
        graph.squash_noop(node)


@register_layer_optimization
class OptimizeL2NormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.L2NormOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeL2NormTranslation, self).axes_to_spatial_first_order(node, graph)

        # transform axis to the correct index, also ensures axis is always positive
        input_buf = graph.get_input_buffers(node)[0]
        axis_map = graph.src_axis_order.permute_sequence[input_buf.rank() - 1]
        if type(node.op.axis) is numpy.ndarray:
            for i in range(len(node.op.axis)):
                node.op.axis[i] = axis_map[node.op.axis[i]]
        else:
            node.op.axis = axis_map[node.op.axis]


@register_layer_optimization
class OptimizeL2PoolTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.L2PoolOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeLayerNormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.LayerNormOp.TRANSLATION_KEY
        self.register_method(MATCH_LAYERNORM, self.match_layer_norm)

    @staticmethod
    def match_layer_norm(graph):
        sequence1 = [
            ("reduce_mean",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")])
             ),
            ("elementwise_sub",
             ("FLEXIBLE_NUM_BUFS", [("reduce_mean", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_power", "ANY"), ("elementwise_div", "ANY")])
             ),
            ("elementwise_power",
             ("MATCH_NUM_BUFS", [("elementwise_sub", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_power", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ANY"), ("elementwise_sub", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_div", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY")]),
             ()
             )
        ]

        sequence2 = [
            ("reshape",
             (),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("reshape", "ALL")]),
             ("MATCH_NUM_BUFS", [("reshape", "ALL")])
             ),
            ("reshape",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")])
             ),
            ("elementwise_sub",
             ("FLEXIBLE_NUM_BUFS", [("reshape", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_power", "ANY"), ("elementwise_div", "ANY")])
             ),
            ("elementwise_power",
             ("MATCH_NUM_BUFS", [("elementwise_sub", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("reshape", "ALL")])
             ),
            ("reshape",
             ("MATCH_NUM_BUFS", [("elementwise_power", "ALL")]),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("reshape", "ALL")]),
             ("MATCH_NUM_BUFS", [("reshape", "ALL")]),
             ),
            ("reshape",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reshape", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ANY"), ("elementwise_sub", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_div", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY")]),
             ()
             )
        ]

        sequences = [sequence1, sequence2]
        for idx, sequence in enumerate(sequences):
            matched_node_list = graph.get_matched_nodes(sequence, ignore_constants=True)

            for node_tuple in matched_node_list:
                if idx == 0:
                    reduce_mean_node = node_tuple[0]
                    reduce_mean_name = graph.naming_policy.get_op_name(reduce_mean_node.op)
                    prunable_nodes = node_tuple[::-1]
                else:
                    reshape_node = node_tuple[0]
                    reshape_node_input_bufs = graph.get_input_buffers(reshape_node)
                    reduce_mean_node = node_tuple[1]
                    reduce_mean_name = graph.naming_policy.get_op_name(reduce_mean_node.op)
                    # return nodes in reverse order except the first reshape node
                    prunable_nodes = node_tuple[:0:-1]

                beta_input_name = node_tuple[-1].input_names[1]
                gamma_input_name = node_tuple[-2].input_names[1]
                axes = [0]
                epsilon = op_adapter.LayerNormOp.EPSILON

                last_node = node_tuple[-1]
                last_node_buf = graph.get_output_buffers(last_node)
                last_node_consumers = last_node_buf[0].consumers
                last_node_consumers_names = [node.op.name for node in last_node_consumers]
                # maps consumers of last node buffer with corresponding input_names
                last_node_consumers_input_names = {}
                for consumer in last_node_consumers:
                    last_node_consumers_input_names[consumer] = copy.deepcopy(consumer.input_names)

                # Prune all matched nodes in reverse order
                for node in prunable_nodes:
                    input_names = node.input_names[:]
                    # determine axes parameter of LayerNorm from ReduceMean Op
                    if isinstance(node.op, op_adapter.ReduceMeanOp):
                        axes = node.op.axes
                    # determine epsilon parameter of LayerNorm from the ElementwiseSumOp Op with constant input of
                    # size 1
                    if isinstance(node.op, op_adapter.ElementwiseSumOp):
                        input_name = [name for name in input_names if
                                      (isinstance(graph.get_producer_op(name), op_adapter.ConstantOp))][0]
                        if graph.get_producer_op(input_name).tensor.size == 1:
                            epsilon = graph.get_producer_op(input_name).tensor[0]

                    graph.prune(node, force_remove=True)

                # reassign input_names to consumers of last node buffer post pruning of the nodes
                for consumer in last_node_consumers:
                    consumer.input_names = last_node_consumers_input_names[consumer]

                layer_norm_op_name = reduce_mean_name + '_LayerNorm'
                layer_norm_input_names = reduce_mean_node.input_names + [gamma_input_name, beta_input_name]
                if idx == 0:
                    layer_norm_output_names = last_node.output_names
                else:
                    layer_norm_output_names = [layer_norm_op_name]

                layer_norm_op = op_adapter.LayerNormOp(layer_norm_op_name, axes=axes, epsilon=epsilon)

                # compute the correct idx to insert layer_norm
                idx_to_insert = 0
                for input_name in layer_norm_input_names:
                    buf = graph.get_buffer(input_name)
                    cur_idx = graph.nodes_in_order.index(buf.producer)
                    if idx_to_insert < cur_idx:
                        idx_to_insert = cur_idx

                layer_node = graph.add(layer_norm_op, input_names=layer_norm_input_names, output_names=layer_norm_output_names,
                                       idx=idx_to_insert+1)

                # add consumers of layer_norm output buffer
                for output_name in layer_norm_output_names:
                    output_buf_ = graph.get_buffer(output_name)
                    output_buf_.consumers = last_node_consumers

                if idx == 1:
                    # add reshape node after layer_norm
                    reshape_name = layer_norm_op_name + "_postprocess_reshape"
                    reshape_op = op_adapter.ReshapeOp(reshape_name, output_shape=reshape_node_input_bufs[0].shape)
                    reshape_node = graph.inject(reshape_op, input_name=layer_node.output_names[0], output_name=last_node.output_names[0],
                                                consumer_names=last_node_consumers_names if last_node_consumers_names else None)


@register_layer_optimization
class OptimizeLstmTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.LstmOp.TRANSLATION_KEY
        self.register_method(UNROLL_LSTM_TIME_STEPS, self.unroll_lstm_time_steps)
        self.register_method(PREPROCESS_LSTM_OPS, self.preprocess_lstm_ops)

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeLstmTranslation, self).axes_to_spatial_first_order(node, graph)

        # At this point, weights are expected in NxK, SNPE requires KxN
        node.op.input_weights = numpy.ascontiguousarray(node.op.input_weights.transpose(), dtype=numpy.float32)
        node.op.hidden_state_weights = numpy.ascontiguousarray(node.op.hidden_state_weights.transpose())

        # LSTM input axis format must be BTF
        input_name = node.input_names[0]
        input_bufs = graph.get_input_buffers(node)
        output_bufs = graph.get_output_buffers(node)
        if input_bufs[0].axis_format == AxisTracker.AxisFormat.NONTRIVIAL \
                or input_bufs[0].axis_format == AxisTracker.AxisFormat.TBF:
            graph.inject_implicit_permute(input_name, node.op.name, AxisTracker.AxisFormat.BTF,
                                          AxisTracker.AxisFormat.TBF_TO_BTF, [node.op.name])

        # Set up LSTM outputs' axis formats
        # First output: BTF
        # Other outputs: NONTRIVIAL
        for i, output_buf in enumerate(output_bufs):
            if i == 0:
                output_buf.axis_format = AxisTracker.AxisFormat.BTF
            else:
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

    def unroll_lstm_time_steps(self, graph):
        sequence = [
            (op_adapter.LstmOp.TRANSLATION_KEY, (), ())
        ]

        def validate_node(nodes_tuple):
            batch_size, seq_length, input_size = graph.get_buffer(nodes_tuple[0].input_names[0]).shape[:]
            return seq_length > 1

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)

        for nodes_tuple in matched_node_list:
            lstm_node = nodes_tuple[0]
            lstm_node_name = lstm_node.op.name

            DATA_IDX, CELL_OUT_IDX, HIDDEN_OUT_IDX = 0, 1, 2
            lstm_node_input_name = lstm_node.input_names[DATA_IDX]
            lstm_node_output_name = lstm_node.output_names[DATA_IDX]
            lstm_node_idx = graph.nodes_in_order.index(lstm_node)

            log_debug("Unrolling LSTM node {}".format(lstm_node_name))

            # Extract and validate inputs, outputs, and sizes
            number_of_outputs = len(lstm_node.output_names)
            all_output_buffer = graph.get_buffer(lstm_node_output_name)
            batch_size, seq_length, input_size = graph.get_buffer(lstm_node_input_name).shape[:]

            if number_of_outputs == 1:
                # Add dummy buffers for missing outputs
                output_size = graph.get_buffer(lstm_node_output_name).shape[-1]
                num_units = lstm_node.op.hidden_size
                hidden_output_dummy_name = lstm_node_name + "_hidden_output_dummy"
                graph.add_output_buffer(lstm_node, hidden_output_dummy_name,
                                        [batch_size, output_size], AxisTracker.AxisFormat.NONTRIVIAL)

                cell_output_dummy_name = lstm_node_name + "_cell_output_dummy"
                graph.add_output_buffer(lstm_node, cell_output_dummy_name,
                                        [batch_size, num_units], AxisTracker.AxisFormat.NONTRIVIAL)
                lstm_node.output_names = [all_output_buffer.name, cell_output_dummy_name, hidden_output_dummy_name]

            hidden_output_buffer = graph.get_buffer(lstm_node.output_names[HIDDEN_OUT_IDX])
            cell_output_buffer = graph.get_buffer(lstm_node.output_names[CELL_OUT_IDX])

            input_x_split_name_list = []
            for i in range(seq_length):
                input_x_i_name = lstm_node_name + "_" + lstm_node_input_name + str(i)
                input_x_split_name_list.append(input_x_i_name)
            input_x_split_name = lstm_node_name + "_" + lstm_node_input_name + "_split"
            time_step_axis = 1
            input_x_split_op = op_adapter.SliceOp(name=input_x_split_name, axis=time_step_axis)

            # Split input to T inputs
            graph.add(input_x_split_op, input_names=[lstm_node.input_names[0]],
                                        output_names=input_x_split_name_list, idx=lstm_node_idx)

            output_y_concat_name_list = []
            output_h_name_list = []
            output_c_name_list = []
            for i in range(seq_length):
                output_y_i_name = lstm_node_output_name + str(i)
                output_y_concat_name_list.append(output_y_i_name)
                output_h_i_name = lstm_node.output_names[HIDDEN_OUT_IDX] + str(i)
                output_h_name_list.append(output_h_i_name)
                output_c_i_name = lstm_node.output_names[CELL_OUT_IDX] + str(i)
                output_c_name_list.append(output_c_i_name)

            for i in range(seq_length):
                curr_idx = i
                if lstm_node.op.backward:
                    curr_idx = seq_length-1-i

                if i == 0:
                    _reset_state_at_time_step_0 = lstm_node.op.reset_state_at_time_step_0
                    _h_0_input_name = lstm_node.op.h_0_input_name
                    _c_0_input_name = lstm_node.op.c_0_input_name
                else:
                    _reset_state_at_time_step_0 = False
                    _h_0_input_name = output_h_name_list[i-1]
                    _c_0_input_name = output_c_name_list[i-1]

                lstm_time_step_i_op_name = lstm_node_name + str(i)
                lstm_time_step_i_op = op_adapter.LstmOp(name=lstm_time_step_i_op_name,
                                                        input_weights=lstm_node.op.input_weights,
                                                        gate_bias=lstm_node.op.gate_bias,
                                                        hidden_state_weights=lstm_node.op.hidden_state_weights,
                                                        hidden_size=lstm_node.op.hidden_size,
                                                        cell_weights=lstm_node.op.cell_weights,
                                                        normalization_weights=lstm_node.op.normalization_weights,
                                                        w_xc_static=lstm_node.op.w_xc_static,
                                                        backward=lstm_node.op.backward,
                                                        reset_state_at_time_step_0=_reset_state_at_time_step_0,
                                                        h_0_input_name=_h_0_input_name,
                                                        c_0_input_name=_c_0_input_name,
                                                        sequence_continuation_name=lstm_node.op.sequence_continuation_name,
                                                        x_static_name=lstm_node.op.x_static_name,
                                                        cell_clip_threshold=lstm_node.op.cell_clip_threshold,
                                                        proj_weights=lstm_node.op.proj_weights,
                                                        proj_bias=lstm_node.op.proj_bias,
                                                        output_clip_threshold=lstm_node.op.output_clip_threshold)

                lstm_op_input_name_list = [input_x_split_name_list[curr_idx], _h_0_input_name, _c_0_input_name]
                lstm_op_output_name_list = [output_y_concat_name_list[i], output_c_name_list[i], output_h_name_list[i]]
                graph.add(lstm_time_step_i_op, input_names=lstm_op_input_name_list, output_names=lstm_op_output_name_list, idx=lstm_node_idx+i+1)

            output_y_concat_name = lstm_node_output_name + "_concat"
            output_y_concat_op_name = lstm_node_name + "_" + lstm_node_output_name + "_concat"
            output_y_concat_op = op_adapter.ConcatOp(name=output_y_concat_op_name, axis=1)

            # Concat output from T outputs
            if lstm_node.op.backward:
                output_y_concat_name_list.reverse()
            graph.add(output_y_concat_op, input_names=output_y_concat_name_list, output_names=output_y_concat_name, idx=lstm_node_idx+seq_length+1)

            for consumer in list(all_output_buffer.consumers):
                output_y_concat_buffer = graph.get_buffer(output_y_concat_name)
                output_y_concat_buffer.consumers.add(consumer)
                consumer.input_names.append(output_y_concat_name)
            for consumer in list(hidden_output_buffer.consumers):
                output_h_buffer = graph.get_buffer(output_h_name_list[seq_length-1])
                output_h_buffer.consumers.add(consumer)
                consumer.input_names.append(output_h_name_list[seq_length-1])
            for consumer in list(cell_output_buffer.consumers):
                ourput_c_buffer = graph.get_buffer(output_c_name_list[seq_length-1])
                ourput_c_buffer.consumers.add(consumer)
                consumer.input_names.append(output_c_name_list[seq_length-1])

            # prune original lstm_node
            graph.prune(lstm_node, force_remove=True)

    # TODO Move to QNN-specific graph transformations once work on GraphTransformer is complete
    # Preprocesses LSTM inputs, outputs, and attributes for QNN consumption
    @staticmethod
    def preprocess_lstm_ops(graph):

        sequence = [
            (op_adapter.LstmOp.TRANSLATION_KEY, (), ())
        ]

        matched_node_list = graph.get_matched_nodes(sequence)

        for nodes_tuple in matched_node_list:
            lstm_node = nodes_tuple[0]
            lstm_node_name = lstm_node.op.name
            lstm_node_idx = graph.nodes_in_order.index(lstm_node)
            number_of_outputs = len(lstm_node.output_names)

            # This variable determines if a reshape needs to be added to the input
            input_reshape_needed = True

            log_debug("Preprocessing LSTM node {} for QNN lowering.".format(lstm_node_name))

            # Extract and validate inputs, outputs, and sizes
            input_shape = graph.get_buffer(lstm_node.input_names[0]).shape
            output_size = graph.get_buffer(lstm_node.output_names[0]).shape[-1]
            num_units = lstm_node.op.hidden_size

            if len(input_shape) == 3:
                batch_size, seq_length, input_size = graph.get_buffer(lstm_node.input_names[0]).shape[:]
                # Check that extracted sequence length is 1
                if seq_length != 1:
                    raise ValueError('Unsupported sequence length for LSTM node {}, expected 1, got {}.'.format(
                        lstm_node_name, seq_length))

                # Since sequence length is one, we need to squeeze the dimension to 2D
                # We can do this by removing a reshape which may have been added by the frontend
                # Or by adding a reshape ourselves to squeeze to 2D.
                candidate_reshape_node = graph.get_producer_node(lstm_node.input_names[0])
                if candidate_reshape_node.op.type == op_adapter.ReshapeOp.TRANSLATION_KEY:
                    input_buffers = graph.get_input_buffers(candidate_reshape_node)
                    output_buffers = graph.get_output_buffers(candidate_reshape_node)
                    # check if no other consumers and input shape for this buffer is 2D i.e this is an unsqueeze
                    # remove reshape to revert back to 2D, input_reshape is not needed
                    # otherwise reshape is needed
                    if len(output_buffers[0].consumers) == 1 and input_buffers[0].shape == [batch_size, input_size]:
                        if graph.squash(candidate_reshape_node, input_buffers[0].name):
                            # update lstm node input buffer shape to pre-reshape input shape
                            lstm_node_input_buffer = graph.get_buffer(lstm_node.input_names[0])
                            lstm_node_input_buffer.shape = input_buffers[0].shape

                            # set no reshape needed
                            input_reshape_needed = False

            elif len(input_shape) == 2:
                input_reshape_needed = False
                batch_size, input_size = graph.get_buffer(lstm_node.input_names[0]).shape[:]
                seq_length = 1
            else:
                raise ValueError('Unsupported input rank for LSTM node {}, expected 3 or 2, got {}.'.format(
                     lstm_node_name, len(input_shape)))

            # Requires initial_h and initial_c inputs to be present
            # The following code adds zero valued tensors provided the conditions below are satisfied
            if len(lstm_node.input_names) != 3:
                if lstm_node.op.c_0_input_name or lstm_node.op.h_0_input_name or len(lstm_node.input_names) != 1:
                    raise ValueError('Unsupported number of inputs for LSTM node {}, expected 3 or 1 if no initial states, got {}.'.format(
                        lstm_node_name, len(lstm_node.input_names)))

                # add zeros for initial h and c inputs since there are needed for QNN
                initial_hidden_state_name = lstm_node_name + '_initial_hidden_state'
                initial_hidden_state_tensor = numpy.zeros((batch_size, output_size), dtype=numpy.float32)
                initial_hidden_state_op = op_adapter.ConstantOp(name=initial_hidden_state_name, tensor=initial_hidden_state_tensor)
                graph.add(initial_hidden_state_op, input_names=[], output_names=[initial_hidden_state_name], idx=lstm_node_idx-1)
                lstm_node.op.h_0_input_name = initial_hidden_state_name

                initial_cell_state_name = lstm_node_name + '_initial_cell_state'
                initial_cell_state_tensor = numpy.zeros((batch_size, output_size), dtype=numpy.float32)
                initial_cell_state_op = op_adapter.ConstantOp(name=initial_cell_state_name, tensor=initial_cell_state_tensor)
                graph.add(initial_cell_state_op, input_names=[], output_names=[initial_cell_state_name], idx=lstm_node_idx-1)
                lstm_node.op.c_0_input_name = initial_cell_state_name

                lstm_node.input_names.extend([initial_hidden_state_name, initial_cell_state_name])

            # Only 1 or 3 outputs are supported for this optimization
            if number_of_outputs != 1 and number_of_outputs != 3:
                raise ValueError("Unsupported number of outputs for LSTM node {}, expected 1 or 3, got {}.".format(
                    lstm_node_name, number_of_outputs))

            initial_h_shape = graph.get_buffer(lstm_node.input_names[1]).shape

            # If the initial hidden state shape (and implicitly initial cell state shape)
            # is not 2D then it should be reshaped
            initial_state_reshape_needed = len(initial_h_shape) != 2 or initial_h_shape != [batch_size, output_size]

            # Input weights are expected in [input_size, 4*hidden_size]
            # Rec weights are expected in   [hidden_size,  4*hidden_size]
            # Biases are expected in [4*hidden_size]
            # All are expected in IFOG format
            # Split weights into 4 sections so that they can be indexed by gate
            lstm_input_split_weights = numpy.split(lstm_node.op.input_weights, indices_or_sections=4, axis=1)
            lstm_hidden_split_weights = numpy.split(lstm_node.op.hidden_state_weights, indices_or_sections=4, axis=1)
            gate_split_biases = numpy.split(lstm_node.op.gate_bias, indices_or_sections=4, axis=0)

            # Adding reshape nodes to squeeze sequence length dimensions from input if input is 3D
            if input_reshape_needed:
                input_x_reshape_node_name = lstm_node_name + "_" + lstm_node.input_names[0] + "_reshape"
                input_x_reshape_output_shape = [batch_size, input_size]
                input_x_reshape_op = op_adapter.ReshapeOp(name=input_x_reshape_node_name,
                                                          output_shape=input_x_reshape_output_shape)
                graph.inject(input_x_reshape_op, input_name=lstm_node.input_names[0],
                             output_name=input_x_reshape_node_name, consumer_names=[lstm_node_name])

            if initial_state_reshape_needed:
                input_h_reshape_node_name = lstm_node_name + "_" + lstm_node.input_names[1] + "_reshape"
                input_h_reshape_output_shape = [batch_size, output_size]
                input_h_reshape_op = op_adapter.ReshapeOp(name=input_h_reshape_node_name,
                                                          output_shape=input_h_reshape_output_shape)
                graph.inject(input_h_reshape_op, input_name=lstm_node.input_names[1],
                             output_name=input_h_reshape_node_name, consumer_names=[lstm_node_name])

                input_c_reshape_node_name = lstm_node_name + "_" + lstm_node.input_names[2] + "_reshape"
                input_c_reshape_output_shape = [batch_size, num_units]
                input_c_reshape_op = op_adapter.ReshapeOp(name=input_c_reshape_node_name,
                                                          output_shape=input_c_reshape_output_shape)
                graph.inject(input_c_reshape_op, input_name=lstm_node.input_names[2],
                             output_name=input_c_reshape_node_name, consumer_names=[lstm_node_name])

            # Must add all inputs derived from IR attributes as ConstantOp inputs in QNN
            # Weights may already be 2D (from TF as an example), but it is cleaner to resize anyway rather than check shape
            # for each input
            input_to_forget_w_name = lstm_node_name + '_input_to_forget_w'
            input_to_forget_w_tensor = numpy.resize(lstm_input_split_weights[1], (num_units, input_size))
            input_to_forget_w_op = op_adapter.ConstantOp(name=input_to_forget_w_name, tensor=input_to_forget_w_tensor)
            graph.add(input_to_forget_w_op, input_names=[], output_names=[input_to_forget_w_name], idx=lstm_node_idx-1)

            input_to_cell_w_name = lstm_node_name + '_input_to_cell_w'
            input_to_cell_w_tensor = numpy.resize(lstm_input_split_weights[3], (num_units, input_size))
            input_to_cell_w_op = op_adapter.ConstantOp(name=input_to_cell_w_name, tensor=input_to_cell_w_tensor)
            graph.add(input_to_cell_w_op, input_names=[], output_names=[input_to_cell_w_name], idx=lstm_node_idx-1)

            input_to_output_w_name = lstm_node_name + '_input_to_output_w'
            input_to_output_w_tensor = numpy.resize(lstm_input_split_weights[2], (num_units, input_size))
            input_to_output_w_op = op_adapter.ConstantOp(name=input_to_output_w_name, tensor=input_to_output_w_tensor)
            graph.add(input_to_output_w_op, input_names=[], output_names=[input_to_output_w_name], idx=lstm_node_idx-1)

            recurrent_to_forget_w_name = lstm_node_name + '_recurrent_to_forget_w'
            recurrent_to_forget_w_op = op_adapter.ConstantOp(name=recurrent_to_forget_w_name,
                                                             tensor=lstm_hidden_split_weights[1])
            graph.add(recurrent_to_forget_w_op, input_names=[], output_names=[recurrent_to_forget_w_name],
                                            idx=lstm_node_idx-1)

            recurrent_to_cell_w_name = lstm_node_name + '_recurrent_to_cell_w'
            recurrent_to_cell_w_op = op_adapter.ConstantOp(name=recurrent_to_cell_w_name,
                                                           tensor=lstm_hidden_split_weights[3])
            graph.add(recurrent_to_cell_w_op, input_names=[], output_names=[recurrent_to_cell_w_name],
                      idx=lstm_node_idx-1)

            recurrent_to_output_w_name = lstm_node_name + '_recurrent_to_output_w'
            recurrent_to_output_w_op = op_adapter.ConstantOp(name=recurrent_to_output_w_name,
                                                             tensor=lstm_hidden_split_weights[2])
            graph.add(recurrent_to_output_w_op, input_names=[], output_names=[recurrent_to_output_w_name],
                      idx=lstm_node_idx-1)

            forget_gate_b_name = lstm_node_name + '_forget_gate_b'
            forget_gate_b_tensor = numpy.resize(gate_split_biases[1], (num_units,))
            forget_gate_b_op = op_adapter.ConstantOp(name=forget_gate_b_name, tensor=forget_gate_b_tensor)
            graph.add(forget_gate_b_op, input_names=[], output_names=[forget_gate_b_name], idx=lstm_node_idx-1)

            cell_gate_b_name = lstm_node_name + '_cell_gate_b'
            cell_gate_b_tensor = numpy.resize(gate_split_biases[3], (num_units,))
            cell_gate_b_op = op_adapter.ConstantOp(name=cell_gate_b_name, tensor=cell_gate_b_tensor)
            graph.add(cell_gate_b_op, input_names=[], output_names=[cell_gate_b_name], idx=lstm_node_idx-1)

            output_gate_b_name = lstm_node_name + '_output_gate_b'
            output_gate_b_tensor = numpy.resize(gate_split_biases[2], (num_units,))
            output_gate_b_op = op_adapter.ConstantOp(name=output_gate_b_name, tensor=output_gate_b_tensor)
            graph.add(output_gate_b_op, input_names=[], output_names=[output_gate_b_name], idx=lstm_node_idx-1)

            # Next four inputs not captured by any FE - pass default of ones to ConstantOp
            input_norm_w_name, forget_norm_w_name, cell_norm_w_name, output_norm_w_name = '', '', '', ''
            if lstm_node.op.normalization_weights is not None:
                input_norm_w_name = lstm_node_name + '_input_norm_w'
                input_norm_w_op = op_adapter.ConstantOp(name=input_norm_w_name,
                                                        tensor=lstm_node.op.normalization_weights[0])
                graph.add(input_norm_w_op, input_names=[], output_names=[input_norm_w_name], idx=lstm_node_idx-1)

                forget_norm_w_name = lstm_node_name + '_forget_norm_w'
                forget_norm_w_op = op_adapter.ConstantOp(name=forget_norm_w_name,
                                                         tensor=lstm_node.op.normalization_weights[1])
                graph.add(forget_norm_w_op, input_names=[], output_names=[forget_norm_w_name], idx=lstm_node_idx-1)

                cell_norm_w_name = lstm_node_name + '_cell_norm_w'
                cell_norm_w_op = op_adapter.ConstantOp(name=cell_norm_w_name,
                                                       tensor=lstm_node.op.normalization_weights[3])
                graph.add(cell_norm_w_op, input_names=[], output_names=[cell_norm_w_name], idx=lstm_node_idx-1)

                output_norm_w_name = lstm_node_name + '_output_norm_w'
                output_norm_w_op = op_adapter.ConstantOp(name=output_norm_w_name,
                                                         tensor=lstm_node.op.normalization_weights[2])
                graph.add(output_norm_w_op, input_names=[], output_names=[output_norm_w_name], idx=lstm_node_idx-1)

            input_to_input_w_name = lstm_node_name + '_input_to_input_w'
            input_to_input_w_tensor = numpy.resize(lstm_input_split_weights[0], (num_units, input_size))
            input_to_input_op = op_adapter.ConstantOp(name=input_to_input_w_name, tensor=input_to_input_w_tensor)
            graph.add(input_to_input_op, input_names=[], output_names=[input_to_input_w_name], idx=lstm_node_idx-1)

            recurrent_to_input_w_name = lstm_node_name + '_recurrent_to_input_w'
            recurrent_to_input_op = op_adapter.ConstantOp(name=recurrent_to_input_w_name,
                                                          tensor=lstm_hidden_split_weights[0])
            graph.add(recurrent_to_input_op, input_names=[], output_names=[recurrent_to_input_w_name],
                      idx=lstm_node_idx-1)

            cell_to_input_w_name, cell_to_forget_w_name, cell_to_output_w_name = '', '', ''
            if lstm_node.op.cell_weights is not None:
                lstm_cell_split_weights = numpy.split(lstm_node.op.cell_weights, indices_or_sections=3, axis=1)
                cell_to_input_w_name = lstm_node_name + '_cell_to_input_w'
                cell_to_input_w_op = op_adapter.ConstantOp(name=cell_to_input_w_name,
                                                           tensor= lstm_cell_split_weights[0])
                graph.add(cell_to_input_w_op, input_names=[], output_names=[cell_to_input_w_name], idx=lstm_node_idx-1)

                cell_to_forget_w_name = lstm_node_name + '_cell_to_forget_w'
                cell_to_forget_w_op = op_adapter.ConstantOp(name=cell_to_forget_w_name,
                                                            tensor= lstm_cell_split_weights[1])
                graph.add(cell_to_forget_w_op, input_names=[], output_names=[cell_to_forget_w_name],
                          idx=lstm_node_idx-1)

                cell_to_output_w_name = lstm_node_name + '_cell_to_output_w'
                cell_to_output_w_op = op_adapter.ConstantOp(name=cell_to_output_w_name,
                                                            tensor= lstm_cell_split_weights[2])
                graph.add(cell_to_output_w_op, input_names=[], output_names=[cell_to_output_w_name],
                          idx=lstm_node_idx-1)

            input_gate_b_name = lstm_node_name + '_input_gate_b'
            input_gate_b_tensor = numpy.resize(gate_split_biases[0], (num_units,))
            input_gate_b_op = op_adapter.ConstantOp(name=input_gate_b_name, tensor=input_gate_b_tensor)
            graph.add(input_gate_b_op, input_names=[], output_names=[input_gate_b_name], idx=lstm_node_idx-1)

            proj_w_name = ""
            if lstm_node.op.proj_weights is not None:
                proj_w_name = lstm_node_name + '_proj_w'
                proj_w_tensor = numpy.resize(lstm_node.op.proj_weights, (output_size, num_units))
                proj_w_op = op_adapter.ConstantOp(name=proj_w_name, tensor=proj_w_tensor)
                graph.add(proj_w_op, input_names=[], output_names=[proj_w_name], idx=lstm_node_idx-1)

            proj_b_name = ""
            if lstm_node.op.proj_bias is not None:
                proj_b_name = lstm_node_name + '_proj_b'
                proj_b_tensor = numpy.resize(lstm_node.op.proj_bias, (output_size,))
                proj_b_op = op_adapter.ConstantOp(name=proj_b_name, tensor=proj_b_tensor)
                graph.add(proj_b_op, input_names=[], output_names=[proj_b_name], idx=lstm_node_idx-1)

            # Prepare input names - inputs not captured by any FE are passed the empty string
            lstm_node.input_names = [
                lstm_node.input_names[0],
                input_to_forget_w_name,
                input_to_cell_w_name,
                input_to_output_w_name,
                recurrent_to_forget_w_name,
                recurrent_to_cell_w_name,
                recurrent_to_output_w_name,
                forget_gate_b_name,
                cell_gate_b_name,
                output_gate_b_name,
                lstm_node.input_names[1],
                lstm_node.input_names[2],
                input_norm_w_name,
                forget_norm_w_name,
                cell_norm_w_name,
                output_norm_w_name,
                input_to_input_w_name,
                recurrent_to_input_w_name,
                cell_to_input_w_name,
                cell_to_forget_w_name,
                cell_to_output_w_name,
                input_gate_b_name,
                proj_w_name,
                proj_b_name
            ]
            all_output_buffer = graph.get_buffer(lstm_node.output_names[0])

            # Reshape is needed to restore time dimension if output buffer is not 2D
            output_reshape_needed = all_output_buffer.rank() != 2

            if number_of_outputs == 3:
                # Modify existing output buffers for QNN specification
                hidden_output_buffer = graph.get_buffer(lstm_node.output_names[2])
                hidden_output_buffer.shape = [batch_size, output_size]
                hidden_output_buffer.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

                cell_output_buffer = graph.get_buffer(lstm_node.output_names[1])
                cell_output_buffer.shape = [batch_size, num_units]
                cell_output_buffer.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                # Add dummy buffers for missing outputs - QNN requires 3
                hidden_output_dummy_name = lstm_node_name + "_hidden_output_dummy"
                graph.add_output_buffer(lstm_node, hidden_output_dummy_name,
                                        [batch_size, output_size], AxisTracker.AxisFormat.NONTRIVIAL)

                cell_output_dummy_name = lstm_node_name + "_cell_output_dummy"
                graph.add_output_buffer(lstm_node, cell_output_dummy_name,
                                        [batch_size, num_units], AxisTracker.AxisFormat.NONTRIVIAL)

            # Adding necessary reshape nodes to unsqueeze seq_length dimension for QNN
            if output_reshape_needed:
                output_all_reshape_node_name = lstm_node.output_names[0] + "_reshape"
                output_all_reshape_output_shape = [batch_size, seq_length, output_size]
                output_all_reshape_op = op_adapter.ReshapeOp(name=output_all_reshape_node_name,
                                                             output_shape=output_all_reshape_output_shape)
                graph.inject(output_all_reshape_op, input_name=all_output_buffer.name,
                             output_name=output_all_reshape_node_name,
                             consumer_names=[consumer.op.name for consumer in list(all_output_buffer.consumers)])
                # Setting up reshape output buffer axis format to be BTF
                reshape_buffer = graph.get_buffer(output_all_reshape_node_name)
                reshape_buffer.axis_format = AxisTracker.AxisFormat.BTF

                # change output buffer shape to 2D
                all_output_buffer.shape = [batch_size, output_size]
                all_output_buffer.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

            # Prepare output names
            if number_of_outputs == 1:
                lstm_node.output_names = [hidden_output_dummy_name, cell_output_dummy_name, all_output_buffer.name]
            else:
                lstm_node.output_names = [hidden_output_buffer.name, cell_output_buffer.name, all_output_buffer.name]


@register_layer_optimization
class OptimizeMatmulTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.MatMulOp.TRANSLATION_KEY
        self.register_method(ALIGN_MATMUL_RANKS, self.align_matmul_input_ranks)

    @staticmethod
    def align_matmul_input_ranks(node, graph):
        inputs = graph.get_input_buffers(node)
        output = graph.get_output_buffers(node)[0]
        log_debug1("Running matmal optimization for {}".format(node.op.name))
        if inputs[0].rank() != inputs[1].rank():
            log_debug1("Matmul {} input {} rank {} != input2 {} rank {}".format(node.op.name, inputs[0].name, inputs[0].rank(), inputs[1].name, inputs[1].rank()))
            lower_rank_input_buf, larger_rank_input_buf = (inputs[0], inputs[1]) \
                            if inputs[0].rank() < inputs[1].rank() else (inputs[1], inputs[0])

            # Adding reshape nodes to expand rank to match other input
            producer = lower_rank_input_buf.producer.op
            new_shape = translation_utils.expand_to_rank(lower_rank_input_buf.shape, len(larger_rank_input_buf.shape))
            log_debug1("This matmul impl requires identical rank, reshaping {} to {}".format(lower_rank_input_buf.shape, new_shape))
            if producer.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                producer.tensor = producer.tensor.reshape(new_shape)
                lower_rank_input_buf.shape = new_shape
            else:
                reshape_node_name = output.name + "_" + lower_rank_input_buf.name + "_reshape"
                reshape_op = op_adapter.ReshapeOp(name=reshape_node_name,
                                                  output_shape=new_shape)
                graph.inject(reshape_op, input_name=lower_rank_input_buf.name,
                             output_name=reshape_node_name, consumer_names=[node.op.name])

    def axes_to_spatial_first_order(self, node, graph):
        for input_name in node.input_names:
            input_buf = graph.get_buffer(input_name)
            # force convergence if necessary
            # use the 'backwards' permute orders because they are self-inverses.
            # Check if input is a permute, if so this means the source framework deliberately added the permute
            # and we do not want to inject another one.
            if input_buf.producer.op.type != op_adapter.PermuteOp.TRANSLATION_KEY:
                if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
                    graph.inject_implicit_permute(input_name, node.op.name, AxisTracker.AxisFormat.NCS,
                                                  AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
                    graph.inject_implicit_permute(input_name, node.op.name, AxisTracker.AxisFormat.TBF,
                                                  AxisTracker.AxisFormat.BTF_TO_TBF, [node.op.name])
                elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
                    pass
                elif input_buf.axis_format == AxisTracker.AxisFormat.FEATURE or \
                        input_buf.axis_format == AxisTracker.AxisFormat.ANY or \
                        input_buf.axis_format == AxisTracker.AxisFormat.NCS:
                    pass
                else:
                    raise ValueError(code_to_message.get_error_message("ERROR_MATMUL_UNEXPECTED_INPUT_ORDER")
                                     (input_buf.axis_format))

        output_buf = graph.get_output_buffers(node)[0]
        if output_buf.rank == 2:
            output_buf.axis_format = AxisTracker.AxisFormat.FEATURE
        else:
            output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL


@register_layer_optimization
class OptimizeMaxYTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.MaxYOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeNeuronTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.NeuronOp.TRANSLATION_KEY
        self.register_method(MATCH_HARDSWISH, self.match_hardswish)

    @staticmethod
    def match_hardswish(graph):
        def is_valid_hardswish(node_tuple):
            def check_for_valid_add_node(input_const_name):
                const_input_node = graph.get_producer_node(input_const_name)
                const_input_value = const_input_node.op.tensor
                const_input_length = reduce(lambda x,y:x * y, const_input_value.shape)
                temp = set(const_input_value.reshape(const_input_length))
                if len(temp) != 1 or int(temp.pop()) != 3:
                    return False
                return True

            def check_for_valid_neuron_node(node):
                if node.op.neuron_type != op_adapter.NeuronOp.Type.RELU_MIN_MAX \
                        or int(node.op.min_clamp) != 0 \
                        or int(node.op.max_clamp) != 6:
                    return False
                return True

            def check_for_valid_div_node(node):
                input_names = node.input_names
                const_input_nodes = get_input_const_nodes(input_names)
                const_input_value = const_input_nodes[0].op.tensor
                if numpy.array_equal(numpy.unique(const_input_value), [6]):
                  return True
                return False

            def check_for_valid_mul_node_with_const_input(node):
                def is_close_to_one_sixth(num):
                    return translation_utils.compare_values(float(num[0]), 1/6, rtol=1.e-3, atol=1.e-5)

                input_names = node.input_names
                const_input_nodes = get_input_const_nodes(input_names)
                const_input_value = const_input_nodes[0].op.tensor
                if const_input_value.shape != (1,) or not is_close_to_one_sixth(const_input_value):
                    return False
                return True

            add_node, neuron_node = node_tuple[0], node_tuple[1]
            add_non_const_input_name, add_const_input_name, mul_node, mul_node_const_input, div_node = [None] * 5
            for input_name in add_node.input_names:
                if graph.get_producer_op(input_name).type == op_adapter.ConstantOp.TRANSLATION_KEY:
                    add_const_input_name = input_name
                else:
                    add_non_const_input_name = input_name

            for node in node_tuple[2:]:
                if node.op.type == op_adapter.ElementwiseDivOp.TRANSLATION_KEY:
                    div_node = node
                else:
                    mul_input_names = node.input_names
                    if len(mul_input_names) != 2:
                        return False
                    if any(op_adapter.ConstantOp.TRANSLATION_KEY == graph.get_producer_op(input_name).type
                           for input_name in mul_input_names):
                        mul_node_const_input = node
                    else:
                        mul_node = node

            if not add_const_input_name or not mul_node or (not div_node and not mul_node_const_input):
                return False

            if add_non_const_input_name not in mul_node.input_names:
                # the add and mul must share same input_name to be matched as hswish
                return False

            return (check_for_valid_add_node(add_const_input_name) and
                    check_for_valid_neuron_node(neuron_node) and
                    (check_for_valid_div_node(div_node) if div_node else
                     check_for_valid_mul_node_with_const_input(mul_node_const_input)))

        def get_input_const_nodes(input_names):
            input_nodes = [graph.buffers[name].producer for name in input_names]
            const_nodes = [node for node in input_nodes if
                           node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY]
            return const_nodes

        def remove_const_nodes(node_tuple, matched_sequence_flag):
            if matched_sequence_flag[-1] in ['1', '3']:
                nodes_with_const_input = [node_tuple[0], node_tuple[3]]
            else:
                nodes_with_const_input = [node_tuple[0], node_tuple[2]]

            for node in nodes_with_const_input:
                const_node = get_input_const_nodes(node.input_names)[0]
                const_node_output_buf = graph.get_buffer(const_node.output_names[0])
                if len(const_node_output_buf.consumers) == 1:
                    # Only prune const_node if node is its only consumer
                    graph.prune(const_node, force_remove=True)
                else:
                    # Else, disconnect from node and leave const_node alone
                    const_node_output_buf.consumers.remove(node)
                    node.input_names.remove(const_node_output_buf.name)

        # Y = X*RELU6(X+3)*(1/6) or X*CLIP(X+3)*(1/6)
        sequence1 = [
            ("elementwise_sum",
             (),
             ("MATCH_NUM_BUFS", [("neuron", "ALL")])
             ),
            ("neuron",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY"),
                                 ("constant", "ANY")]),
             ()
             )
        ]

        # Y = X*(RELU6(X+3)*(1/6)) or X*(CLIP(X+3)*(1/6))
        sequence2 = [
            ("elementwise_sum",
             (),
             ("MATCH_NUM_BUFS", [("neuron", "ALL")])
             ),
            ("neuron",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("MATCH_NUM_BUFS", [("neuron", "ANY"),
                                 ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             (),
             ()
             )
        ]

        # Y = X*RELU6(X+3)/6 or X*CLIP(X+3)/6
        sequence3 = [
            ("elementwise_sum",
             (),
             ("MATCH_NUM_BUFS", [("neuron", "ALL")])
             ),
            ("neuron",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY"),
                                 ("constant", "ANY")]),
             ()
             )
        ]

        # Y = X*(RELU6(X+3)/6) or X*(CLIP(X+3)/6)
        sequence4 = [
            ("elementwise_sum",
             (),
             ("MATCH_NUM_BUFS", [("neuron", "ALL")])
             ),
            ("neuron",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("neuron", "ANY"),
                                 ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             (),
             ()
             )
        ]

        sequences = [sequence1, sequence2, sequence3, sequence4]

        for index, sequence in enumerate(sequences):
            matched_sequence_flag = 'matched_sequence' + str(index + 1)
            matched_node_list = graph.get_matched_nodes(sequence, validator=is_valid_hardswish, ignore_constants=True)

            for node_tuple in matched_node_list:
                remove_const_nodes(node_tuple, matched_sequence_flag)
                add_node = node_tuple[0]
                for node in node_tuple[:0:-1]:
                    for input_name in node.input_names:
                        if len(graph.get_buffer(input_name).consumers) == 1:
                            # per the sequence matching we know one of the inputs are squashable, hence
                            # check which either one has 1 consumer
                            graph.squash(node, input_name=input_name)

                add_op = add_node.op
                add_op_name = graph.naming_policy.get_op_name(add_op)
                hardswish_op_name = add_op_name + '_Hswish'
                hardswish_op = op_adapter.NeuronOp(hardswish_op_name, op_adapter.NeuronOp.Type.HSWISH)
                graph.replace(add_op, hardswish_op)


@register_layer_optimization
class OptimizeNoopTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.NoopOp.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        output_buf = graph.get_output_buffers(node)[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf.shape = input_buf.shape
        output_buf.axis_format = input_buf.axis_format

    @staticmethod
    def remove_noop(node, graph):
        graph.squash_noop(node)


@register_layer_optimization
class OptimizeOneHotTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.OneHotOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizePadTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PadOp.TRANSLATION_KEY
        self.register_method(SQUASH_PAD, self.squash_pad)

    @staticmethod
    def squash_pad(graph):
        def validate_node(nodes_tuple):
            pad_node_ = nodes_tuple[0]
            pads = pad_node_.op.pads
            # squash if all values are 0s
            if all(not (pad_0 or pad_1) for pad_0, pad_1 in pads) and \
                    len(graph.get_buffer(pad_node_.input_names[0]).consumers) == 1:
                return True
            return False

        sequence = [
            ("pad", (), ())
        ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        for node_tuple in matched_node_list:
            pad_node = node_tuple[0]
            graph.squash(pad_node, input_name=pad_node.input_names[0], no_op = True)

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            node.op.pads = AxisTracker.permute_shape(node.op.pads, AxisTracker.AxisFormat.NCS_TO_NSC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            node.op.pads = AxisTracker.permute_shape(node.op.pads, AxisTracker.AxisFormat.TBF_TO_BTF)
        node.op.pads = numpy.asarray(node.op.pads, dtype=numpy.dtype('int32'))
        AxisTracker.eltwise_to_spatial_first_order(node, graph)


@register_layer_optimization
class OptimizePoolTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PoolOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizePermuteTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PermuteOp.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]
        # check for trivial cases first, which will end up
        # in removal. Otherwise, just set output order to nontrivial
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            # special case: transforming to NSC, will become noop
            if node.op.order == [0, 2, 3, 1]:
                node.op.order = [0, 1, 2, 3]
                output_buf.axis_format = AxisTracker.AxisFormat.NSC
                return
            else:
                # going to nontrivial
                graph.inject_implicit_permute(input_name, node.op.name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            if node.op.order == [1, 0, 2]:
                node.op.order = [0, 1, 2]
                output_buf.axis_format = AxisTracker.AxisFormat.BTF
            else:
                graph.inject_implicit_permute(input_name, node.op.name, AxisTracker.AxisFormat.TBF,
                                              AxisTracker.AxisFormat.TBF_TO_BTF, [node.op.name])
                output_buf.axis_format = AxisTracker. AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL or \
                input_buf.axis_format == AxisTracker.AxisFormat.FEATURE:
            output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        else:
            raise ValueError(code_to_message.get_error_message("ERROR_PERMUTE_UNEXPECTED_INPUT_ORDER")
                             (input_buf.axis_format))

    @staticmethod
    def remove_noop(node, graph):
        input_buffer = graph.get_input_buffers(node)[0]
        output_buffer = graph.get_output_buffers(node)[0]
        if input_buffer.axis_format == output_buffer.axis_format and node.op.order == list(range(len(node.op.order))):
            # this permute is trivial, remove it
            graph.squash(node, input_name=input_buffer.name)


@register_layer_optimization
class OptimizePreluTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PreluOp.TRANSLATION_KEY

    @classmethod
    def _permute_coeff(cls, node, graph):
        input_buf = graph.get_buffer(node.input_names[0])
        coeff_shape = node.op.coeff.shape

        # determine the permute order(if any) after spatial first transformation
        # Note: only NSC, BTF formats imply permute was done.
        input_permute_order = None
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            input_permute_order = AxisTracker.AxisFormat.NCS_TO_NSC
        elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            input_permute_order = AxisTracker.AxisFormat.TBF_TO_BTF

        if len(node.op.coeff.shape) != 1 and len(node.op.coeff.shape) != len(input_buf.shape):
            raise ValueError("Prelu coefficient rank must equal either 1 or input rank {} for node {}. Got {} instead."
                             .format(len(input_buf.shape), node.op.name, len(node.op.coeff.shape)))

        if input_permute_order is not None and len(coeff_shape) > 1:
            # The input has been permuted hence we also need to permute coeff so that broadcasting persists
            node.op.coeff = numpy.ascontiguousarray(numpy.transpose(node.op.coeff, input_permute_order))
            coeff_shape = node.op.coeff.shape

        if not translation_utils.broadcastable(input_buf.shape, coeff_shape):
            raise ValueError(code_to_message.get_error_message("ERROR_OPERATION_INPUTS_NOT_BROADCASTABLE")
                             (node.op.name, input_buf.name, "coeff", input_buf.shape, coeff_shape))

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizePreluTranslation, self).axes_to_spatial_first_order(node, graph)
        # Input buffer axis might have been transformed, coeff need to be transformed as well
        OptimizePreluTranslation._permute_coeff(node, graph)


@register_layer_optimization
class OptimizeProposalTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ProposalOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):

        # change input dims to 4D as required by snpe. Handling this here since converter allows for
        # none 4D inputs. Note: only change dimensions if it is input and no other node is consuming it
        # TODO: how should this be really handled
        im_info_input_buf = graph.get_input_buffers(node)[-1]
        if im_info_input_buf.producer.op.type == op_adapter.InputOp.TRANSLATION_KEY \
                and len(im_info_input_buf.consumers) == 1 \
                and im_info_input_buf.rank() != 4:
            shape = translation_utils.expand_to_rank(im_info_input_buf.shape, 4)
            im_info_input_buf.shape = shape
            im_info_input_buf.producer.op.shape = shape
            im_info_input_buf.axis_format = AxisTracker.AxisFormat.NSC

        super(OptimizeProposalTranslation, self).axes_to_spatial_first_order(node, graph)


@register_layer_optimization
class OptimizeQuantizeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.QuantizeOp.TRANSLATION_KEY
        self.register_method(REMOVE_QUANT_NODES, self.remove_quant_nodes)
        self.register_method(SQUASH_QUANT_NODES, self.squash_quant_dequant_to_convert)

    @staticmethod
    def squash_quant_dequant_to_convert(graph):
        sequence = [
                    (op_adapter.QuantizeOp.TRANSLATION_KEY,
                        (),
                        ()
                     ),
                    (op_adapter.DequantizeOp.TRANSLATION_KEY,
                        (),
                        ()
                     )
                   ]

        matched_node_list = graph.get_matched_nodes(sequence)
        for node_tuple in matched_node_list:

                # We found a quant/dequant combo, extract the nodes.
                first, second = node_tuple
                second_input_buffer = graph.get_input_buffers(second)[0]
                first_input_buffer = graph.get_input_buffers(first)[0]
                first_output_buffer = graph.get_output_buffers(first)[0]
                producer = first_input_buffer.producer

                # Fold these nodes into a convert op. Quant params are folded as part of squashing
                convert_name = producer.output_names[0] + "_convert_quant_dequant"
                convert_op = op_adapter.ConvertOp(convert_name)
                convert_node = graph.inject(convert_op, input_name=first_input_buffer.name, output_name=convert_name, consumer_names=[first.op.name])
                convert_input_buffer = graph.get_output_buffers(producer)[0]
                log_debug('Injecting convert op {} with input {} and output {}'.format(convert_name, convert_input_buffer.name, convert_name))
                convert_output_buffer = graph.get_output_buffers(convert_node)[0]
                log_debug('Found {} and {} nodes to squash into {} '.format(first.op.name,second.op.name,convert_op.name))
                graph.squash(second, input_name=second_input_buffer.name)
                graph.squash(first, input_name=convert_output_buffer.name)

    @staticmethod
    def remove_quant_nodes(node, graph):
        # Squash the quant node. The quant params are folded as part of squashing
        graph.squash(node, input_name=node.input_names[0])
        log_debug("Remove quantize op {}".format(node.op.name))


class OptimizeReduceTranslationBase(OptimizationTranslationBase):
    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]

        if input_buf.axis_format in format_to_permute_order:
            target_format = format_to_format[input_buf.axis_format]
            permute_order = format_to_permute_order[input_buf.axis_format]
            # If keep dims = 0 we must permute as it will remove dimensions
            if not node.op.keep_dims:
                graph.inject_implicit_permute(input_name, node.op.name, target_format,
                                              permute_order, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                AxisTracker.eltwise_to_spatial_first_order(node, graph)
                axis_map = permute_order
                node.op.axes = [axis_map[axis] for axis in node.op.axes]
        else:
            output_buf = graph.get_buffer(node.output_names[0])
            output_buf.axis_format = input_buf.axis_format


@register_layer_optimization
class OptimizeReduceMaxTranslation(OptimizeReduceTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ReduceMaxOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeReduceMeanTranslation(OptimizeReduceTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ReduceMeanOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeReduceMinTranslation(OptimizeReduceTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ReduceMinOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeReduceProdTranslation(OptimizeReduceTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ReduceProdOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeReduceSumTranslation(OptimizeReduceTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ReduceSumOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeReshapeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ReshapeOp.TRANSLATION_KEY
        self.register_method(MATCH_CHANNELSHUFFLE, self.match_channelshuffle)
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        # force convergence if necessary
        # use the 'backwards' permute orders because they are self-inverses.
        # Check if input is a permute, if so this means the source framework deliberately added the permute
        # and we do not want to inject another one.
        if input_buf.producer.op.type != op_adapter.PermuteOp.TRANSLATION_KEY:
            if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
                graph.inject_implicit_permute(input_name, node.op.name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
                graph.inject_implicit_permute(input_name, node.op.name, AxisTracker.AxisFormat.TBF,
                                              AxisTracker.AxisFormat.BTF_TO_TBF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
                pass
            elif input_buf.axis_format == AxisTracker.AxisFormat.FEATURE or \
                    input_buf.axis_format == AxisTracker.AxisFormat.ANY or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCS:
                pass
            else:
                raise ValueError(code_to_message.get_error_message("ERROR_RESHAPE_UNEXPECTED_INPUT_ORDER")
                                 (input_buf.axis_format))

        output_buf = graph.get_output_buffers(node)[0]
        output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

    @staticmethod
    def match_channelshuffle(graph):
        def is_valid_channelshuffle(nodes_tuple):
            def check_for_valid_reshape_1(node):
                input_buffer = graph.get_input_buffers(node)[0]
                output_buffer = graph.get_output_buffers(node)[0]
                reshape_1_input_shape = input_buffer.shape
                reshape_1_output_shape = output_buffer.shape

                return (len(reshape_1_input_shape) == 4 and len(reshape_1_output_shape) == 5 and
                        reshape_1_input_shape[0] == reshape_1_output_shape[0] and
                        reshape_1_input_shape[2] == reshape_1_output_shape[3] and
                        reshape_1_input_shape[3] == reshape_1_output_shape[4])

            def check_for_valid_permute(node):
                # Assuming the input shape is N[GC']HW
                return node.op.type == op_adapter.PermuteOp.TRANSLATION_KEY and node.op.order == [0, 2, 1, 3, 4]

            def check_for_valid_reshape_2(node):
                input_buffer = graph.get_input_buffers(node)[0]
                output_buffer = graph.get_output_buffers(node)[0]
                reshape_2_input_shape = input_buffer.shape
                reshape_2_output_shape = output_buffer.shape

                return (len(reshape_2_input_shape) == 5 and len(reshape_2_output_shape) == 4 and
                        reshape_2_input_shape[0] == reshape_2_output_shape[0] and
                        reshape_2_input_shape[3] == reshape_2_output_shape[2] and
                        reshape_2_input_shape[4] == reshape_2_output_shape[3])

            first_, second_, third_ = nodes_tuple
            input_shape_ = graph.get_input_buffers(first_)[0].shape
            output_shape_ = graph.get_output_buffers(third_)[0].shape

            return ((output_shape_ == input_shape_) and
                    check_for_valid_reshape_1(first_) and
                    check_for_valid_permute(second_) and
                    check_for_valid_reshape_2(third_))

        sequence = [
                    ("reshape",
                        (),
                        ("MATCH_NUM_BUFS", [("permute", "ALL")])
                     ),
                    ("permute",
                        (),
                        ("MATCH_NUM_BUFS", [("reshape", "ALL")])
                     ),
                    ("reshape",
                        (),
                        ()
                     )
                   ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=is_valid_channelshuffle)

        for node_tuple in matched_node_list:

                # ChannelShuffle Op found,
                # Squash Permute and 2nd Reshape Op and
                # Replace 1st ReshapeOp with ShuffleOp
                first, second, third = node_tuple
                third_input_buffer = graph.get_input_buffers(third)[0]
                graph.squash(third, input_name=third_input_buffer.name)

                second_input_buffer = graph.get_input_buffers(second)[0]
                graph.squash(second, input_name=second_input_buffer.name)

                output_shape = first.op.output_shape
                # Assuming the shape is N[GC']HW
                groups = output_shape[1]
                shuffle_op = op_adapter.ChannelShuffleOp(None, groups=groups)
                shuffle_op.name = graph.naming_policy.get_op_name(shuffle_op)
                graph.replace(first.op, shuffle_op)
                log_debug2(code_to_message.get_debugging_message("DEBUG_CHANNEL_SHUFFLE_REPLACE")(first.op.name,
                                                                                                  second.op.name,
                                                                                                  third.op.name,
                                                                                                  shuffle_op.name))

    @staticmethod
    def remove_noop(node, graph):
        input_buffer = graph.get_input_buffers(node)[0]
        consumers = list(graph.get_buffer(node.output_names[0]).consumers)
        ret = False
        # Remove reshape if same shape as input as this reshape has no effect, remove it
        if input_buffer.shape == node.op.output_shape and len(input_buffer.consumers) == 1:
            ret = graph.squash(node, input_name=input_buffer.name)
        # Remove reshape  if the batch dimension is maintained through the reshape when consumer of reshape is
        # fc layer
        elif len(consumers) == 1 and isinstance(consumers[0].op, op_adapter.FullyConnectedOp) and \
                input_buffer.shape[0] == node.op.output_shape[0]:
            ret = graph.squash(node, input_name=input_buffer.name, squash_into_next=True)
        if ret:
            log_debug("Squash Reshape op {} due to Noop. "
                      "Input shape {}, shape attr {}".format(node.op.name,
                                                             input_buffer.shape,
                                                             node.op.output_shape))


@register_layer_optimization
class OptimizeRNormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RNormOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeRoiAlignTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RoiAlignOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.enforce_input_type(graph, node.input_names[0], node.op.name, AxisTracker.AxisFormat.NSC,
                                         AxisTracker.AxisFormat.NCS_TO_NSC)
        output_buf = graph.get_output_buffers(node)[0]
        node.op.output_shape = output_buf.shape = AxisTracker.permute_shape(output_buf.shape,
                                                                            AxisTracker.AxisFormat.NCS_TO_NSC)
        output_buf.axis_format = AxisTracker.AxisFormat.NSC


@register_layer_optimization
class OptimizeRoiPoolingTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RoiPoolingOp.TRANSLATION_KEY
        self.register_method("PREPROCESS_ROI_POOL_INPUTS", self.preprocess_roi_pool_inputs)

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.enforce_input_type(graph, node.input_names[0], node.op.name, AxisTracker.AxisFormat.NSC,
                                         AxisTracker.AxisFormat.NCS_TO_NSC)
        output_buf = graph.get_output_buffers(node)[0]
        node.op.output_shape = output_buf.shape = AxisTracker.permute_shape(output_buf.shape,
                                                                            AxisTracker.AxisFormat.NCS_TO_NSC)
        output_buf.axis_format = AxisTracker.AxisFormat.NSC

    @staticmethod
    def preprocess_roi_pool_inputs(graph):
        def validate_node(nodes_tuple):
            roi_node = nodes_tuple[0]
            roi_buf = graph.get_buffer(roi_node.input_names[1])
            # Batch indices are embedded in the ROI input for some frameworks
            # as (batch_index, x1, y1, x2, y2....). In this case the ROI must be static
            # so that the batch index input can be extracted
            if roi_buf.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY or len(roi_node.input_names) == 3:
                return True
            return False

        sequence = [(op_adapter.RoiPoolingOp.TRANSLATION_KEY, (), ())]

        matched_nodes_list = graph.get_matched_nodes(sequence, validator=validate_node)

        for nodes_tuple in matched_nodes_list:
            roi_node = nodes_tuple[0]
            roi_buf = graph.get_buffer(roi_node.input_names[1])

            # Batch indices are embedded in the ROI input for some frameworks
            # as (batch_index, x1, y1, x2, y2....). In this case the ROI must be static
            # so that the batch index input can be extracted
            if roi_buf.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                if roi_buf.shape[-1] == 5:
                    # QNN needs roi values to be separated from batch index
                    roi_values = roi_buf.producer.op.tensor
                    roi_values_no_batch = roi_values[:, 1:]

                    # Update ROI values in constant op to new values
                    roi_buf.producer.op.tensor = roi_values_no_batch

                    # Set batch indices to first sub-tensor of ROI values
                    batch_indices_name = roi_buf.name + "_batch_indices"
                    batch_indices = numpy.asarray(roi_values[:, 0], numpy.int32)

                    # Add a new constant op to capture batch indices

                    # constant op needs to be added before roi node
                    roi_idx = graph.nodes_in_order.index(roi_node)
                    graph.add(op_adapter.ConstantOp(batch_indices_name, batch_indices, quantizable=False), [],
                              [batch_indices_name], idx=roi_idx)

                    # add input name to roi node
                    roi_node.input_names.append(batch_indices_name)

                else:
                    raise ValueError("Expected 5 dimensions for static ROI buffer: {}, instead got {}"
                                     .format(roi_buf.name, roi_buf.shape[-1]))
            elif len(roi_node.input_names) != 3:
                raise AttributeError("Missing batch indices input. "
                                     "Expected 3 inputs for ROI operation instead got: {}"
                                     .format(len(roi_node.input_names)))


@register_layer_optimization
class OptimizeResizeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ResizeOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        node.op.output_shape = AxisTracker.permute_shape(node.op.output_shape, AxisTracker.AxisFormat.NCS_TO_NSC)
        AxisTracker.image_to_spatial_first_order(node, graph)


@register_layer_optimization
class OptimizeRnnTransformationTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RnnTransformationOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.time_series_to_spatial_first_order(node, graph)


@register_layer_optimization
class OptimizeScaleTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ScaleOp.TRANSLATION_KEY
        self.register_method(SQUASH_SCALE, self.squash_scale)

    @staticmethod
    def squash_scale(graph):
        def validate_node(nodes_tuple):
            scale_node_ = nodes_tuple[0]
            input_buffer_ = graph.get_input_buffers(scale_node_)[0]
            input_node_ = input_buffer_.producer
            input_ir_shapes = [graph.src_axis_order.permute_shape_to_ir(input_buffer_.get_buf_dims()),
                               graph.src_axis_order.permute_shape_to_ir(scale_node_.op.weights.shape)]
            # scale should only be folded if it is the only layer that depends on the output of the previous
            # batchnorm layer/op.
            if len(input_buffer_.consumers) == 1:
                # Only valid to squash if the bn_op has act output that is broadcastable with the scale weights AND
                # the scale weights and bias are same shape with bn_op bias
                if input_node_.op.bias.shape == scale_node_.op.bias.squeeze().shape and \
                    input_node_.op.bias.shape == scale_node_.op.weights.squeeze().shape and \
                        translation_utils.broadcastable(*input_ir_shapes):
                    return True
            return False

        sequence = [
                    (op_adapter.ScaleOp.TRANSLATION_KEY,
                        # Check if the previous layer was a batchnorm
                        ("MATCH_NUM_BUFS", [(op_adapter.BatchnormOp.TRANSLATION_KEY, "ALL")]),
                        ()
                     )
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        for node_tuple in matched_node_list:
            # retain scale information in batchnorm op so that it can be used for quantization
            # scale_weights and scale_bias map to gamma and beta respectively.
            node = node_tuple[0]
            prev = graph.get_input_buffers(node)[0].producer
            prev.op.gamma = node.op.weights
            prev.op.beta = node.op.bias

        squash_node_into_nn_node(graph, matched_node_list)

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeScaleTranslation, self).axes_to_spatial_first_order(node, graph)
        buf = graph.get_buffer(node.output_names[0])
        if buf.axis_format == AxisTracker.AxisFormat.NSC:
            axis_map = graph.src_axis_order.permute_sequence[buf.rank() - 1]
            node.op.axis = axis_map[node.op.axis]


@register_layer_optimization
class OptimizeScatterNDTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ScatterNDOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name, indices_name, updates_name = node.input_names
        input_buf = graph.get_input_buffers(node)[0]
        indices_buf = graph.get_input_buffers(node)[1]
        updates_buf = graph.get_input_buffers(node)[2]

        output_buf = graph.get_output_buffers(node)[0]

        # Check if any of the buffers have been changed into NSC, BTF order and revert if so
        if input_buf.axis_format in [AxisTracker.AxisFormat.NSC, AxisTracker.AxisFormat.BTF]:
            graph.inject_implicit_permute(input_name, node.op.name, format_to_format[input_buf.axis_format],
                                          format_to_permute_order[input_buf.axis_format], [node.op.name])
            # output_shape = input shape, so format is NON-TRIVIAL
            output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        if indices_buf.axis_format in [AxisTracker.AxisFormat.NSC, AxisTracker.AxisFormat.BTF]:
            graph.inject_implicit_permute(input_name, node.op.name, format_to_format[indices_buf.axis_format],
                                          format_to_permute_order[indices_buf.axis_format], [node.op.name])
        if updates_buf.axis_format in [AxisTracker.AxisFormat.NSC, AxisTracker.AxisFormat.BTF]:
            graph.inject_implicit_permute(input_name, node.op.name, format_to_format[updates_buf.axis_format],
                                          format_to_permute_order[updates_buf.axis_format], [node.op.name])


@register_layer_optimization
class OptimizeSliceTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SliceOp.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format in format_to_permute_order:
            axis_map = format_to_permute_order[input_buf.axis_format]
            node.op.axis = axis_map[node.op.axis]
        AxisTracker.eltwise_to_spatial_first_order(node, graph)

    @staticmethod
    def remove_noop(node, graph):
        if not len(node.op.slice_points):
            graph.squash(node, input_name=node.input_names[0])


@register_layer_optimization
class OptimizeSoftmaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SoftmaxOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        # NB will probably want to switch to 'eltwise' version when we
        # support axis parameter.
        input_buf = graph.get_buffer(node.input_names[0])
        # Added this check for any 4D input for frcnn_vgg_compressed model
        # where it expects a permute after reshape
        if input_buf.rank() == 4 and input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL and node.op.axis == 3:
            log_debug("Unsupported axis param {} in native axis format, don't permute".format(node.op.axis))
            output_buf = graph.get_buffer(node.output_names[0])
            output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        else:
            AxisTracker.eltwise_to_spatial_first_order(node, graph)

        # Ensure we're using the correct input buffer as a permute might have been inserted above
        input_buf = graph.get_buffer(node.input_names[0])
        if input_buf.axis_format in format_to_permute_order:
            axis_map = format_to_permute_order[input_buf.axis_format]
            node.op.axis = axis_map[node.op.axis]
            log_debug('Mapping axis from {} to {}: '.format(node.op.axis,  axis_map[node.op.axis]))


@register_layer_optimization
class OptimizeStaticTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.StaticOp.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        pass

    @staticmethod
    def remove_noop(node, graph):
        graph.prune(node)


@register_layer_optimization
class OptimizeSubtractMeanTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SubtractMeanOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeUdlTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.UdlOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_names = node.input_names
        for input_name in input_names:
            input_buf = graph.get_buffer(input_name)
            current_input_order = input_buf.get_axis_annotations()
            expected_input_order = []
            for dims in node.op.expected_input_axis_orders:
                if len(dims) == input_buf.rank():
                    expected_input_order = dims
            target_input_type = AxisTracker.get_axis_format_from_annotation(expected_input_order)
            permute_order = AxisTracker.compute_permute_order(current_input_order, expected_input_order)
            if len(permute_order) and permute_order != list(range(len(permute_order))):
                graph.inject_implicit_permute(input_name, node.op.name, target_input_type,
                                              permute_order, [node.op.name])

            target_output_order = []
            output_buffers = graph.get_output_buffers(node)
            for output_buf in output_buffers:
                for dims in node.op.expected_output_axis_orders:
                    if len(dims) == output_buf.rank():
                        target_output_order = dims
                output_buf.axis_format = AxisTracker.get_axis_format_from_annotation(target_output_order)


@register_layer_optimization
class OptimizeUpsampleIndexBaseTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.UpsampleIndexBasedOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeUpsampleSparseTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.UpsampleSparseOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeCropAndResizeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CropAndResizeOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeEmbeddingTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.EmbeddingOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeExtractGlimpseTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ExtractGlimpseOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeImageProjectiveTransformTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ImageProjectiveTransformOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeMomentTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.MomentOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeNonMaxSuppressionTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.NonMaxSuppresionOp.TRANSLATION_KEY
        self.register_method(ADJUST_NMS_FEATURE_DIMS, self.adjust_nms_feature_dimensions)

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeNonMaxSuppressionTranslation, self).axes_to_spatial_first_order(node, graph)
        boxes_shape = graph.get_buffer(node.input_names[0]).shape
        scores_shape = graph.get_buffer(node.input_names[1]).shape

        if scores_shape[1] == boxes_shape[1]:
            pass
        elif scores_shape[2] == boxes_shape[1]:
            graph.inject_implicit_permute(node.input_names[1], node.op.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                          [0, 2, 1], [node.op.name])
        else:
            raise ValueError(
                "Unable to get proper axis order for {} to expected [batch, num_boxes, num_classes]. Cannot match "
                "input shapes {} and {}".format(node.input_names[1], boxes_shape, scores_shape)
            )

    @staticmethod
    def adjust_nms_feature_dimensions(graph):
        """
        By default nms requires 2 inputs for boxes and score whose input and output shape is handled in
        TF translation. With the extra input_features they do not typically come with batch dimensions, so handle
        here by verifying required second dimension equality with num_boxes
        TODO: remove once backend consolidate input/output shapes of features to MultiClassNms. This should be
        handled during TF translation similar to the boxes and scores input.
        """

        def validate_node(nodes_tuple):
            nms_node_ = nodes_tuple[0]
            # adjustment of features only needed if features are given as inputs
            if len(nms_node_.input_names) > 2 and len(nms_node_.output_names) > 4 and \
                    "scale_y" not in nms_node_.op.attrs:
                return True
            return False

        sequence = [
            ("non_max_suppression",
             (),
             ()
             )
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        for node_tuple in matched_node_list:
            nms_node = node_tuple[0]
            nms_input_names = nms_node.input_names
            nms_output_names = nms_node.output_names
            num_boxes = graph.get_buffer(nms_node.input_names[0]).shape[1]
            for i in range(2, len(nms_node.input_names)):
                input_feature_buf = graph.get_buffer(nms_input_names[i])
                input_feature_shape = input_feature_buf.shape
                if len(input_feature_shape) == 1 or input_feature_shape[1] != num_boxes:
                    input_feature_node = graph.get_producer_node(nms_input_names[i])
                    # add reshape node to add batch dimension to the input features
                    expected_input_feature_shape = [1, *input_feature_shape]
                    # verify this is will result in expected input
                    log_assert(expected_input_feature_shape[1] == num_boxes,
                               "Unable to adjust input feature to match expected num_boxes on second dimension. "
                               "Got: {}, Expected num_boxes {}".format(expected_input_feature_shape, num_boxes))

                    if input_feature_node.op.type == op_adapter.ReshapeOp.TRANSLATION_KEY and \
                            graph.get_buffer(input_feature_node.input_names[0]).shape == expected_input_feature_shape:
                        # there was a squeeze done to remove batch dim, remove it and adjust to expected
                        # input feature instead.
                        graph.squash(input_feature_node, input_name=input_feature_node.input_names[0])
                        graph.get_buffer(input_feature_node.output_names[0]).set_buf_dims(expected_input_feature_shape)
                    else:
                        # add the reshape to add batch dim
                        input_feature_reshape_node_name = nms_input_names[i] + "_reshape_batch_add"
                        input_feature_reshape_op = op_adapter.ReshapeOp(name=input_feature_reshape_node_name,
                                                                        output_shape=expected_input_feature_shape)
                        graph.inject(input_feature_reshape_op, input_name=nms_input_names[i],
                                     output_name=input_feature_reshape_node_name,
                                     consumer_names=[nms_node.op.name])

                    # since we are reshaping input, output from nms will need to be adjusted as intermediate and
                    # will require a post reshape to remove batch dimension added.
                    output_name_idx = i + 2  # accounting for class and num_det output
                    output_feature_name = nms_output_names[output_name_idx]
                    output_feature_buf = graph.get_buffer(output_feature_name)
                    # replace the nms output as intermediate and the post reshaped output as the src fw output_feature
                    graph.delete_buffer(output_feature_name)
                    output_feature_reshape_op = op_adapter.ReshapeOp(name=output_feature_name,
                                                                     output_shape=output_feature_buf.shape)
                    # adjust to expected buffer shape for nms feature output(i.e with batch dim) and rename buffer as
                    # intermediate
                    output_feature_buf.set_buf_dims([1, *output_feature_buf.shape])
                    intermediate_output_name = output_feature_name + "_intermediate"
                    output_feature_buf.name = intermediate_output_name
                    graph.add_buffer(output_feature_buf)
                    nms_output_names[output_name_idx] = intermediate_output_name
                    graph.inject(output_feature_reshape_op, input_name=intermediate_output_name,
                                 output_name=output_feature_name)

                    # Addition of a const tensor to features should not be quantized
                    # TODO: add conditional that it should be set non quantizable based on tensortype and
                    #       quantization info of input tensor when irgraph supports these info
                    output_feature_reshape_buf = graph.get_buffer(output_feature_name)
                    for consumer in output_feature_reshape_buf.consumers:
                        if isinstance(consumer.op, op_adapter.ElementwiseOp):
                            for input_name in consumer.input_names:
                                eltwise_input_node = graph.get_producer_node(input_name)
                                if eltwise_input_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                                    eltwise_input_node.op.quantizable = False


@register_layer_optimization
class OptimizeFakeNonMaxSuppressionTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.FakeNonMaxSuppressionOp.TRANSLATION_KEY
        self.register_method(MERGE_LOW_LEVEL_OPS_TO_LAYERS, self.merge_low_level_ops_to_layers)

    def merge_low_level_ops_to_layers(self, graph):
        validate_node = None

        sequence1 = [
            ("fake_non_max_suppression",
             (),
             ("MATCH_NUM_BUFS", [("gather", "ALL")])
            ),
            ("constant",
             (),
             ("MATCH_NUM_BUFS", [("gather", "ALL")])
            ),
            ("gather",
             ("MATCH_NUM_BUFS", [("fake_non_max_suppression", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("reshape", "ALL")]),
            ),
            ("reshape",
             ("MATCH_NUM_BUFS", [("gather", "ALL")]),
             ("FLEXIBLE_NUM_BUFS", [("gather", "ALL")])
            ),
        ]
        sequence2 = [
            ("fake_non_max_suppression",
             (),
             ("MATCH_NUM_BUFS", [("strided_slice", "ALL")])
            ),
            ("strided_slice",
             ("MATCH_NUM_BUFS", [("fake_non_max_suppression", "ALL")]),
             ("FLEXIBLE_NUM_BUFS", [("gather", "ALL")])
            )
        ]

        sequences = [sequence1, sequence2]

        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
            for node_tuple in matched_node_list:
                fake_nms_node = node_tuple[0]
                fake_nms_op = node_tuple[0].op
                nms_output_names = ['{}_boxes'.format(fake_nms_op.name),
                                    '{}_scores'.format(fake_nms_op.name),
                                    '{}_classes'.format(fake_nms_op.name),
                                    'num_detections']

                nms_max_total_detections = node_tuple[0].op.max_total_detections
                nms_iou_threshold = node_tuple[0].op.iou_threshold
                nms_score_threshold = node_tuple[0].op.score_threshold

                # Replace to real NMS
                nms_op_name = graph.naming_policy.get_op_name(fake_nms_op) + '_gather'
                nms_op = op_adapter.NonMaxSuppresionOp(nms_op_name,
                                                       max_total_detections=nms_max_total_detections,
                                                       iou_threshold=nms_iou_threshold,
                                                       score_threshold=nms_score_threshold
                                                      )
                nms_input_names = node_tuple[0].input_names.copy()
                last_node = node_tuple[-1]
                last_output_buf = graph.get_output_buffers(last_node)[0]

                pruned_nodes = []
                box_n_class_succors = []
                box_n_class_succors_input = []
                feature_consumer_succors = []
                for consumer in last_output_buf.consumers:
                    if consumer.op.type == 'gather':
                        consumer_input_names = consumer.input_names
                        gather_data_inputs = [input_name for input_name in consumer_input_names if input_name != last_output_buf.name]
                        # boxes and classes nodes have been done in nms_output_names
                        # therefore no need to create an extra output from gather op
                        if gather_data_inputs[0] in nms_input_names[:2]:
                            box_n_class_succors_input.append(nms_output_names[nms_input_names.index(gather_data_inputs[0])])
                            box_n_class_succors.append(graph.get_output_buffers(consumer)[0].consumers)
                        # feature parts, which need to be added as extra outputs
                        # connected the graph by nms output[4:]
                        else:
                            nms_input_names.extend(gather_data_inputs)
                            nms_output_names.extend(consumer.output_names)
                            # gather has only one output buffer
                            feature_consumer_succors.append(graph.get_output_buffers(consumer)[0].consumers)
                        pruned_nodes.append(consumer)
                for node in pruned_nodes:
                    graph.prune(node, force_remove=True)

                # Prune the nodes after extract required information
                for node_in_tuple in reversed(node_tuple):
                    graph.prune(node_in_tuple, force_remove=True)
                idx_to_insert = 0
                for input_name in nms_input_names:
                    buf = graph.get_buffer(input_name)
                    cur_idx = graph.nodes_in_order.index(buf.producer)
                    if idx_to_insert < cur_idx:
                        idx_to_insert = cur_idx + 1
                nms_node = graph.add(nms_op, input_names=nms_input_names, output_names=nms_output_names,idx=idx_to_insert)
                # re-connected the nodes after gather
                # box, scores part
                for idx, succs in enumerate(box_n_class_succors):
                    for succ_node in succs:
                        succ_node.input_names.append(box_n_class_succors_input[idx])
                        nms_output_buf = graph.get_buffer(nms_output_names[idx])
                        nms_output_buf.consumers.add(succ_node)
                # feature part
                for idx, succs in enumerate(feature_consumer_succors):
                    succ_input_name = nms_output_names[4+idx]
                    for succ_node in succs:
                        succ_node.input_names.append(succ_input_name)
                        nms_output_buf = graph.get_buffer(nms_output_names[4+idx])
                        nms_output_buf.consumers.add(succ_node)


@register_layer_optimization
class OptimizePackTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PackOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        for input_name in node.input_names:
            input_buf = graph.get_buffer(input_name)
            # Check if input is a permute, if so this means the source framework deliberately added the permute
            # and we do not want to inject another one.
            if input_buf.producer.op.type != op_adapter.PermuteOp.TRANSLATION_KEY:
                if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
                    graph.inject_implicit_permute(input_name, node.op.name, AxisTracker.AxisFormat.NCS,
                                                  AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
                    graph.inject_implicit_permute(input_name, node.op.name, AxisTracker.AxisFormat.TBF,
                                                  AxisTracker.AxisFormat.BTF_TO_TBF, [node.op.name])

        output_buf = graph.get_output_buffers(node)[0]
        output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL


@register_layer_optimization
class OptimizePixelShuffleTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PixelShuffleOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeStridedSliceTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.StridedSliceOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeStridedSliceTranslation, self).axes_to_spatial_first_order(node, graph)
        # begin, end and strides need to reorder to follow axis format
        input_buf = graph.get_buffer(node.input_names[0])
        if input_buf.axis_format != AxisTracker.AxisFormat.NONTRIVIAL:
            node.op.begin = graph.src_axis_order.permute_shape_to_ir(node.op.begin)
            node.op.end = graph.src_axis_order.permute_shape_to_ir(node.op.end)
            node.op.strides = graph.src_axis_order.permute_shape_to_ir(node.op.strides)


@register_layer_optimization
class OptimizeSpaceToDepthTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SpaceToDepthOp.TRANSLATION_KEY
        self.register_method(MATCH_SPACETODEPTH, self.match_spacetodepth)

    @staticmethod
    def match_spacetodepth(graph):
        # check shapes in the reshape layers
        # [n, c, h, w] -> [n, c * blk**2, h/blk, w/blk]
        def is_valid_spacetodepth(node_tuple):
            input_buf = graph.get_input_buffers(node_tuple[0])[0]
            input_shape = input_buf.shape
            first_reshape_output_shape = node_tuple[0].op.output_shape
            if len(input_shape) == 4 and len(first_reshape_output_shape) == 6:
                blocksize = first_reshape_output_shape[3]
                sequence_output_shape = node_tuple[-1].op.output_shape

                batch, height, width, depth = graph.src_axis_order.extract_spatial_dims(input_shape)
                expected_shape = graph.src_axis_order.format_spatial_output_shape(batch_size=batch,
                                                                                  depth=depth * (blocksize**2),
                                                                                  height=height//blocksize,
                                                                                  width=width//blocksize)
                return sequence_output_shape == expected_shape
            else:
                return False

        # reshape:   [n, c, h/blk1, blk1, w/blk2, blk2], blk1 == blk2, number is for transpose order.
        # transpose: [n, c, h/blk1, w/blk2, blk1, blk2]
        # reshape:   [n, c, h/blk * w/blk, blk ** 2]
        # transpose: [n, c, blk ** 2, h/blk * w/blk]
        # reshape:   [n, c, blk ** 2, h/blk, w/blk]
        # transpose: [n, blk ** 2, c, h/blk, w/blk]
        # reshape:   [n, c*(blk**2), h/blk, w/blk]
        sequence = [
            ("reshape",
             (),
             ("MATCH_NUM_BUFS", [("permute", "ALL")])
            ),
            ("permute",
             ("MATCH_NUM_BUFS", [("reshape", "ALL")]),
             ("MATCH_NUM_BUFS", [("reshape", "ALL")])
            ),
            ("reshape",
             ("MATCH_NUM_BUFS", [("permute", "ALL")]),
             ("MATCH_NUM_BUFS", [("permute", "ALL")])
            ),
            ("permute",
             ("MATCH_NUM_BUFS", [("reshape", "ALL")]),
             ("MATCH_NUM_BUFS", [("reshape", "ALL")]),
            ),
            ("reshape",
             ("MATCH_NUM_BUFS", [("permute", "ALL")]),
             ("MATCH_NUM_BUFS", [("permute", "ALL")])
            ),
            ("permute",
             ("MATCH_NUM_BUFS", [("reshape", "ALL")]),
             ("MATCH_NUM_BUFS", [("reshape", "ALL")]),
            ),
            ("reshape",
             ("MATCH_NUM_BUFS", [("permute", "ALL")]),
             ()
            )
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=is_valid_spacetodepth, ignore_constants=True)
        for node_tuple in matched_node_list:
            blocksize = node_tuple[0].op.output_shape[3]
            reshape_node = node_tuple[0]
            # Squash all nodes except the first reshape in reverse order
            # the first reshape op will be replaced
            for node in node_tuple[:0:-1]:
                for input_name in node.input_names:
                    graph.squash(node, input_name=input_name)
            reshape_op = reshape_node.op
            reshape_op_name = graph.naming_policy.get_op_name(reshape_op)
            spacetodepth_op_name = reshape_op_name + '_space_to_depth'
            spacetodepth_op = op_adapter.SpaceToDepthOp(spacetodepth_op_name, downscale_factor=blocksize)
            graph.replace(reshape_op, spacetodepth_op)


@register_layer_optimization
class OptimizeSsdTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SsdOp.TRANSLATION_KEY
        self.register_method(SQUASH_BOX_DECODER, self.squash_box_decoder)

    @staticmethod
    def squash_box_decoder(graph):
        def validate_node(nodes_tuple):
            nms_node_ = nodes_tuple[0]
            nms_input_names_ = nms_node_.input_names
            if op_adapter.ReshapeOp.TRANSLATION_KEY == graph.get_producer_op(nms_input_names_[0]).type:
                # remove optional reshape input to check if previous is box decoder(ssd) below
                reshape_node_ = graph.get_producer_node(nms_node_.input_names[0])
                nms_input_names_ = [nms_input_names_[1], *reshape_node_.input_names]

            if any(op_adapter.SsdOp.TRANSLATION_KEY == graph.get_producer_node(name_).op.TRANSLATION_KEY
                   for name_ in nms_input_names_):
                return True

            return False

        sequence = [
                    ("non_max_suppression",
                        (),
                        ()
                     )
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)

        for node_tuple in matched_node_list:
            nms_node = node_tuple[0]
            nms_op = nms_node.op
            # update the boxes input of nms to be box decoder's inputs along with box decoder's op attributes.
            #  [boxes]_______[anchor or priorboxes]
            #            |
            #       [box_decoder(ssd_op)]   <- remove
            #                  |
            #        remove->([Reshape] (optional))_______[scores]
            #                                         |
            #                                 [non_max_suppression]
            # Updated input for nms will be: [scores, boxes, anchor(priorboxes)]

            nms_boxes_input_name, nms_scores_input_name = nms_node.input_names
            if op_adapter.ReshapeOp.TRANSLATION_KEY == graph.get_producer_op(nms_boxes_input_name).type:
                # update inputs for nms and subsequently the boxes_node
                reshape_node = graph.get_producer_node(nms_boxes_input_name)
                reshape_buf = graph.get_buffer(nms_boxes_input_name)
                nms_boxes_input_name = reshape_node.input_names[0]

                # update consumer relation with reshape buf and prune if applicable
                reshape_buf.consumers.remove(nms_node)
                if len(reshape_buf.consumers) == 0:
                    graph.prune(reshape_node)

            # fold box_decoder(ssd) node
            box_decoder_node = graph.get_producer_node(nms_boxes_input_name)
            box_decoder_buf = graph.get_buffer(nms_boxes_input_name)
            # Copy over input_names and all op attrs to nms op
            nms_node.input_names = [nms_scores_input_name, *box_decoder_node.input_names]
            for key in box_decoder_node.op.attrs:
                # New attributes must be added to Op instances using assertattr or add_attr - else, values will be
                # added as member variables of the Op instance
                nms_op.assertattr(key, box_decoder_node.op.attrs)

            # update consumer relation with nms node, box_decoder node and input to box_decoder and
            # prune if applicable
            for name in box_decoder_node.input_names:
                buf = graph.get_buffer(name)
                buf.consumers.add(nms_node)
            if nms_node in box_decoder_buf.consumers:
                box_decoder_buf.consumers.remove(nms_node)
            if len(box_decoder_buf.consumers) == 0:
                graph.prune(box_decoder_node)

            # Update Anchors inputs to fit DetectionOut spec
            anchor_buf = graph.get_buffer(nms_node.input_names[-1])
            anchor_data = anchor_buf.producer.op.tensor

            # TF style (decodeBox+nms) comes as CORNER_SIZE spec requires CENTER_SIZE
            for batch in range(0, anchor_buf.shape[0]):
                for i in range(0, anchor_buf.shape[1]):
                    y_min, x_min, y_max, x_max = anchor_data[batch][i]
                    height = (y_max - y_min)
                    width = (x_max - x_min)
                    anchor_data[batch][i][0] = y_min + height / 2.  # center_y
                    anchor_data[batch][i][1] = x_min + width / 2.  # center_x
                    anchor_data[batch][i][2] = height  # height
                    anchor_data[batch][i][3] = width

            # Addition of a const tensor to class labels should not be quantized
            classes_buf = graph.get_buffer(nms_node.output_names[2])
            for consumer in classes_buf.consumers:
                if consumer.op.type == op_adapter.ElementwiseSumOp.TRANSLATION_KEY:
                    for input_name in consumer.input_names:
                        add_input_node = graph.get_producer_node(input_name)
                        if add_input_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                            add_input_node.op.quantizable = False

            # change shape for anchor input from [batch, num_anchors, 4] to [batch * num_anchors, 4] per spec
            anchor_buf.shape = [anchor_buf.shape[0] * anchor_buf.shape[1], anchor_buf.shape[2]]
            anchor_buf.producer.op.tensor = anchor_data.reshape(anchor_buf.shape)

            log_debug2(code_to_message.get_debugging_message("DEBUG_BOXDECODER_SQUASH")(box_decoder_node.op.name,
                                                                                        nms_node.op.name))


@register_layer_optimization
class OptimizeTileTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.TileOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.eltwise_to_spatial_first_order(node, graph)
        input_buf = graph.get_buffer(node.input_names[0])
        if input_buf.axis_format in format_to_permute_order:
            node.op.multiples = graph.src_axis_order.permute_shape_to_ir(node.op.multiples)


@register_layer_optimization
class OptimizeTopKTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.TopKOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeTopKTranslation, self).axes_to_spatial_first_order(node, graph)

        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format in format_to_permute_order:
            axis_map = format_to_permute_order[input_buf.axis_format]
            node.op.axis = axis_map[node.op.axis]


@register_layer_optimization
class OptimizeUnpackTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.UnpackOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        # Check if input is a permute, if so this means the source framework deliberately added the permute
        # and we do not want to inject another one.
        if input_buf.producer.op.type != op_adapter.PermuteOp.TRANSLATION_KEY:
            if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
                graph.inject_implicit_permute(input_name, node.op.name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
                graph.inject_implicit_permute(input_name, node.op.name, AxisTracker.AxisFormat.TBF,
                                              AxisTracker.AxisFormat.BTF_TO_TBF, [node.op.name])

        output_buf = graph.get_output_buffers(node)[0]
        output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL


@register_layer_optimization
class OptimizeUpsampleTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.Upsample.TRANSLATION_KEY
