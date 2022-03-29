# ==============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import re
import sys

# tvm, relay
try:
    import tvm
    from tvm import relay
except ModuleNotFoundError as e:
    print("Error while importing Relay...\n")
    raise e
except ImportError as e:
    print("TVM not found in PYTHONPATH. Ensure PYTHONPATH includes <path/to/tvm>/python.\n"
          "You can download and install TVM from https://tvm.apache.org/docs/install/from_source.html\n")
    sys.exit(1)
except Exception as e:
    print("Error while importing TVM...\n")
    raise e

from qti.aisw.converters.common.converter_base import ConverterFrontend
from qti.aisw.converters.common.converter_ir.op_adapter import Op
from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph
from qti.aisw.converters.common.utils.converter_utils import (
    log_error,
    log_debug1,
    log_debug2,
    log_verbose,
    log_warning,
    converter_type
)

from qti.aisw.converters.relay.importers.relay_importer import RelayImporter
from .translations import RelayTranslations

global QUIR_GRAPH
global RELAY_PARAMS
global EXPR_TO_OP_OUT_NAMES
global CONVERTER_CTX

def get_key_from_expr(expr: relay.expr):
    return hash(expr)

def get_translation(expr):
    op_type = str(expr.op.name).split(('.'))[-1]
    log_debug2("op_type in get_translation {}", op_type)
    if converter_type(op_type, 'relay') in RelayTranslations.translations:
        return RelayTranslations.translations[converter_type(op_type, 'relay')]
    else:
        raise TypeError("Unsupported Op type {}".format(expr.op.name))


class RelayConverterContext:
    """
    Class that contains all data structures and methods for Op Conversion
    """
    def __init__(self, quir_graph: IROpGraph, output_names_dict: dict=None):
        self.output_names_provided_at_init = True
        if output_names_dict is None:
            output_names_dict = {}
            self.output_names_provided_at_init = False
        self.expr_to_name_dict = {}
        self.expr_to_out_names_dict = output_names_dict
        if self.expr_to_out_names_dict:
            log_verbose("Output Names in expr_to_output_names Dict:")
            for _, (expr, out_names) in self.expr_to_out_names_dict.items():
                log_verbose("\t{}", out_names)
            log_verbose("\n")
        self.type_count = {}
        self.quir_graph = quir_graph

    def get_op_name(self, expr: relay.expr, op_type: str):
        """
        Generates Op name that is unique using ref count per Op Type
        :param expr: Relay Expr
        :param op_type: QuIR Op Type
        :return: Str
        """
        count = self.type_count.get(op_type, 0)
        self.type_count[op_type] = count + 1
        op_name = "%s_%d" % (op_type, count)
        self.expr_to_name_dict[expr] = op_name
        log_verbose("op_name {}", op_name)
        return op_name

    def get_input_names(self, expr: relay.expr):
        """
        Get Input Names for input Relay Expr. It uses recursive tree traversal to get output names of
        inputs to the Input Expr
        :param expr: Relay Expr
        :return: List of input names
        """
        inputs = []
        if isinstance(expr, relay.Call) or isinstance(expr, relay.expr.Call):
            for arg in expr.args:
                outputs = self.get_output_names(arg)
                log_verbose("Call outputs {}", outputs)
                inputs.extend(outputs)
        elif isinstance(expr, relay.expr.Var) or \
                isinstance(expr, relay.Var):
            k = get_key_from_expr(expr)
            if k in self.expr_to_out_names_dict:
                log_verbose("Var name {} outputs {}", expr.name_hint, self.expr_to_out_names_dict[k][1])
                inputs.append(self.expr_to_out_names_dict[k][1])
            else:
                raise KeyError("Expr for {} not found in dictionary EXPR_TO_OP_OUT_NAMES".format(expr))
        elif isinstance(expr, relay.expr.TupleGetItem) or \
                isinstance(expr, relay.TupleGetItem):
            log_verbose("tuple item input index {}", expr.index)
            tuple_inputs = self.get_output_names(expr.tuple_value)[expr.index]
            log_verbose("Appending tuple item input {}", tuple_inputs)
            inputs.extend(tuple_inputs)
        elif isinstance(expr, relay.expr.Tuple) or \
                isinstance(expr, relay.Tuple):
            for elem in expr.fields:
                log_verbose("inputs before Tuple {}", inputs)
                inputs.extend(self.get_output_names(elem))
                log_verbose("inputs after Tuple {}", inputs)
        else:
            raise TypeError("Unsupported Expr type {} for get_input_names".format(type(expr)))

        return inputs

    def get_input_shapes(self, expr: relay.expr):
        """
        Get Buffer Shapes from QuIR Graph for inputs to the Relay expr
        :param expr: Relay Expr
        :return: List of input shapes
        """
        inputs = self.get_input_names(expr)
        input_shapes = []
        for input_name in inputs:
            try:
                input_shapes.append(self.quir_graph.get_buffer(input_name).shape)
            except KeyError as e:
                log_error("input_name {} is not found in graph buffers", input_name)
                raise e
        log_verbose("input_shapes {}", *zip(inputs, input_shapes))
        return input_shapes

    def get_output_names(self, expr: relay.expr, num_outputs: int=None):
        """
        Get output names of given Relay Expr
        :param expr: Relay Expr
        :param num_outputs:
        :return:
        """

        key = get_key_from_expr(expr)

        if key in self.expr_to_out_names_dict:
            outputs = self.expr_to_out_names_dict[key][1] if isinstance(self.expr_to_out_names_dict[key][1], list) else \
                [self.expr_to_out_names_dict[key][1]]
        elif isinstance(expr, relay.expr.Var) or \
                isinstance(expr, relay.Var):
            if key in self.expr_to_out_names_dict:
                log_verbose("Var name {} outputs {}", expr.name_hint, self.expr_to_out_names_dict[key][1])
                outputs=[self.expr_to_out_names_dict[key][1]]
            else:
                log_verbose("Var name {}", expr.name_hint)
                outputs = [expr.name_hint]
                self.expr_to_out_names_dict[key] =(expr, outputs)
        elif isinstance(expr, relay.expr.Tuple) or \
                isinstance(expr, relay.Tuple):
            outputs = []
            for elem in expr.fields:
                log_verbose("tuple outputs before {}", outputs)
                outputs.extend(self.get_output_names(elem))
                log_verbose("tuple outputs after {}", outputs)
        elif isinstance(expr, relay.expr.TupleGetItem) or \
                isinstance(expr, relay.TupleGetItem):
            outputs = [self.get_output_names(expr.tuple_value)[expr.index]]
            log_verbose("Appending tuple item output {}", outputs)
        else:
            # expr is not in self.expr_to_out_names_dict
            if num_outputs:
                outputs = self.generate_output_names(expr, num_outputs)
                self.expr_to_out_names_dict[key] = (expr, outputs)
            else:
                log_error("Unknown expr:\n{}\ntype {}\n", expr, type(expr))
                raise KeyError("Unknown Expr found while getting output names")

        return outputs

    def generate_output_names(self, expr: relay.expr, num_outputs: int):
        """
        Generate output tensor names for given Relay Expr since they were not already provided
        :param expr: Relay Expr
        :param num_outputs:
        :return:
        """
        k = get_key_from_expr(expr)
        if k not in self.expr_to_out_names_dict:
            output_names = [self.expr_to_name_dict[expr] + '_' +
                            str(i) for i in range(num_outputs)]
            log_verbose("generated output names {}", output_names)
        return output_names

    def add_op_to_graph(self,
                        expr: relay.expr,
                        op: Op,
                        input_names: list,
                        output_names: list,
                        axis_formats: list=None,
                        idx: int=-1):
        """
        Add QuIR Op to QuIR OpGraph and update the dictionary of expr to output names
        :param expr: Relay Expr
        :param op: QuIR Op
        :param input_names: List of input names
        :param output_names: List of output names
        :param axis_formats:
        :param idx: Index in graph to insert the Node
        :return: QuIR OpNode
        """
        key = get_key_from_expr(expr)
        if key not in self.expr_to_out_names_dict:
            self.expr_to_out_names_dict[key] = (expr, output_names)
        return QUIR_GRAPH.add(op, input_names, output_names, axis_formats, idx)


class RelayConverterFrontend(ConverterFrontend):
    class ArgParser(ConverterFrontend.ArgParser):
        def __init__(self, **kwargs):
            super(RelayConverterFrontend.ArgParser, self).__init__(**kwargs)
            self.add_optional_argument('--dump_relay', type=str, default=None,
                                       help="Dump Relay ASM and Params at the path provided with the argument\n"
                                            "Usage: --dump_relay <path_to_dump>")

    def __init__(self, args, importer: RelayImporter=None, mod: tvm.IRModule=None, params: dict=None, **kwargs):
        super(RelayConverterFrontend, self).__init__(args,
                                                     **kwargs)
        self.importer = importer
        if self.importer and isinstance(self.importer, RelayImporter):
            self.relay_mod, self.relay_params, self.expr_to_out_names_dict =\
                self.importer.convert_to_relay(self.input_model_path)
            self.preprocess_out_names_dict()
        else:
            mod = mod
            params = params
            if not mod or not params:
                raise SyntaxError("{} should be initialized with either an importer or with (mod, params). "
                                  "None of these provided.".format(self.__class__.__name__))
            self.expr_to_out_names_dict = {}
            self.relay_mod = mod
            self.relay_params = params

        if args.dump_relay:
            self.dump_relay_data(args)
        self.converter_context = RelayConverterContext(self.graph,
                                                       output_names_dict=self.expr_to_out_names_dict)
        self._init_globals()

    def dump_relay_data(self, args):
        ########## debugging ###########
        import os
        if not args.dump_relay:
            path = '/'.join(os.path.realpath(args.input_network).split('/')[:-1])
        else:
            path = args.dump_relay

        log_verbose("Dumping Relay data at {}", path)

        full_mod_txt_path = os.path.join(path, "mod.txt")
        full_mod_json_path = os.path.join(path, "mod.json")
        self.dump_mod(full_mod_txt_path, full_mod_json_path)

        full_params_path = os.path.join(path, "params.txt")
        self.dump_params(full_params_path)
        ########## end debugging ###########

    def _init_globals(self):
        global QUIR_GRAPH
        QUIR_GRAPH = self.graph

        global RELAY_PARAMS
        RELAY_PARAMS = self.relay_params

        global EXPR_TO_OP_OUT_NAMES
        EXPR_TO_OP_OUT_NAMES = self.expr_to_out_names_dict

        global CONVERTER_CTX
        CONVERTER_CTX = self.converter_context

    def dump_params(self, file_name):
        with open(file_name, "w") as f:
            for k, v in self.relay_params.items():
                f.write(k)
                f.write(':')
                f.write(str(v))
                f.write('\n')

    def dump_mod(self, mod_txt_path, mod_json_path):
        with open(mod_txt_path, "w") as f:
            f.write(self.relay_mod.astext(show_meta_data=False))

        with open(mod_json_path, "w") as f:
            f.write(tvm.ir.save_json(self.relay_mod))

    def preprocess_out_names_dict(self):
        def preprocess_expr(expr, output_name, tuples_dict):
            if isinstance(expr, relay.TupleGetItem) or \
                    isinstance(expr, relay.expr.TupleGetItem):
                new_expr = expr.tuple_value
                if isinstance(new_expr, relay.expr.Call):
                    k = get_key_from_expr(new_expr)
                    log_verbose("k:\n{}\n", new_expr)
                    if k not in tuples_dict:
                        tuples_dict[k] = (new_expr, [])
                    if len(tuples_dict[k][1]) <= expr.index:
                        for i in range(len(tuples_dict[k][1]), (int(expr.index)+1)):
                            tuples_dict[k][1].append("")
                    tuples_dict[k][1][expr.index] = output_name
                    log_verbose("v:\n{}", tuples_dict[k][1])
                elif isinstance(new_expr, relay.expr.Tuple) or \
                        isinstance(new_expr, relay.Tuple):
                    new_expr = new_expr.fields[expr.index]
                    if not isinstance(new_expr, relay.expr.Call):
                        raise RecursionError("Error processing expr:\n{} ",format(expr))
                    k = get_key_from_expr(new_expr)
                    tuples_dict[k] = (new_expr, output_name)
                    log_verbose("k:\n{}\n", new_expr)
                    log_verbose("v:\n{}\n", output_name)
            elif isinstance(expr, relay.expr.Tuple) or \
                     isinstance(expr, relay.Tuple):
                outputs = []
                for elem in expr.fields:
                    log_verbose("tuple outputs before {}", outputs)
                    outputs.extend(self.converter_context.get_output_names(elem))
                    log_verbose("tuple outputs after {}", outputs)
                k = get_key_from_expr(expr)
                tuples_dict[k] = (expr, outputs)
            else:
                if not isinstance(expr, relay.expr.Call) and not isinstance(expr, relay.expr.Var):
                    log_error("\nUnknown expr type {}\noutput_name: {}\n", type(expr), output_name)
                    raise TypeError("Unsupported type {}".format(type(expr)))

            return tuples_dict

        tuples_dict = {}
        log_verbose("\n\nPreprocessing Dict...\n")

        for _, (expr, output_name) in self.expr_to_out_names_dict.items():
            tuples_dict = preprocess_expr(expr, output_name, tuples_dict)

        self.expr_to_out_names_dict.update(tuples_dict)
        log_verbose("\nDone Preprocessing\n\n")

    @staticmethod
    def add_input(expr: relay.expr):
        if not isinstance(expr, relay.expr.Var):
            return

        global QUIR_GRAPH
        global RELAY_PARAMS
        global EXPR_TO_OP_OUT_NAMES

        var_name = str(expr).split("\n")[1].lstrip("%v")

        k = get_key_from_expr(expr)

        if var_name in RELAY_PARAMS:
            if k not in EXPR_TO_OP_OUT_NAMES:
                EXPR_TO_OP_OUT_NAMES[k] = (expr, [var_name])
            param = RELAY_PARAMS[var_name]
            log_verbose("param {}", var_name)
            log_verbose("shape {}", param.shape)
        else:
            log_verbose("input {}", var_name)
            log_verbose("type {}", type(expr))
            log_verbose('shape {}', list(expr.type_annotation.shape))
            input_shape = [int(val) for val in expr.type_annotation.shape]
            QUIR_GRAPH.add_input(var_name, input_shape)
            if k not in EXPR_TO_OP_OUT_NAMES:
                EXPR_TO_OP_OUT_NAMES[k] = (expr, [var_name])

    @staticmethod
    def add_constant(expr: relay.expr):
        if not isinstance(expr, relay.expr.Constant):
            return

        global QUIR_GRAPH
        global RELAY_PARAMS
        global EXPR_TO_OP_OUT_NAMES
        global CONVERTER_CTX

        key = get_key_from_expr(expr)

        if key not in EXPR_TO_OP_OUT_NAMES:
            constant_name = CONVERTER_CTX.get_op_name(expr, 'relay_constant')
            constant_array = expr.data
            # Add const array to param
            RELAY_PARAMS[constant_name] = constant_array
            EXPR_TO_OP_OUT_NAMES[key] = (expr, [constant_name])

    @staticmethod
    def add_op(expr: relay.expr):
        if isinstance(expr, relay.expr.Call):
            # op_name = str(expr.op.name).replace("nn.", "")
            # log_verbose("name {}", expr.op.name)
            log_debug1("")
            log_debug1("Relay Op name {}", expr.op)

            ##### DEBUG PRINTS #####
            attributes = {}
            if expr.attrs:
                for attr in expr.attrs.list_field_info():
                    attributes[attr.name] = {}
                    attributes[attr.name]['value'] = getattr(expr.attrs, attr.name)
            log_verbose("attributes:")
            for k, v in attributes.items():
                log_verbose("\t{}:{}", k, v)
            ##### END DEBUG #####

            translation = get_translation(expr)

            global CONVERTER_CTX
            global RELAY_PARAMS
            translation.add_op(expr, QUIR_GRAPH, converter_context=CONVERTER_CTX, relay_params=RELAY_PARAMS)
        else:
            pass

    @staticmethod
    def visit_module(expr: relay.expr):
        log_debug2("")
        log_debug2("##### NEW OP Translation #####")
        if isinstance(expr, relay.expr.Var):
            RelayConverterFrontend.add_input(expr)
        elif isinstance(expr, relay.expr.Constant):
            RelayConverterFrontend.add_constant(expr)
        elif isinstance(expr, relay.expr.Call):
            RelayConverterFrontend.add_op(expr)
        else:
            log_verbose("{}", type(expr))

        log_debug2("\n")

    def convert_to_ir(self):
        relay.analysis.post_order_visit(self.relay_mod["main"], RelayConverterFrontend.visit_module)
        self.graph.eval_macs_params()
        return self.graph

    def convert(self):
        # Wrapper for combination of convert_to_relay and convert_to_ir
        return self.convert_to_ir()
