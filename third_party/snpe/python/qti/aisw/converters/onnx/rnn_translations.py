# ==============================================================================
#
#  Copyright (c) 2018-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np

from .onnx_translations import *
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrders, AxisTracker
from qti.aisw.converters.common.utils import code_to_message

# ------------------------------------------------------------------------------
#  RNNTranslationBase
# ------------------------------------------------------------------------------
OPTIONAL_INPUTS = NamedDict(initial_c='', initial_h='')
RNN_INPUT_TYPES = ('_initial_h', '_initial_c')
RNN_OUTPUT_TYPES = ('_all_hidden', '_final_hidden', '_final_cell')


class OnnxRnnTranslationsBase(OnnxTranslationBase, metaclass=ABCMeta):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.input_names = []
        self.weights = []
        self.rec_weights = []
        self.bias = None
        self.params = NamedDict()
        self.no_of_gates = 1
        self.backward = False
        self.output_names = []
        self.num_directions = 1
        self.timesteps = []

    def extract_params_for_type(self, src_op, graph):
        self.input_names = list(map(str, src_op.input))
        self.output_names = self.extract_output_names(src_op, graph)
        self.params.direction = str(self.params.direction).lower()
        self.backward = False if self.params.direction == 'forward' else True
        self.num_directions = 1

        # Ensure weights or rec_weights are not passed in dynamically
        weights_input_name = self.input_names[1]
        if graph.weights.has(weights_input_name) and not graph.has_buffer(weights_input_name):
            self.weights = graph.weights.fetch(weights_input_name)
        else:
            raise ValueError("Unsupported dynamic weights for input {} of RNN node {}".format(weights_input_name,
                                                                                              src_op.name))

        rec_weights_input_name = self.input_names[2]
        if graph.weights.has(rec_weights_input_name) and not graph.has_buffer(rec_weights_input_name):
            self.rec_weights = graph.weights.fetch(rec_weights_input_name)
        else:
            raise ValueError("Unsupported dynamic weights for input {} of RNN node {}".format(rec_weights_input_name,
                                                                                              src_op.name))

        # ONNX may use empty string as a placeholder
        # So add an and-condition to further check it.
        if len(self.input_names) >= 4 and self.input_names[3]:
            self.bias = graph.weights.fetch(self.input_names[3])

        if len(self.input_names) >= 5 and self.input_names[4]:
            self.timesteps = graph.weights.fetch(self.input_names[4])

        # If it is bi-directional, we include support for custom activations (although
        # not available in snpe as yet). Also check that weights and rec_weights
        # have the right shape
        if self.params.direction == "bidirectional":
            self.params.activations = list(map(op_adapter.NeuronOp.extract_activation,
                                               self.params.activations))
            log_assert(self.weights.shape[0] == 2 and self.rec_weights.shape[0] == 2,
                       "Node {}: Bidirectional input requires two sets of weights and recurrent "
                       "weights each. Got only {} set of weights",
                       src_op.name, self.weights.shape[0])
        else:
            # Limit the length of the user defined activations to the number of gates if
            # unidirectional
            self.params.activations = list(map(op_adapter.NeuronOp.extract_activation,
                                               self.params.activations[0:self.no_of_gates]))

    def convert_params_to_snpe(self, weights, rec_weights, bias, hidden_size):

        no_of_gates = self.no_of_gates

        if bias is None:
            bias = np.zeros((no_of_gates, hidden_size), dtype=numpy.float32)
        else:
            # for probably vendor specific reasons, ONNX defines GRU bias to
            # be separated into forward and recurrent parts, that are always
            # added together (unless linear_before_reset is false, but we
            # don't support that). So we will always combine.
            # We need to reshape bias which is in (2*no_of_gates*hidden_size)
            # into (2, no_of_gates * hidden_size).
            bias = np.reshape(bias, (2, no_of_gates * hidden_size))
            new_bias = np.empty((no_of_gates * hidden_size), dtype=numpy.float32)
            # Elements are stored in [weights, rec_weights] where each column
            # represents the gate and the number of rows is the hidden size
            np.add(bias[0, :], bias[1, :], out=new_bias[:])
            bias = new_bias.reshape(no_of_gates, hidden_size)

        # weights and rec_weights are also laid out as (no_of_gates*hidden_size, input_size)
        # and (no_of_gates*hidden_size, hidden_size)respectively. We need to reshape
        # to SNPE format depending on the rnn type.
        weights = np.reshape(weights, (no_of_gates, hidden_size, weights.shape[-1]))
        rec_weights = np.reshape(rec_weights, (no_of_gates, hidden_size, hidden_size))

        return weights, rec_weights, bias

    def extract_input_names(self, src_op, graph):

        # empty string for initial_h (initial hidden state) and initial_c (initial cell state)
        # means we don't need to add the input buffer to the ir_graph.
        formatted_input_names = [self.input_names[0], OPTIONAL_INPUTS.initial_h,
                                 OPTIONAL_INPUTS.initial_c]

        return [name for name in formatted_input_names if name is not '']

    def extract_output_names(self, src_op, graph):
        onnx_output_names = [output for i, output in enumerate(src_op.output) if output]
        onnx_output_names_len = len(onnx_output_names)
        if onnx_output_names_len < 3:
            return onnx_output_names
        else:
            # ONNX LSTM output order: Y, Y_h, Y_c
            # SNPE LSTM output order: h, c_T, h_T
            output_names = [onnx_output_names[0], onnx_output_names[2], onnx_output_names[1]]
            for i in range(3, onnx_output_names_len):
                output_names.append(onnx_output_names[i])
            return output_names

    def create_rnn(self, src_op, graph, create_unidirectional_func, create_bidirectional_func):

        if self.params.direction == "bidirectional":
            create_bidirectional_func(src_op, graph)
        else:
            # set up naming so that the buffers are all different and tagged correctly
            input_names = self.extract_input_names(src_op, graph)
            output_names = self.extract_output_names(src_op, graph)
            output_names_len = len(output_names)
            module_name = src_op.name if len(src_op.name) != 0 else \
                output_names[0] if output_names_len != 0 else \
                src_op.op_type

            # set up rnn ops
            rnn_op = create_unidirectional_func(name=module_name)

            # set up reshape ops
            reshape_ops = []
            input_shapes = [graph.get_buffer(src_op.input[0]).shape[:]]
            batch_size, time_steps, input_size = AxisOrders.ONNX.extract_time_series_dims(input_shapes[0])
            hidden_size = rnn_op.hidden_state_weights.shape[-1]
            output_shapes = []
            for i in range(0, output_names_len):
                if i == 0:
                    output_shapes.append([time_steps, self.num_directions, batch_size, hidden_size])
                else:
                    output_shapes.append([self.num_directions, batch_size, hidden_size])
            for i in range(0, output_names_len):
                # the outputs are reshaped to be ONNX format
                reshape_ops.append(op_adapter.ReshapeOp(output_names[i] + RNN_OUTPUT_TYPES[i] + '_reshape', output_shape=output_shapes[i]))

            rnn_output_names = [str(name) + RNN_OUTPUT_TYPES[i] + "_rnn"
                                for i, name in enumerate(output_names)]

            reshape_output_names = [str(name) for name in output_names]

            self.add_src_op_info(rnn_op.name, src_op, graph)
            return_op = graph.add(rnn_op, input_names, rnn_output_names)

            for i in reversed(range(0, output_names_len)):
                # add reshape for outputs to make shape to be ONNX's format
                log_debug("Adding reshape op {} while creating unidirectional RNN unit".format(reshape_ops[i].name))
                return_op = graph.add(reshape_ops[i], [rnn_output_names[i]], [reshape_output_names[i]])
                graph.add_src_op_info(reshape_ops[i].name, [rnn_output_names[i]], [reshape_output_names[i]])

            return return_op

    def create_bidirectional_module(self, src_op, graph, weights, rec_weights, bias, params,
                                    create_rnn_type):
        # set up naming so that the buffers are all different and tagged correctly
        src_op_inputs = list(src_op.input)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)
        module_name = src_op.name if len(src_op.name) != 0 else \
            output_names[0] if len(output_names) != 0 else \
            src_op.op_type

        src_op_inputs_len = len(src_op_inputs)
        if src_op_inputs_len < 7:
            raise ValueError("Unsupported number of inputs on source op {}. Expected >= 7, got {}".
                             format(src_op.name, src_op_inputs_len))

        input_names_len = len(input_names)
        if input_names_len > 3:
            raise ValueError("Unsupported number of inputs for bidirectional unit. Expected < 3, got {}".
                             format(input_names_len))

        output_names_len = len(output_names)
        if output_names_len > 3:
            raise ValueError("Unsupported number of outputs for bidirectional unit. Expected < 3, got {}".
                             format(output_names_len))

        # set up split ops
        split_ops = []
        for i in range(1, input_names_len):
            split_ops.append(op_adapter.SliceOp(str(input_names[i]) + RNN_INPUT_TYPES[i-1] + "_slice", axis=0, slice_points=[1]))

        # set up forward op
        forward_op = create_rnn_type(module_name + '_forward',
                                     weights=weights[0, :, :],
                                     rec_weights=rec_weights[0, :, :],
                                     bias=bias[0, :] if bias.all() else None,
                                     hidden_size=params.hidden_size,
                                     backward=False,
                                     h_0_input_name=str(input_names[1]) + RNN_INPUT_TYPES[0] + "_forward_split" if
                                         input_names_len > 1 else OPTIONAL_INPUTS.initial_h,
                                     c_0_input_name=str(input_names[2]) + RNN_INPUT_TYPES[1] + "_forward_split" if
                                         input_names_len > 2 else OPTIONAL_INPUTS.initial_c)

        # set up backward op
        backward_op = create_rnn_type(module_name + '_backward',
                                      weights=weights[1, :, :],
                                      rec_weights=rec_weights[1, :, :],
                                      bias=bias[1, :] if bias.all() else None,
                                      hidden_size=params.hidden_size,
                                      backward=True,
                                      h_0_input_name=str(input_names[1]) + RNN_INPUT_TYPES[0] + "_backward_split" if
                                          input_names_len > 1 else OPTIONAL_INPUTS.initial_h,
                                      c_0_input_name=str(input_names[2]) + RNN_INPUT_TYPES[1] + "_backward_split" if
                                          input_names_len > 2 else OPTIONAL_INPUTS.initial_c)

        # set up reshape ops
        reshape_ops = []
        input_shapes = [graph.get_buffer(src_op.input[0]).shape[:]]
        batch_size, time_steps, input_size = AxisOrders.ONNX.extract_time_series_dims(input_shapes[0])
        hidden_size = forward_op.hidden_state_weights.shape[-1]
        output_shapes = []
        for i in range(0, output_names_len):
            if i == 0:
                output_shapes.append([time_steps, self.num_directions, batch_size, hidden_size])
            else:
                output_shapes.append([self.num_directions, batch_size, hidden_size])
        for i in range(0, output_names_len):
            # the outputs are reshaped to be ONNX format
            reshape_ops.append([op_adapter.ReshapeOp(output_names[i] + RNN_OUTPUT_TYPES[i] + '_reshape_backward', output_shape=output_shapes[i]),
                                op_adapter.ReshapeOp(output_names[i] + RNN_OUTPUT_TYPES[i] + '_reshape_forward', output_shape=output_shapes[i])])

        # set up concat ops
        concat_ops = []
        for i in range(0, output_names_len):
            # The first output is used to concat all hidden output values at last axis
            axis = 1 if i == 0 else 0
            concat_ops.append(op_adapter.ConcatOp(output_names[i] + '_concat', axis=axis))

        for input_name in input_names:
            # Add constant op if one of the inputs is Initializer and not added to graph
            if graph.weights.has(input_name) and not graph.has_buffer(input_name):
                tensor = graph.weights.fetch(input_name, prunable=False)
                graph.add(op_adapter.ConstantOp(input_name, tensor), [], input_name)

        split_output_names = [[str(input_names[i]) + RNN_INPUT_TYPES[i-1] + "_backward_split",
                               str(input_names[i]) + RNN_INPUT_TYPES[i-1] + "_forward_split"]
                              for i in range(1, input_names_len)]
        backward_output_names = [str(name) + RNN_OUTPUT_TYPES[i] + "_backward" for i, name in
                                 enumerate(output_names)]
        forward_output_names = [str(name) + RNN_OUTPUT_TYPES[i] + "_forward" for i, name in
                                enumerate(output_names)]
        reshape_output_names = [[str(name) + RNN_OUTPUT_TYPES[i] + "_reshape_backward",
                                 str(name) + RNN_OUTPUT_TYPES[i] + "_reshape_forward"]
                                for i, name in enumerate(output_names)]
        concat_output_names = [str(name) for name in output_names]

        # add split ops to the graph, each split op should have two outputs, one going into each RNN
        # inputs[i] where i >= 1 must be split along num_directions dimension in ONNX
        split_src_op_inputs = src_op_inputs[5:]
        split_input_names = input_names[1:]

        if len(split_src_op_inputs) < len(split_input_names):
            raise ValueError("Source op {} requires initial_h and initial_c inputs.".format(src_op.name))

        for i in range(0, len(split_input_names)):
            log_debug("Adding split op {} while creating bidirectional RNN unit".format(split_ops[i].name))
            graph.add(split_ops[i], [split_input_names[i]], split_output_names[i])
            graph.add_src_op_info(split_ops[i].name, split_src_op_inputs[i], split_output_names[i])

        # Modify input names to be different according to split
        backward_input_names = [input_names[0]] + [split_output_names[i][0] for i in range(len(split_input_names))]
        forward_input_names = [input_names[0]] + [split_output_names[i][1] for i in range(len(split_input_names))]

        log_debug("Adding backward RNN op {} while creating bidirectional RNN unit".format(backward_op.name))
        graph.add(backward_op, backward_input_names, backward_output_names)
        graph.add_src_op_info(backward_op.name, backward_input_names, backward_output_names)

        log_debug("Adding forward RNN op {} while creating bidirectional RNN unit".format(forward_op.name))
        graph.add(forward_op, forward_input_names, forward_output_names)
        graph.add_src_op_info(forward_op.name, forward_input_names, forward_output_names)

        # add concat op to the graph, should end up as a child of both forward and backward ops.
        # we need more than one concat node, depending on the number of outputs.
        for i in range(0, output_names_len):
            # add reshape for outputs to make shape to be ONNX's format
            log_debug("Adding reshape op {} and {} while creating bidirectional RNN unit".format(reshape_ops[i][0].name, reshape_ops[i][1].name))
            graph.add(reshape_ops[i][0], [backward_output_names[i]], [reshape_output_names[i][0]])
            graph.add(reshape_ops[i][1], [forward_output_names[i]], [reshape_output_names[i][1]])
            graph.add_src_op_info(reshape_ops[i][0].name, [backward_output_names[i]], [reshape_output_names[i][0]])
            graph.add_src_op_info(reshape_ops[i][1].name, [forward_output_names[i]], [reshape_output_names[i][1]])

            log_debug("Adding concat op {} while creating bidirectional RNN unit".format(concat_ops[i].name))
            graph.add(concat_ops[i], [reshape_output_names[i][1], reshape_output_names[i][0]], [concat_output_names[i]])
            graph.add_src_op_info(concat_ops[i].name, [reshape_output_names[i][1], reshape_output_names[i][0]], [concat_output_names[i]])


# ------------------------------------------------------------------------------
#  GRU
# ------------------------------------------------------------------------------

class OnnxGruTranslation(OnnxRnnTranslationsBase):
    def __init__(self):
        OnnxRnnTranslationsBase.__init__(self)
        schema_dict = self.register_op_schema('GRU', [1, 7], [['clip',
                                                               'activation_alpha',
                                                               'activation_beta',
                                                               'output_sequence']])
        schema_dict.replace_default_values(activations=['Sigmoid', 'Sigmoid', 'Tanh'] * 2)
        schema_dict.register_method(self.validate_attribute_values)
        self.no_of_gates = 3

    def extract_parameters(self, src_op, graph):
        self.params = extract_attributes(src_op, schema=self.op_schema())

        self.extract_params_for_type(src_op, graph)

        # gru output is only for Y
        log_assert(self.output_names == 1,
                   "Node {}: Only the first output, Y, of the Onnx GRU Op is supported",
                   src_op.name)

        if len(self.input_names) >= 6:
            # check if initial_h is included
            OPTIONAL_INPUTS.initial_h = self.input_names[5]

    def add_op(self, src_op, graph):
        self.extract_parameters(src_op, graph)
        return self.create_rnn(src_op, graph, self.create_unidirectional_gru,
                               self.create_bidirectional_gru)

    def create_unidirectional_gru(self, name='gru', **kargs):

        if kargs:
            [weights, rec_weights, bias] = self.convert_params_to_snpe(kargs['weights'],
                                                                       kargs['rec_weights'],
                                                                       kargs['bias'],
                                                                       kargs['hidden_size'])
        else:
            [weights, rec_weights, bias] = self.convert_params_to_snpe(self.weights,
                                                                       self.rec_weights,
                                                                       self.bias,
                                                                       self.params.hidden_size)
        # gru specific organization into separate gates
        activations = self.params.activations
        control_gate = {'weights': weights[0, :, :].T,
                        'rec_weights': rec_weights[0, :, :].T,
                        'bias': bias[0, :]}
        forget_gate = {'weights': weights[1, :, :].T,
                       'rec_weights': rec_weights[1, :, :].T,
                       'bias': bias[1, :]}
        state_gate = {'weights': weights[2, :, :].T,
                      'rec_weights': rec_weights[2, :, :].T,
                      'bias': bias[2, :]}

        return op_adapter.GruOp(name,
                                state_gate,
                                forget_gate,
                                control_gate,
                                activation=activations[0],
                                gate_activation=activations[1],
                                rec_gate_activation=activations[2],
                                h_0_input_name=OPTIONAL_INPUTS.initial_h,
                                backward=self.backward if not kargs else kargs['backward'])

    def create_bidirectional_gru(self, src_op, graph):
        return self.create_bidirectional_module(src_op, graph, self.weights, self.rec_weights,
                                                self.bias,
                                                self.params, self.create_unidirectional_gru)

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'activations' or attr_name == 'linear_before_reset':
            OpSchemaBase.validate_attribute_values(src_op, attr_name, attr_value)


OnnxTranslations.register_translation(OnnxGruTranslation(),
                                      converter_type('GRU', 'onnx'),
                                      op_adapter.GruOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#  LSTM
# ------------------------------------------------------------------------------
class OnnxLSTMTranslation(OnnxRnnTranslationsBase):
    def __init__(self):
        OnnxRnnTranslationsBase.__init__(self)
        schema_dict = self.register_op_schema('LSTM', [1, 7], [['clip',
                                                                'activation_alpha',
                                                                'activation_beta',
                                                                'output_sequence']])
        schema_dict.replace_default_values(
            activations=['Sigmoid', 'Sigmoid', 'Sigmoid', 'Tanh'] * 2)
        schema_dict.register_method(self.validate_attribute_values)
        self.no_of_gates = 4
        self.peephole_weights = []

    def extract_parameters(self, src_op, graph):
        self.params = extract_attributes(src_op, schema=self.op_schema(), validate=True)

        # set parameters
        self.extract_params_for_type(src_op, graph)

        if len(self.input_names) >= 7:
            # check if initial_h and initial_c are included
            # snpe requires that if they are included, then all
            # 3 outputs will be returned.
            OPTIONAL_INPUTS.initial_c = self.input_names[6]
            OPTIONAL_INPUTS.initial_h = self.input_names[5]

    def add_op(self, src_op, graph):
        self.extract_parameters(src_op, graph)
        self.add_src_op_info(src_op.name, src_op, graph)
        return self.create_rnn(src_op, graph, self.create_unidirectional_lstm,
                               self.create_bidirectional_lstm)

    def create_unidirectional_lstm(self, name='lstm', **kargs):

        if kargs:
            [gate_weights, gate_rec_weights, gate_bias] = self.convert_params_to_snpe(
                kargs['weights'],
                kargs['rec_weights'],
                kargs['bias'],
                kargs['hidden_size'])
            h_0_input_name = kargs['h_0_input_name']
            c_0_input_name = kargs['c_0_input_name']
        else:
            [gate_weights, gate_rec_weights, gate_bias] = self.convert_params_to_snpe(self.weights,
                                                                                      self.rec_weights,
                                                                                      self.bias,
                                                                                      self.params.hidden_size)
            h_0_input_name = OPTIONAL_INPUTS.initial_h
            c_0_input_name = OPTIONAL_INPUTS.initial_c

        # transform from iofg to ifog
        if self.no_of_gates == 4:
            hidden_size = int(gate_bias.size / self.no_of_gates)

            new_gate_bias = np.empty(gate_bias.shape, dtype=np.float32)
            for new, old in enumerate([0, 2, 1, 3]):
                new_gate_bias[new, :] = gate_bias[old, :]
            gate_bias = new_gate_bias

            new_gate_rec_weights = np.empty(gate_rec_weights.shape, dtype=np.float32)
            for new, old in enumerate([0, 2, 1, 3]):
                new_gate_rec_weights[new, :, :] = gate_rec_weights[old, :, :]
            gate_rec_weights = new_gate_rec_weights

            new_gate_weights = np.empty(gate_weights.shape, dtype=np.float32)
            for new, old in enumerate([0, 2, 1, 3]):
                new_gate_weights[new, :, :] = gate_weights[old, :, :]
            gate_weights = new_gate_weights

        # LSTM specific organization into gate format
        gate_bias = gate_bias.reshape(-1, )
        gate_rec_weights = gate_rec_weights.reshape(self.no_of_gates * self.params.hidden_size, -1)
        gate_weights = gate_weights.reshape(self.no_of_gates * self.params.hidden_size, -1)

        return op_adapter.LstmOp(name,
                                 input_weights=gate_weights,
                                 hidden_state_weights=gate_rec_weights,
                                 gate_bias=gate_bias,
                                 backward=self.backward if not kargs else kargs['backward'],
                                 c_0_input_name=c_0_input_name if
                                     len(self.input_names) >= 7 and len(self.output_names) == 3 else '',
                                 h_0_input_name=h_0_input_name if
                                     len(self.input_names) >= 7 and len(self.output_names) == 3 else '',
                                 # if c_0 and h_0 exist, reset_state_at_time_step_0 will be False
                                 reset_state_at_time_step_0=True if
                                     len(self.input_names) < 7 or len(self.output_names) < 3 else False,
                                 hidden_size=self.params.hidden_size)

    def create_bidirectional_lstm(self, src_op, graph):
        return self.create_bidirectional_module(src_op, graph, self.weights, self.rec_weights,
                                                self.bias,
                                                self.params, self.create_unidirectional_lstm)

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'activations' or attr_name == 'input_forget':
            OpSchemaBase.validate_attribute_values(src_op, attr_name, attr_value)


OnnxTranslations.register_translation(OnnxLSTMTranslation(),
                                      converter_type('LSTM', 'onnx'),
                                      op_adapter.LstmOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
# RNN
# ------------------------------------------------------------------------------

class OnnxRNNTranslation(OnnxRnnTranslationsBase):
        def __init__(self):
            OnnxRnnTranslationsBase.__init__(self)
            self.register_op_schema('RNN', [1, 7], [['clip', 'activation_alpha', 'activation_beta']]) \
                .register_method(self.validate_attribute_values)
            self.no_of_gates = 1

        def extract_parameters(self, src_op, graph):
            self.params = extract_attributes(src_op, schema=self.op_schema(), validate=True)

            self.extract_params_for_type(src_op, graph)

        def add_op(self, src_op, graph):
            self.extract_parameters(src_op, graph)
            self.add_src_op_info(src_op.name, src_op, graph)
            return self.create_rnn(src_op, graph, self.create_unidirectional_rnn,
                                   self.create_bidirectional_rnn)

        def create_unidirectional_rnn(self, name='rnn', **kargs):

            if kargs:
                [weights, _, bias] = self.convert_params_to_snpe(kargs['weights'],
                                                                 kargs['rec_weights'],
                                                                 kargs['bias'],
                                                                 kargs['hidden_size'])
            else:
                [weights, _, bias] = self.convert_params_to_snpe(self.weights,
                                                                 self.rec_weights,
                                                                 self.bias,
                                                                 self.params.hidden_size)
            return op_adapter.RnnTransformationOp(name,
                                                  weights=weights,
                                                  bias=bias,
                                                  activation=self.params.activations)

        def create_bidirectional_rnn(self, src_op, graph):
            return self.create_bidirectional_module(src_op, graph, self.weights, self.rec_weights, self.bias,
                                                    self.params, self.create_unidirectional_rnn)

        @staticmethod
        def validate_attribute_values(src_op, attr_name, attr_value):
            if attr_name == 'activations':
                OpSchemaBase.validate_attribute_values(src_op, attr_name, attr_value)


OnnxTranslations.register_translation(OnnxRNNTranslation(),
                                      converter_type('RNN', 'onnx'),
                                      op_adapter.RnnTransformationOp.TRANSLATION_KEY)
