# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
from qti.aisw.converters.common.converter_ir.op_adapter import LstmOp, ReshapeOp
from qti.aisw.converters.tensorflow.layers.constant import ConstantLayerResolver
from qti.aisw.converters.tensorflow.layers.slice import SliceLayerResolver
from qti.aisw.converters.tensorflow.layers.pack import UnPackLayerResolver, PackLayerResolver
from qti.aisw.converters.tensorflow.layers.concat import ConcatLayerResolver
from qti.aisw.converters.tensorflow.layers.fullyconnected import FullyConnectedLayerResolver
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.util import ConverterError, OperationNotFoundError, TensorNotFoundError
from qti.aisw.converters.tensorflow.sequences.lstm import concat_input_weight_cell_sequence, matmul_weight_cell_sequence
from qti.aisw.converters.common.utils import code_to_message

# Tensorflow gates weights and biases are organized in I, C, F, O
_TENSORFLOW_INPUT_GATE_INDEX = 0
_TENSORFLOW_FORGET_GATE_INDEX = 2
_TENSORFLOW_OUTPUT_GATE_INDEX = 3
_TENSORFLOW_STATE_GATE_INDEX = 1


# Define a function to re-order gate data
def reorder_gate_data_from_igfo_to_ifog(gate_data, concat_axis):
    reordered = [
        gate_data[_TENSORFLOW_INPUT_GATE_INDEX],
        gate_data[_TENSORFLOW_FORGET_GATE_INDEX],
        gate_data[_TENSORFLOW_OUTPUT_GATE_INDEX],
        gate_data[_TENSORFLOW_STATE_GATE_INDEX],
    ]
    return np.concatenate(reordered, axis=concat_axis)


def has_same_scope_name(op1, op2):
    scope_name_1 = op1.name.split('/')[:-1]
    scope_name_2 = op2.name.split('/')[:-1]

    return scope_name_1 == scope_name_2


def perform_static_matmul(weights_1, weights_2, biases_1=None):
    matmul_out = np.matmul(weights_1, weights_2)
    biases_out = np.dot(weights_2.T, biases_1) if biases_1 is not None else None

    return matmul_out, biases_out


class LstmLayerResolver(LayerResolver):
    class Descriptor(LayerDescriptor):
        def __init__(self,
                     name,
                     operations,
                     *,
                     cell_input_op,
                     cell_output_op,
                     init_cell_state_op,
                     final_cell_state_output_op,
                     forget_bias_value,
                     all_gates_matmul_op=None,
                     all_gates_biases_op=None):
            """
            The descriptor that defines a Tensorflow LSTM layer
            :param name: The name of the operation
            :param operations: The list of consumed nodes
            :param cell_input_op: The input op that consumes input data and initial hidden data
            :param cell_output_op: The output op that produces the final hidden state
            :param init_cell_state_op: The input op that consumes initial cell state data
            :param final_cell_state_output_op: The output op that produces the final cell state
            :param forget_bias_value: Optional value that to be added to the forget gate bias
            :param all_gates_matmul_op:  The op that consumes all input weight and all recurrent weight data
            :param all_gates_biases_op:  The op that consumes all input bias and recurrent gate bias data
            """
            super(LstmLayerResolver.Descriptor, self).__init__('LSTM', name, operations)
            self.cell_input_op = cell_input_op
            self.all_gates_matmul_op = all_gates_matmul_op
            self.all_gates_biases_op = all_gates_biases_op
            self.cell_output_op = cell_output_op
            self.init_cell_state_op = init_cell_state_op
            self.final_cell_state_output_op = final_cell_state_output_op
            self.forget_bias_value = forget_bias_value
            self.merge_time_steps = True
            self.cell_0 = self
            self.unrolled_cells = [self]
            self._is_stacked_cell = False
            self.pre_computed_biases_list = None
            self.proj_weights = None
            self.proj_biases = None
            self.scope_name = ''.join(self.cell_output_op.name.split("/")[:-1])

        def get_output_names_for(self, input_tensors):
            """
            Returns the output names for a given set of input tensors. If the op is a stacked cell i.e all the hidden state outputs
            are concatenated, then the unrolled cell name is returned. Otherwise, the base class function is called.
            :type input_tensors: [tensorflow.Tensor]
            :rtype list: List of output names if any are found
            """
            if not self._is_stacked_cell:
                return super(LstmLayerResolver.Descriptor, self).get_output_names_for(input_tensors)
            else:
                return [t.name for t in input_tensors if t.name == self.rolled_cell_output_name]

        def is_input_op(self, op):
            return op == self.cell_input_op or op == self.init_cell_state_op

        def is_unrolled_cell_of(self, lstm_descriptor):
            """
            An LSTM is said to be an unrolled cell if each time-step is spun into its own set of (similar) ops.
            This function checks if the passed descriptor is a time-step of the current LSTM descriptor.
            This is the case when the matmul ops have the same set of weights between both descriptors.
            :param lstm_descriptor: The candidate descriptor that could be a time-step
            :return: True if weight tensors are identical, False otherwise
            """
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_GENERAL_ABSTRACT_CLASS_MUST_BE_INHERITED'))

        def is_output_op(self, op):
            return op.outputs[0].name in self._output_tensor_names or op.outputs[0].name == self.rolled_cell_output_name

        @property
        def output_names(self):
            if not self._is_stacked_cell:
                return self._output_tensor_names
            else:
                # stacked cells do not return final and hidden cell states
                return [self.rolled_cell_output_name]

        @property
        def _output_tensor_names(self):
            # Return a pair of final cell state and hidden state
            return [str(self.unrolled_cells[-1].final_cell_state_output_op.outputs[0].name),
                    str(self.unrolled_cells[-1].cell_output_op.outputs[0].name)]

        def resolve_biases(self, graph_helper):
            if self.pre_computed_biases_list is not None:
                gates_biases = self.pre_computed_biases_list
            else:
                # split biases into 4 separate sections
                gates_biases = graph_helper.evaluate_tensor_output(self.all_gates_biases_op.inputs[1])
            gates_biases = np.split(gates_biases, indices_or_sections=4, axis=0)
            # add forget bias value
            gates_biases[_TENSORFLOW_FORGET_GATE_INDEX] += self.forget_bias_value
            # re-order gates biases to I,F,O,C from I,C,F,O
            gates_biases = reorder_gate_data_from_igfo_to_ifog(gate_data=gates_biases, concat_axis=0)
            return gates_biases

        def resolve_weights(self, graph_helper, state_shape, **kwargs):
            """
            Weights are organized differently depending on the LSTM type. Each sub class must define this function

            :returns A pair of recurrent gate weights and input gate weights organized in I, F, O, G format
                     where input gate weights dimension is [input_size, 4*hidden_size] and recurrent gate weights
                     dimension is [hidden_size, 4*hidden_size]. Function must return gate_weights, input_weights in
                     that order
            """
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_GENERAL_ABSTRACT_CLASS_MUST_BE_INHERITED'))

        @property
        def rolled_cell_output_name(self):
            """
            The output for an LSTM can be a full sequence of intermediate hidden states for all time-steps.
            This could be present in a model as a concat op whose inputs are the hidden states.
            If the concat op is found, its name is used as a rolled cell output name, otherwise,
            a tensor name is created to capture the output produced by the runtimes.
            :return:
            """
            cell_child_op = self.cell_0.child_ops[-1]
            # if intermediate outputs were already stacked, then use existing stack output name otherwise return new name
            out_tensor_names = self._output_tensor_names
            if cell_child_op.type in ["Pack"] and \
                    cell_child_op.inputs[0].shape == self.cell_output_op.outputs[0].shape:
                return cell_child_op.outputs[0].name
            return '{}_all_time_steps'.format(out_tensor_names[-1])

        def returns_state(self):
            """
            Checks if any of the final states are to be consumed by non-LSTM child ops
            :return: True if consumers are found, false otherwise
            """
            last_cell = self.cell_0.unrolled_cells[-1]
            cell_state_consumers = last_cell.final_cell_state_output_op.outputs[0].consumers()
            hidden_state_consumers = last_cell.cell_output_op.outputs[0].consumers()

            return not all(consumer in self.cell_0.child_ops for consumer in hidden_state_consumers) or \
                   not all(consumer in self.cell_0.child_ops for consumer in cell_state_consumers)

        def set_is_stacked_cell(self, is_stacked_cell):
            # TO-DO: Need to evaluate if we really need this
            self._is_stacked_cell = is_stacked_cell

        @property
        def initial_state_names(self):
            cell_state_input_name = self.init_cell_state_op.inputs[0].name
            hidden_state_input_name = self.cell_input_op.inputs[1].name

            # we need to double check the cell input name because the init cell state input is a mul op
            # so the state can be either inputs[1] or inputs[0]
            # First check makes sure it does not contain Sigmoid (forget gate output), second check
            # checks scoping (the initial state is usually un-scoped)
            if "Sigmoid" in cell_state_input_name or self.scope_name in cell_state_input_name:
                cell_state_input_name = self.init_cell_state_op.inputs[1].name

            return [cell_state_input_name, hidden_state_input_name]

        def time_steps(self):
            return len(self.unrolled_cells)

    @staticmethod
    def _match_common_input_ops(matched_sequence):
        # The init_cell state op consumes initial cell state [c_t-1] and output of ft to produce ft*c_t-1
        init_cell_state_op = matched_sequence['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul']

        return init_cell_state_op

    @staticmethod
    def _match_common_output_ops(matched_sequence):
        # The cell state output op is c_t above
        final_cell_state_output_op = matched_sequence['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1']

        # The cell output op produces the output of h_t.
        # Note that each individual time-step will be matched, and that the output of this op could be fed into
        # the cell_input_op for another match (as the initial hidden state)
        cell_output_op = matched_sequence['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2']

        return cell_output_op, final_cell_state_output_op

    @staticmethod
    def _match_common_bias_ops(matched_sequence, graph_helper):
        # The gates biases op consumes X*W_input + H_t-1*W_recurrent and the gate biases [B_input, B_recurrent]
        # Following the bias add, the output is split and fed into each respective gate
        all_gates_biases_op = matched_sequence['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/BiasAdd']

        # get forget bias value if the add operation is present
        # if not, use the default value of 0.0
        if matched_sequence['forget_gate_input'].type in ["Add", "AddV2"]:
            forget_bias_op = matched_sequence['forget_gate_input']
            _, forget_bias_tensor = graph_helper.get_op_input_tensors(forget_bias_op, ('?', 'Const'))
            forget_bias_value = float(graph_helper.evaluate_tensor_output(forget_bias_tensor))
        else:
            forget_bias_value = 0.0

        return all_gates_biases_op, forget_bias_value

    def resolve_layer(self, graph_matcher, graph_helper):
        """
            The LSTM op is matched as a sequence of ops which perform the following set of computations.
            The results h_t and c_t are considered to be the outputs of the descriptor that will be matched.
            Please adjust the computations to reflect any additions to the existing supported sequence.
            -  gate_data = X_t*W_input + H_t-1*W_recurrent + B
            - i_t  = sigmoid(split(gate_data)[0])
            - f_t  = sigmoid(split(gate_data)[2]) (argument +1 if unit forget bias is set)
            - g_t  = tanh(split(gate_data)[1])
            - c_t = f_t (.) c_t-1 + i_t (.) c_t
            - o_t = sigmoid(split(gate_data)[3])) if peepholes are set
            - h_t = ot (.) h(Ct)
        """
        raise ConverterError(code_to_message.get_error_message('ERROR_TF_GENERAL_ABSTRACT_CLASS_MUST_BE_INHERITED'))


class MergedWeightsLstmLayerResolver(LstmLayerResolver):
    class Descriptor(LstmLayerResolver.Descriptor):
        def __init__(self, name, operations, *, cell_input_op, cell_output_op, init_cell_state_op, final_cell_state_output_op, forget_bias_value,
                     all_gates_matmul_op, all_gates_biases_op):
            super().__init__(name, operations, cell_input_op=cell_input_op, cell_output_op=cell_output_op, init_cell_state_op=init_cell_state_op,
                             final_cell_state_output_op=final_cell_state_output_op, forget_bias_value=forget_bias_value,
                             all_gates_matmul_op=all_gates_matmul_op, all_gates_biases_op=all_gates_biases_op)

        def is_input_tensor(self, op, tensor):
            # Ignores a static axis input which has already been consumed by the resolver
            if tensor.op.type == "Const" and (len(self.cell_input_op.inputs) == 3 and
                                              tensor == self.cell_input_op.inputs[-1]):
                return False
            return True

        def is_unrolled_cell_of(self, lstm_descriptor):
            if not isinstance(lstm_descriptor, MergedWeightsLstmLayerResolver.Descriptor):
                return False

            # if the output ops have the same scope name, then we can trivially conclude
            # that they are the same rnn. If not, we will check if the matmul ops (weight ops)
            # are shared.
            if has_same_scope_name(self.cell_output_op, lstm_descriptor.cell_output_op):
                return True

            return self.all_gates_matmul_op.inputs[1].op == lstm_descriptor.all_gates_matmul_op.inputs[1].op

        def resolve_weights(self, graph_helper, state_shape, **kwargs):
            merged_weights = graph_helper.evaluate_tensor_output(self.all_gates_matmul_op.inputs[1])
            input_weights_slice_index = np.shape(merged_weights)[0] - state_shape[-1]
            weights_list = np.split(merged_weights,
                                    indices_or_sections=[input_weights_slice_index],
                                    axis=0)

            input_weights = np.split(weights_list[0], indices_or_sections=4, axis=1)
            input_weights = reorder_gate_data_from_igfo_to_ifog(input_weights, concat_axis=1)

            gates_weights = np.split(weights_list[1], indices_or_sections=4, axis=1)
            gates_weights = reorder_gate_data_from_igfo_to_ifog(gates_weights, concat_axis=1)

            return gates_weights, input_weights

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(concat_input_weight_cell_sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            # The cell input op consumes the input data [X] and initial hidden state [h_t-1] as inputs
            cell_input_op = match['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/concat']

            # The gates matmul op consumes concatenated [X, h] and gate weights [ W_input, W_recurrent] as inputs
            all_gates_matmul_op = match['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/MatMul']

            # Match all other common operations in the sequence
            init_cell_state_op = self._match_common_input_ops(match)
            cell_output_op, final_cell_state_output_op = self._match_common_output_ops(match)
            all_gates_biases_op, forget_bias_value = self._match_common_bias_ops(match, graph_helper)

            d = MergedWeightsLstmLayerResolver.Descriptor(name=str(cell_output_op.name),
                                                          operations=match.consumed_nodes,
                                                          cell_input_op=cell_input_op,
                                                          all_gates_matmul_op=all_gates_matmul_op,
                                                          all_gates_biases_op=all_gates_biases_op,
                                                          cell_output_op=cell_output_op,
                                                          init_cell_state_op=init_cell_state_op,
                                                          final_cell_state_output_op=final_cell_state_output_op,
                                                          forget_bias_value=forget_bias_value)
            descriptors.append(d)

        if len(descriptors) == 0:
            return []

        return descriptors


class SplitWeightsLstmLayerResolver(LstmLayerResolver):
    class Descriptor(LstmLayerResolver.Descriptor):
        def __init__(self, name, operations, *, input_weights_matmul_op, rec_weights_matmul_op, cell_input_op, cell_output_op, init_cell_state_op,
                     final_cell_state_output_op, forget_bias_value, all_gates_biases_op):
            super().__init__(name, operations, cell_input_op=cell_input_op, cell_output_op=cell_output_op, init_cell_state_op=init_cell_state_op,
                             final_cell_state_output_op=final_cell_state_output_op, forget_bias_value=forget_bias_value,
                             all_gates_biases_op=all_gates_biases_op)
            self.input_weights_matmul_op = input_weights_matmul_op
            self.rec_weights_matmul_op = rec_weights_matmul_op
            self.pre_computed_input_weights_list = None
            self.pre_computed_rec_weights_list = None
            self._initial_state_names = None
            self.merge_time_steps = True

        def is_input_op(self, op):
            return op in [self.input_weights_matmul_op, self.rec_weights_matmul_op, self.init_cell_state_op]

        def is_unrolled_cell_of(self, lstm_descriptor):
            if not isinstance(lstm_descriptor, SplitWeightsLstmLayerResolver.Descriptor):
                return False

            # if the output ops have the same scope name, then we can trivially conclude
            # that they are the same rnn. If not, we will check if the matmul ops (weight ops)
            # are shared.
            if has_same_scope_name(self.cell_output_op, lstm_descriptor.cell_output_op):
                return True

            return self.input_weights_matmul_op.inputs[1].op == lstm_descriptor.input_weights_matmul_op.inputs[1].op and \
                   self.rec_weights_matmul_op.inputs[1].op == lstm_descriptor.rec_weights_matmul_op.inputs[1].op

        def resolve_weights(self, graph_helper, state_shape, **kwargs):
            if self.pre_computed_input_weights_list is not None:
                merged_input_weights = self.pre_computed_input_weights_list
            else:
                merged_input_weights = graph_helper.evaluate_tensor_output(self.input_weights_matmul_op.inputs[1])

            input_weights = np.split(merged_input_weights, indices_or_sections=4, axis=1)
            input_weights = reorder_gate_data_from_igfo_to_ifog(input_weights, concat_axis=1)

            if self.pre_computed_rec_weights_list is not None:
                merged_rec_weights = self.pre_computed_rec_weights_list
            else:
                merged_rec_weights = graph_helper.evaluate_tensor_output(self.rec_weights_matmul_op.inputs[1])

            gates_weights = np.split(merged_rec_weights, indices_or_sections=4, axis=1)
            gates_weights = reorder_gate_data_from_igfo_to_ifog(gates_weights, concat_axis=1)

            return gates_weights, input_weights

        @property
        def initial_state_names(self):
            if self._initial_state_names is not None:
                return self._initial_state_names
            return [self.init_cell_state_op.inputs[-1].name, self.rec_weights_matmul_op.inputs[0].name]

        @initial_state_names.setter
        def initial_state_names(self, value):
            self._initial_state_names = value

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(matmul_weight_cell_sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            # The input weights matmul op consumes [X] and gate weights [ W_input] as inputs
            input_weights_matmul_op = match['rnn/MatMul']

            # The rec weights matmul op consumes [h] and gate weights [ W_recurrent] as inputs
            rec_weights_matmul_op = match['rnn/MatMul_1']

            # In this case the cell input op is also the input weights op (X is input[0])
            cell_input_op = input_weights_matmul_op

            # Match all other common operations in the sequence
            init_cell_state_op = self._match_common_input_ops(match)
            cell_output_op, final_cell_state_output_op = self._match_common_output_ops(match)
            all_gates_biases_op, forget_bias_value = self._match_common_bias_ops(match, graph_helper)

            d = SplitWeightsLstmLayerResolver.Descriptor(name=str(cell_output_op.name),
                                                         operations=match.consumed_nodes,
                                                         cell_input_op=cell_input_op,
                                                         input_weights_matmul_op=input_weights_matmul_op,
                                                         rec_weights_matmul_op=rec_weights_matmul_op,
                                                         all_gates_biases_op=all_gates_biases_op,
                                                         cell_output_op=cell_output_op,
                                                         init_cell_state_op=init_cell_state_op,
                                                         final_cell_state_output_op=final_cell_state_output_op,
                                                         forget_bias_value=forget_bias_value)
            descriptors.append(d)

        if len(descriptors) == 0:
            return []

        return descriptors


class LstmLayerBuilder(LayerBuilder):
    @classmethod
    def _add_reshape_to_restore_time_dimension(cls, ir_graph, descriptor, input_name, input_shape):
        """
        This functions inserts a reshape op from 2D to 3D before the LSTM op. This is based on an observation
        that LSTM models may have a split op to unpack the data for each individual time-step. Since we combining
        all time-steps into a single op and removing the split, we need to re-insert a reshape after the input data.
        """

        if len(input_shape) != 2:
            raise ValueError("Input shape to restore for LSTM layer: {} must be of size 2. "
                             "Got {} instead".format(descriptor.layer_name, len(input_shape)))

        reshape_layer_name = '{}_reshape'.format(descriptor.layer_name)
        reshape_output = [input_shape[0], descriptor.time_steps(), input_shape[1]]
        ir_graph.add(ReshapeOp(reshape_layer_name,
                               reshape_output),
                     input_name,
                     reshape_layer_name)
        return reshape_layer_name

    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: LstmLayerResolver.UnrolledTimeStepDescriptor
        :rtype: int
        """
        input_shape = converter_context.graph_helper.get_op_output_shape(descriptor.cell_input_op.inputs[0].op)
        state_shape = converter_context.graph_helper.get_op_output_shape(descriptor.init_cell_state_op.inputs[0].op)

        # Weights are organized in I, C, F, O but converter supports I, O, F, C
        gates_weights, input_weights = descriptor.resolve_weights(converter_context.graph_helper, state_shape)
        gates_biases = descriptor.resolve_biases(converter_context.graph_helper)

        def is_cell_input_descriptor(cell_descriptor):

            # Check simple case of initial cell state
            # Input descriptors are not cell inputs if they are inputs to init_cell_state_op
            if cell_descriptor.child_ops[-1].outputs[0] in descriptor.init_cell_state_op.inputs:
                return False
            else:
                output_shape = []
                output_ops = [op for op in cell_descriptor.child_ops if cell_descriptor.is_output_op(op)]
                if len(output_ops) > 0:
                    output_shape = converter_context.graph_helper.get_op_output_shape(output_ops[-1])

                if len(output_shape) == 3:
                    has_expected_shape = output_shape[-2] == descriptor.time_steps() and output_shape[-1] == input_shape[1]
                else:
                    has_expected_shape = output_shape == input_shape

                if len(output_shape) >= 2 and has_expected_shape:
                    # We need to account for the corner case where initial hidden state and input data have the same shape
                    # In that case both the hidden input and data input descriptor will pass the above test
                    # We can check to make sure the input data produced is not in the initial state names
                    if input_shape == state_shape:
                        return cell_descriptor.output_names[0] not in descriptor.initial_state_names
                    else:
                        return True
                return False

        cell_input_descriptors = list(filter(is_cell_input_descriptor, input_descriptors))
        cell_state_descriptors = list(filter(lambda x: any(True for out_name in x.output_names
                                                           if out_name in descriptor.initial_state_names), input_descriptors))

        # This is the case when distinct LSTM cells are stacked above each other
        is_stacked_above_cell = self.is_stacked_cell(input_descriptors)

        # get the input layer name
        # Use the original cell input op in case the cell input descriptor has more than one output
        input_layer_name = descriptor.cell_input_op.inputs[0].name

        if not is_stacked_above_cell:
            # There can be at most one cell input descriptor in this case
            if len(list(cell_input_descriptors)) != 1:
                raise ConverterError('Unable to resolve LSTM input layer name.')

            # Sanity check the input layer name given that the LSTM is not stacked
            # Fall back to the cell input descriptor first output name as a default
            if input_layer_name not in cell_input_descriptors[0].output_names:
                input_layer_name = cell_input_descriptors[0].output_names[0]

            # need to reshape if the input to the cell is 2D and reshape input has been ignored
            if self._needs_reshape_to_restore_time_dimension(converter_context,
                                                             descriptor, cell_input_descriptors[0], input_shape):
                input_layer_name = self._add_reshape_to_restore_time_dimension(
                    ir_graph, descriptor, input_layer_name, input_shape)

        else:
            # sanity check input layer name
            if input_layer_name not in input_descriptors[0].output_names:
                input_layer_name = cell_input_descriptors[0].output_names[-1]

        # This checks for sequential LSTM layers i.e the output of this LSTM descriptor is the input to another
        is_stacked_below_cell = self.is_stacked_cell(output_descriptors)
        descriptor.set_is_stacked_cell(is_stacked_below_cell)

        # User initial state determines if an initial state was passed and if a final cell state will be returned.
        if cell_state_descriptors:
            # There are two cell state descriptors
            # Case 1: LSTM op returns the final and hidden state to be consumed by other ops,
            #         in this case we need user initial state to be true.
            # Case 2: If it is a stacked below cell,
            #         then a returned state is consumed by another LSTM. We will decide on the initial state
            #         based on the value of the initial state
            if descriptor.returns_state() and not is_stacked_below_cell:
                user_initial_state = True
            else:
                # Case 3: Initial state is provided but is zero and final states are not consumed by other ops
                # TO-DO: remove once backend spec is aligned
                user_initial_state = all(d.value.any() for d in cell_state_descriptors
                                         if isinstance(d, ConstantLayerResolver.Descriptor))
        else:
            # Case 4: No initial state
            user_initial_state = False

        # At the very least each runtime, will return a buffer containing all time-steps
        output_names = [descriptor.rolled_cell_output_name]
        h_0_input_name, c_0_input_name = "", ""

        # if there is no user initial state then the initial hidden state and cell state inputs are empty
        # Else, the initial hidden state and cell state are expected to have predefined input names
        # and all possible output buffers are produced
        if not user_initial_state:
            # if timesteps were not merged, then we should use the original final hidden state name
            if not descriptor.merge_time_steps:
                output_names = [descriptor.output_names[-1]]
        elif user_initial_state:
            c_0_input_name, h_0_input_name = descriptor.initial_state_names
            # if the user initial state is provided, then the final cell state and hidden state are returned
            output_names = [descriptor.rolled_cell_output_name, *descriptor.output_names]

        input_names = [input_layer_name]

        if h_0_input_name:
            input_names.append(h_0_input_name)
        if c_0_input_name:
            input_names.append(c_0_input_name)

        # The internal state of the cell is reset at each time step if there is no user initial state or
        # if the cell is stacked. Note if the cell is stacked, then the initial state is the previous
        # final cell state and hidden state at time step 0
        reset_at_time_step_0 = not user_initial_state or is_stacked_below_cell

        return ir_graph.add(LstmOp(name=descriptor.cell_0.child_ops[-1].name,
                                   input_weights=input_weights,
                                   gate_bias=gates_biases,
                                   hidden_state_weights=gates_weights,
                                   reset_state_at_time_step_0=reset_at_time_step_0,
                                   c_0_input_name=c_0_input_name,
                                   h_0_input_name=h_0_input_name,
                                   hidden_size=state_shape[1],
                                   proj_weights=descriptor.proj_weights,
                                   proj_bias=descriptor.proj_biases), input_names=input_names, output_names=output_names)

    @staticmethod
    def _merge_concat_timestep_descriptor(converter_context, descriptor, output_descriptors):
        # This function merges the concat descriptor into cell_0 if one exists and
        # it only concatenates all time-steps

        lstm_concat_descs = [d for d in output_descriptors if
                               isinstance(d, (ConcatLayerResolver.Descriptor, PackLayerResolver.Descriptor))]
        for concat_desc in lstm_concat_descs:
            # ensure input descriptors to concat are all lstm ops
            lstm_concat_input_descriptors = converter_context.topology_resolver.get_input_layers_for(
                concat_desc)
            # These need to be in order
            if lstm_concat_input_descriptors == descriptor.cell_0.unrolled_cells:
                # This could be either a concatenation of all time-steps or a concatenation of a single timestep's hidden state and cell state
                # We only support merging the former
                concat_op = concat_desc.child_ops[-1]
                concat_input_names = [input_.name for input_ in concat_op.inputs if concat_desc.is_input_tensor(concat_op, input_)]
                hidden_state_out_names = [desc.cell_output_op.outputs[0].name for desc in lstm_concat_input_descriptors]

                # check that only the hidden state output names are present, ignore axis attribute
                if hidden_state_out_names == concat_input_names:
                    converter_context.merge_descriptors(concat_desc, descriptor.cell_0)

    @staticmethod
    def _fold_matmul_weights(descriptor, input_descriptors, converter_context):
        if not isinstance(descriptor, SplitWeightsLstmLayerResolver.Descriptor):
            raise ConverterError("Matmul folding is not applicable to LSTM descriptor: {}".format(descriptor.layer_name))

        # extract matmul descriptors
        matmul_descriptors = [desc for desc in input_descriptors if isinstance(desc, FullyConnectedLayerResolver.Descriptor)]

        in_weights_value, _, _ = converter_context.graph_helper. \
            get_static_data_info(descriptor.input_weights_matmul_op, descriptor.input_weights_matmul_op.inputs[1])

        rec_weights_value, _, _ = converter_context.graph_helper. \
            get_static_data_info(descriptor.rec_weights_matmul_op, descriptor.rec_weights_matmul_op.inputs[1])

        descriptor.pre_computed_biases_list = converter_context.graph_helper.evaluate_tensor_output(descriptor.all_gates_biases_op.inputs[1])
        folded_matmul_bias = None

        for matmul_desc in matmul_descriptors:
            # check that matmul op has same scope
            if not has_same_scope_name(matmul_desc.matmul_op, descriptor.cell_output_op) or matmul_desc.is_ignored:
                # do nothing because this is probably a projection layer
                continue
            elif matmul_desc.transpose_b or matmul_desc.transpose_a:
                # do nothing because the math does not check out if these vars are set
                continue
            else:
                weights_tensor = matmul_desc.weights
                # Check that matmul output is input to either input weights or rec weights
                if matmul_desc.matmul_op.outputs[0] in descriptor.input_weights_matmul_op.inputs:
                    descriptor.pre_computed_input_weights_list, folded_matmul_bias = perform_static_matmul(weights_tensor, in_weights_value,
                                                                                                           biases_1=matmul_desc.biases)
                    descriptor.cell_input_op = matmul_desc.matmul_op
                    matmul_desc.set_ignored(True)
                elif matmul_desc.matmul_op.outputs[0] in descriptor.rec_weights_matmul_op.inputs:
                    descriptor.pre_computed_rec_weights_list, folded_matmul_bias = perform_static_matmul(weights_tensor, rec_weights_value,
                                                                                                         biases_1=matmul_desc.biases)
                    matmul_desc.set_ignored(True)

                    # reset initial state names since matmul output for hidden has now been ignored
                    descriptor.initial_state_names = [descriptor.init_cell_state_op.inputs[-1].name, matmul_desc.matmul_op.inputs[0].name]

                if folded_matmul_bias is not None:
                    descriptor.pre_computed_biases_list += folded_matmul_bias

    @staticmethod
    def _ignore_split_reshape_layers(converter_context, descriptor, input_descriptors):
        # This function ignores the reshape and split layers if applicable provided timestep merging has occurred
        # Note that the reshape and split layers simply divide the feature data into time-steps, given merging
        # these layers can be safely ignored.
        reshape_descriptor = None
        split_descriptor = None

        def is_split_or_unpack(in_desc):
            return isinstance(in_desc, (SliceLayerResolver.Descriptor, UnPackLayerResolver.Descriptor))

        lstm_split_inputs = list(filter(is_split_or_unpack, input_descriptors))

        # In the case of split weights lstm, we need to use the folded matmul layer's input descriptors
        # since the matmul layer has been ignored.
        if not lstm_split_inputs and isinstance(descriptor, SplitWeightsLstmLayerResolver.Descriptor) \
             and descriptor.cell_0.pre_computed_input_weights_list is not None:
            for in_desc in input_descriptors:
                if isinstance(in_desc, FullyConnectedLayerResolver.Descriptor) and in_desc.is_ignored:
                    # get input descriptors for ignored folded matmul
                    folded_matmul_in_descriptors = converter_context.topology_resolver.get_input_layers_for(in_desc)
                    input_descriptors.extend(folded_matmul_in_descriptors)
            lstm_split_inputs = list(filter(is_split_or_unpack, input_descriptors))

        if len(lstm_split_inputs) == 1:
            lstm_split_out_descriptors = converter_context.topology_resolver.get_output_layers_for(lstm_split_inputs[0])
            # ensure number of output descriptors for split op matches number of unrolled cells
            # These need not be in order
            if len(lstm_split_out_descriptors) == len(descriptor.cell_0.unrolled_cells):
                split_descriptor = lstm_split_inputs[0]
                try:
                    # This block of code searches for the (Placeholder, reshape) input pattern to the lstm
                    # and removes the reshape if it simply squeezes the dimension
                    # note that by checking if the reshape shape is in the placeholder dimension
                    # we should be able to guarantee that all other dimensions must be 1
                    _, reshape_tensor = converter_context.graph_helper.get_op_input_tensors(
                        split_descriptor.child_ops[-1], ("Const", "Reshape"))
                    result = converter_context.graph_helper.get_op_sequence(reshape_tensor.op,
                                                                            ['Reshape', 'Placeholder'])
                    placeholder_tensor_shape = converter_context.graph_helper.get_op_output_shape(result[1])
                    reshape_tensor_shape = converter_context.graph_helper.get_op_output_shape(result[0])

                    if all(r_shape in placeholder_tensor_shape for r_shape in reshape_tensor_shape):
                        reshape_descriptor = converter_context.topology_resolver.get_input_layers_for(split_descriptor)[
                            0]
                        reshape_descriptor.set_ignored(True)
                except (OperationNotFoundError, TensorNotFoundError):
                    # this means the reshape merging failed, and that is no error so it should be skipped
                    pass
                # check if there is only one
                split_descriptor.set_ignored(True)

        return reshape_descriptor is not None and split_descriptor is not None

    @classmethod
    def is_stacked_cell(cls, descriptors):
        return len(descriptors) >= 1 and isinstance(descriptors[0], LstmLayerResolver.Descriptor)

    @classmethod
    def _needs_reshape_to_restore_time_dimension(cls, converter_context, cell_descriptor, in_descriptor, input_shape):

        if len(input_shape) != 2:
            return False

        input_shape = converter_context.graph_helper.get_op_output_shape(
            cell_descriptor.cell_input_op.inputs[0].op)
        in_descriptor_shape = converter_context.graph_helper.get_op_output_shape(in_descriptor.child_ops[-1])
        return in_descriptor_shape != [input_shape[0], cell_descriptor.time_steps(), input_shape[1]]

    @classmethod
    def _merge_unrolled_input_cells(cls, converter_context, input_descriptors, descriptor):
        lstm_inputs = [i for i in input_descriptors if isinstance(i, LstmLayerResolver.Descriptor)]
        unrolled_inputs = [d for d in lstm_inputs if descriptor.is_unrolled_cell_of(d.cell_0)]
        for input_descriptor in unrolled_inputs:
            converter_context.merge_descriptors(descriptor, input_descriptor.cell_0)
            input_descriptor.cell_0.unrolled_cells.append(descriptor)
            descriptor.cell_0 = input_descriptor.cell_0

    def transform_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):

        # For SplitWeights LSTM variant, we can collapse matmul operations for both input and recurrent weights
        # This happens in the special case of SVD LSTM, where both weight matrices are split into pairs.
        if isinstance(descriptor, SplitWeightsLstmLayerResolver.Descriptor):
            self._fold_matmul_weights(descriptor, input_descriptors, converter_context)

        # Merge the current descriptor into cell_0 if it appears to be a time-step of cell_0
        # Only applicable when merge time step is set
        if descriptor.merge_time_steps:
            self._merge_unrolled_input_cells(converter_context, input_descriptors, descriptor)

        # if both reshape and split have been ignored then we may not need to reshape
        self._ignore_split_reshape_layers(converter_context, descriptor, input_descriptors)

        # Merge concat descriptor only if it concatenates all time-steps and there are no other LSTM layers left to roll
        if not any(isinstance(out_desc, LstmLayerResolver.Descriptor) for out_desc in output_descriptors) and descriptor.merge_time_steps:
            self._merge_concat_timestep_descriptor(converter_context, descriptor, output_descriptors)

        return
