# =============================================================================
#
#  Copyright (c) 2017-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
import copy

cell_sequence = GraphSequence([
    NonConsumableConverterSequenceNode('rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read', ['Identity', 'Const']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/BiasAdd',
                          ['BiasAdd']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split', ['Split']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh', ['Tanh']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1', ['Sigmoid']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid', ['Sigmoid']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1', ['Mul']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul', ['Mul']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1', ['Add', 'AddV2']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2', ['Sigmoid']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1', ['Tanh']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2', ['Mul']),
    NonConsumableConverterSequenceNode('stub_19', ['?']),
    NonConsumableConverterSequenceNode('stub_20', ['?']),
    NonConsumableConverterSequenceNode('stub_21', ['?']),
    NonConsumableConverterSequenceNode('stub_22', ['?']),
    ConverterSequenceNode('forget_gate_input', ['?']),
])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid',
                              ['forget_gate_input'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul',
                              ['stub_22', 'rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/BiasAdd',
                              ['stub_21',
                               'rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1',
                               'rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul',
                               'rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split',
                              ['stub_20',
                               'rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/BiasAdd'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1',
                               'rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split'])
cell_sequence.set_inputs('rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read', ['stub_19'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split'])
cell_sequence.set_outputs(['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2'])


# define entry point sequence for concatenated input and weight data
concat_input_weight_sequence = GraphSequence([
    NonConsumableConverterSequenceNode('rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read', ['Identity', 'Const']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/concat',
                          ['ConcatV2']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/MatMul',
                          ['MatMul']),

    NonConsumableConverterSequenceNode('stub_25', ['?']),
    NonConsumableConverterSequenceNode('stub_26', ['?']),
    NonConsumableConverterSequenceNode('stub_27', ['?']),
    NonConsumableConverterSequenceNode('stub_28', ['?']),
])

concat_input_weight_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/concat',
                              ['stub_25', 'stub_26', 'stub_27'])
concat_input_weight_sequence.set_inputs('rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read', ['stub_28'])
concat_input_weight_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/MatMul',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/concat',
                               'rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read'])

# connect sequence to cell sequence
concat_input_weight_cell_sequence = concat_input_weight_sequence
concat_input_weight_cell_sequence.update(copy.deepcopy(cell_sequence))
concat_input_weight_cell_sequence.clear_inputs_for_nodes(['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/BiasAdd'])
concat_input_weight_cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/BiasAdd',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/MatMul',
                               'rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read'])
concat_input_weight_cell_sequence.set_outputs(['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2'])

matmul_weight_no_concat_sequence = GraphSequence([
    ConverterSequenceNode('rnn/MatMul', ['MatMul']),
    ConverterSequenceNode('rnn/MatMul_1', ['MatMul']),
    ConverterSequenceNode('rnn/add', ['Add', 'AddV2']),
    NonConsumableConverterSequenceNode('stub_30', ['?']),
    NonConsumableConverterSequenceNode('stub_31', ['?']),
    NonConsumableConverterSequenceNode('stub_32', ['?']),
    NonConsumableConverterSequenceNode('stub_33', ['?']),
])
matmul_weight_no_concat_sequence.set_inputs('rnn/MatMul_1', ['stub_30', 'stub_31'])
matmul_weight_no_concat_sequence.set_inputs('rnn/MatMul', ['stub_32', 'stub_33'])
matmul_weight_no_concat_sequence.set_inputs('rnn/add', ['rnn/MatMul', 'rnn/MatMul_1'])

# connect sequence to cell sequence
matmul_weight_cell_sequence = matmul_weight_no_concat_sequence
matmul_weight_cell_sequence.update(copy.deepcopy(cell_sequence))
matmul_weight_cell_sequence.clear_inputs_for_nodes(['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/BiasAdd'])
matmul_weight_cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/BiasAdd',
                              ['rnn/add', 'rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read'])
matmul_weight_cell_sequence.set_outputs(['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2'])
