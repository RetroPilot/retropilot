#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2018-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from __future__ import print_function

import os
import zipfile
from functools import reduce
from collections import OrderedDict
import csv
import logging
import sys
import math

try:
    from qti.aisw.dlc_utils import modeltools
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that $SNPE_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)

csv_file_flag = False

# sizes of various C data types, used for memory estimation
SIZEOF_INT8_T = 1
SIZEOF_UINT8_T = 1
SIZEOF_INT16_T = 2
SIZEOF_HALF = 2
SIZEOF_FLOAT = 4

def setUpLogger(verbose):
    formatter = '%(asctime)s - %(lineno)d - %(levelname)s - %(message)s'
    lvl = logging.INFO
    if verbose:
        lvl = logging.DEBUG
    logger = logging.getLogger()
    logger.setLevel(lvl)
    formatter = logging.Formatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def neuron_name(type_code):
        func_name = { modeltools.NEURON_RELU:"relu",
                      modeltools.NEURON_RELU_MIN_MAX:"relu_min_max",
                      modeltools.NEURON_LOGISTIC:"logistic",
                      modeltools.NEURON_TANH:"tanh",
                      modeltools.NEURON_ELU:"elu",
                      modeltools.NEURON_HSWISH:"hswish",
                      modeltools.NEURON_NONE:"none"}
        return func_name[type_code]

def color_space_name(type_code):
        # tracks the DNN PB format
        cs_name = { 0:"rgb",
                    1:"argb32",
                    2:"rgba",
                    3:"nv21",
                    4:"bgr",
                    5:"blob1d",
                    6:"blob2d" }

        return cs_name[type_code]

def input_type_name(type_code):
        # tracks the DNNPB format
        type_name = { 0:"default",
                      1:"image" ,
                      2:"opaque"}
        return type_name[type_code]

def padding_mode_name(type_code):
        mode_name = { 1:"zero",
                      2:"reflect",
                      3:"constant",
                      4:"edge"}

        return mode_name[type_code]

def resize_mode_name(type_code):
        resize_mode = { 0:"bilinear",
                        1:"nearest_neighbor" }
        return resize_mode[type_code]

def prior_box_name(type_code):
        prior_box_type = { 0:"corner",
                           1:"center_size",
                           2:"corner_size" }
        return prior_box_type[type_code]

def nms_type_name(nms_type_code):
        nms_type = { 0:"fast",
                     1:"regular" }
        return nms_type[nms_type_code]

def print_row(values, col_sizes, csv_content = []):
    str = '| '
    for value, size in zip(values, col_sizes):
       str += '{0:<{1}}|'.format(value, size) + ' '
    str = str[:-1]
    print (str)
    if csv_file_flag:
        csv_content.append(values)

def print_value(value, csv_content = []):
    print(value)
    if csv_file_flag:
        for i in value.split('\n'):
            csv_content.append([i])

def product(numbers):
    if len(numbers) == 0:
        return 1
    else:
        return reduce((lambda x, y: x * y), numbers)

def get_si_notation(n, total):
    if (total > 0):
        percent = 100*float(n)/total
    else:
        percent = 0
    if n < 1000:
        return "%d (%.3g%%)" % (n, percent)
    elif n < 1000*1000:
        return '%dk (%.3g%%)' % (n/1000, percent)
    else:
        return '%dM (%.3g%%)' % (n/(1000*1000), percent)

def get_binary_si_notation(n, total):
    result_string = ""

    if n < 1024:
        result_string = "%d B" % (n)
    elif n < 1024*1024:
        result_string = '%.01f kiB' % (n/1024.0)
    else:
        result_string = '%.01f MiB' % (n/(1024.0*1024.0))

    if total is not None:
        if (total > 0):
            percent = 100*float(n)/total
        else:
            percent = 0
        if n < 1024:
            result_string += "(%.2f%%)" % (percent)
        elif n < 1024*1024:
            result_string += '(%.2f%%)' % (percent)
        else:
            result_string += '(%.2f%%)' % (percent)

    return result_string

class LayerRow(object):
    def __init__(self, layer, prev_rows, model):
        self.layer = layer
        self.name = layer['name']
        self.id = layer['id']
        self.type = layer['type']
        self.input_names = layer['input_names']
        self.input_dims = [model.get_buffer_dims(i) for i in self.input_names]
        if len(self.input_dims) < 1:
            self.input_batch = 0
        elif len(self.input_dims[0]) < 1:
            self.input_batch = 0
        else:
            self.input_batch = self.input_dims[0][0]
        self.output_names = layer['output_names']
        self.output_dims_list = []
        self.output_precisions = []

        self.valid_cpu = layer['valid_cpu']
        self.valid_gpu = layer['valid_gpu']
        self.valid_dsp = layer['valid_dsp']
        self.valid_aip = layer['valid_aip']

        self.macs = 0
        self.param_count = 0 # i.e. size of weights
        self.nativeDimSize = 3
        self.layer_affinity = model.get_layer_affinity(self.name)
        for name in self.output_names:
           self.output_dims_list.append(model.get_buffer_dims(name))
           self.output_precisions.append(model.get_buffer_precision(name))

        self.parms = []

        self.memory_cpu = 0
        self.memory_fxp = 0

        # every layer uses at least the memory for its own output buffers
        for output_index in range(0, len(self.output_dims_list)):

            output_dims = self.output_dims_list[output_index]

            if self.output_precisions[output_index] == modeltools.PRECISION_FIXED_8:
                cpu_element_size = SIZEOF_INT8_T
            else:
                cpu_element_size = SIZEOF_FLOAT

            self.memory_cpu += self.calc_alignedbuffer_size(product(output_dims), cpu_element_size)
            self.memory_fxp += self.calc_alignedbuffer_size(product(output_dims), SIZEOF_UINT8_T)

        encoding_type = model.get_tf_encoding_type()
        if encoding_type == 'QMN':
            format_func = lambda encoding: "Q%d.%d" % encoding
            output_encoding = model.get_fxp_output_encoding(self.name)
            if output_encoding is not None:
                self.add_parm("output encoding", format_func(output_encoding))

            weight_extract_func = model.get_fxp_weight_encoding


        elif encoding_type == 'TF':
            format_func = lambda encoding: "min %.12f, max %.12f, delta %.12f, offset %.12f bitwidth %d" % encoding
            try:
                for i in range(len(self.output_names)):
                    encoding = model.get_tf_output_encoding_by_index(self.name, i)
                    if encoding is not None:
                        self.add_parm("output encoding", format_func(encoding))
            except:
                pass # no encoding was set for this layer.
            weight_extract_func = model.get_tf_weight_encoding
            bias_extract_func = model.get_tf_bias_encoding
        else:
            def weight_extract_func(name, i):
                raise NotImplemented()

            format_func = None

        # get any weight encodings
        try:
            i = 0
            # eventually i will be more than the number of weights the layer has,
            # and at that point weight_extract_func will throw an exception and
            # we will break out of the loop.
            while True:
                encoding = weight_extract_func(self.name, i)
                if encoding is not None:
                    axis = model.get_tf_weight_encoding_axis(self.name, i)
                    parm_name = "weight encoding" if i == 0 else "weight encoding %d" % i
                    self.add_parm(parm_name, format_func(encoding))
                    if axis >= 0:
                       num_elements = model.get_tf_weight_encoding_num_elements(self.name, i)
                       axis_quant_format_func = lambda axis, num_elements: "axis: %d, num_elements: %d" % (axis, num_elements)
                       self.add_parm("axis-quant", axis_quant_format_func(axis, num_elements))
                i += 1
        except:
            pass # no more encodings

        # get any bias encodings
        try:
            encoding = bias_extract_func(self.name)
            if encoding is not None:
                parm_name = "bias encoding"
                self.add_parm(parm_name, format_func(encoding))
        except:
            pass # no more encodings

        def extract_noop(layer):
            pass
        extractor = getattr(self, 'extract_%s' % layer['type'], extract_noop)
        extractor(layer)

    def dump(self, model, col_sizes, csv_content):
        print_row([str(self.id), self.name, self.type, self.get_input(0), self.get_output(0),
                   self.outputs_string(0), self.get_runtimes(), self.get_parm(0)], col_sizes, csv_content)

        extra_rows = max(len(self.input_names), len(self.parms))
        extra_rows = max( extra_rows, len(self.output_names) )

        for i in range(1,extra_rows):
            print_row(["","","", self.get_input(i), self.get_output(i), self.outputs_string(i), "", self.get_parm(i)],
                      col_sizes, csv_content)

    def outputs_string(self, idx):
        if idx >= len(self.output_dims_list):
            return ""
        else:
            return 'x'.join(map(str, self.output_dims_list[idx]))

    # calculate the size in memory that an alignedbuffer would need to be in order to
    # hold the given number of elements
    def calc_alignedbuffer_size(self, num, size, alignment=16):
        return (num + alignment) * size; # AlignedBuffer always allocs an extra 16 elements for alignment

    def id(self):
        return self.id

    def id_width(self):
        return len(str(self.id))

    def name(self):
        return self.name

    def name_width(self):
        return len(self.name)

    def type(self):
        return self.type

    def type_width(self):
        return len(self.type)

    def input_names(self):
        return self.input_names

    def input_width(self):
        return max(list(map(len,self.input_names))+[0])

    def output_names(self):
        return self.output_names

    def output_width(self):
        return max(list(map(len,self.output_names))+[0])

    def output_dims_width(self):
        return len(self.outputs_string(0))

    def parms_width(self):
        return max(list(map(len, self.parms))+[0])

    def runtimes_width(self):
        return len(self.get_runtimes())

    def get_parm(self, i):
        if i >= len(self.parms):
            return ""
        else:
            return self.parms[i]

    def get_parm_list(self):
        return self.parms

    def get_input(self,i):
        if i >= len(self.input_names):
            return ""
        else:
            return self.input_names[i]

    def get_input_list(self):
        return self.input_names

    def get_output(self,i):
        if i >= len(self.output_names):
            return ""
        else:
            return self.output_names[i]

    def get_output_list(self):
        return self.output_names

    def get_runtimes(self):
        runtimes_str = ""
        runtimes_str += "A " if self.valid_aip else "  "
        runtimes_str += "D " if self.valid_dsp else "  "
        runtimes_str += "G " if self.valid_gpu else "  "
        runtimes_str += "C" if self.valid_cpu else " "
        return runtimes_str

    def get_num_params(self):
        return self.param_count

    def get_macs(self):
        return self.macs

    # Gets the amount of memory needed to set up the layer in bytes
    def get_memory_cpu(self):
        return self.memory_cpu

    def get_memory_fxp(self):
        return self.memory_fxp

    def add_parm( self, key, val ):
        if type(val) is float:
            valstring = "%.4g" % val
        else:
            valstring = str(val)
        self.parms.append("%s: %s" % (key, valstring))

    def extract_batchnorm(self, layer):
        weights = layer['weights']
        self.add_parm("num channels", weights.shape[0])
        self.add_parm("compute stats", layer['compute_statistics'])
        self.add_parm("use mu sigma", layer['use_mu_sigma'])
        self.add_parm("across spatial", layer['across_spatial'])
        self.add_parm("normalize variance", layer['normalize_variance'])
        self.add_parm("epsilon", layer['epsilon'])

        self.param_count = weights.shape[0] * 2  # weights + bias (same shape of channel for each)
        input_tensor_size = len(self.input_dims[0])
        if(input_tensor_size == 1 or input_tensor_size == 2):
            self.nativeDimSize = 1
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        if(layer['compute_statistics']):
            if(layer['use_mu_sigma']):
                self.macs = product(native_output_dims)*5
                self.macs = product(native_output_dims)*5
            else:
                self.macs = product(native_output_dims)*3
        else:
            self.macs = product(native_output_dims)

        self.memory_cpu += self.calc_alignedbuffer_size(weights.shape[0], SIZEOF_FLOAT)
        self.memory_fxp += self.calc_alignedbuffer_size(weights.shape[0], SIZEOF_UINT8_T)

    def extract_constant(self, layer):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.param_count = product(native_output_dims)

    def extract_layernorm(self, layer):
        self.add_parm("axes", layer['axes'])
        self.add_parm("epsilon", layer['epsilon'])
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims) * 5  # 5 to account for mu and sigma calculation

        weights = layer['weights']
        self.memory_cpu += self.calc_alignedbuffer_size(weights.shape[0], SIZEOF_FLOAT)
        self.memory_fxp += self.calc_alignedbuffer_size(weights.shape[0], SIZEOF_UINT8_T)

    def extract_bbox_transform(self, layer):
        self.add_parm("weights", "%s" %( layer['weights']))
        self.add_parm("apply scale", layer['apply_scale'])
        self.add_parm("correct transform coords", layer['correct_transform_coords'])

    def extract_box_with_nms_limit(self, layer):
        self.add_parm("score_threshold", layer['score_threshold'])
        self.add_parm("nms_threshold", layer['nms_threshold'])
        self.add_parm("detections_per_im", layer['detections_per_im'])
        self.add_parm("soft_nms_enabled", layer['soft_nms_enabled'])
        self.add_parm("soft_nms_method", layer['soft_nms_method'])
        self.add_parm("soft_nms_sigma", layer['soft_nms_sigma'])
        self.add_parm("soft_nms_min_score_threshold", layer['soft_nms_min_score_threshold'])

    def extract_channel_shuffle(self, layer):
        for parm in [ "groups", "shuffle_type"]:
            self.add_parm(parm, layer[parm])

    def extract_cmrn(self, layer):
        for parm in [ "window_size", "alpha", "beta", "k"]:
            self.add_parm(parm, layer[parm])
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)*3

    def extract_convolutional(self, layer):
        self.add_parm("padding x", layer['padx'])
        self.add_parm("padding y", layer['pady'])
        if 'padding_mode' in layer:
            self.add_parm("padding mode", padding_mode_name(layer['padding_mode']))
        self.add_parm("stride x", layer['stridex'])
        self.add_parm("stride y", layer['stridey'])
        if 'dilationx' in layer:
            self.add_parm("dilation x", layer['dilationx'])
        if 'dilationy' in layer:
            self.add_parm("dilation y", layer['dilationy'])
        weights = layer['weights']
        self.add_parm("num filters", weights.shape[3])
        self.add_parm("kernel", "%dx%d" %  weights.shape[0:2])

        if 'groups' in layer:
            self.add_parm("groups", layer['groups'])

        self.param_count = product(weights.shape)
        self.param_count += weights.shape[3] # biases
        native_input_dims = self.input_dims[0][-self.nativeDimSize:]
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]

        # calculate size of expanded buffer
        conv_image_width = weights.shape[0] * weights.shape[1] * native_input_dims[2]
        rounding_amount = 32;
        expanded_stride = ((conv_image_width + rounding_amount - 1) / rounding_amount) * rounding_amount;
        expanded_height = native_output_dims[0] * native_output_dims[1];
        expanded_buffer_size = expanded_height * expanded_stride

        # calculate size of weight array
        weights_size = product(weights.shape[0:4])

        # = filter size * number of filter positions
        self.macs = product(weights.shape[0:3])*product(native_output_dims)

        self.memory_cpu += self.calc_alignedbuffer_size(expanded_buffer_size, SIZEOF_FLOAT) + self.calc_alignedbuffer_size(weights_size, SIZEOF_FLOAT)
        self.memory_fxp += self.calc_alignedbuffer_size(expanded_buffer_size, SIZEOF_UINT8_T) + self.calc_alignedbuffer_size(weights_size, SIZEOF_UINT8_T)

    def extract_correlation(self, layer):
        self.add_parm('displacement', layer['displacement'])
        self.add_parm('shift', layer['shift'])
        self.add_parm('stride', layer['stride'])

    def extract_crop(self, layer):
        for i, o in enumerate(layer['offsets']):
            self.add_parm('offsets[%d]' % i, o)
        if 'counts' in layer:
            for i, o in enumerate(layer['counts']):
                self.add_parm('counts[%d]' % i, o)

    def extract_data(self, layer):
        encode_in = color_space_name(layer["image_encoding_in"])
        encode_out = color_space_name(layer["image_encoding_out"])
        if encode_in == encode_out:
            self.add_parm("input_preprocessing", "passthrough" )
        else:
            self.add_parm("input_encoding_in", encode_in )
            self.add_parm("input_encoding_out", encode_out )
        input_type = input_type_name(layer["input_type"])
        self.add_parm("input_type", input_type)

    def extract_deconvolution(self, layer):
        self.extract_convolutional(layer)

        # add deconv specific output padding
        self.add_parm("output padding x", layer['output_paddingx'])
        self.add_parm("output padding y", layer['output_paddingy'])

        # for deconvolution, macs are computed off number of input positions.
        native_input_dims = self.input_dims[0][-self.nativeDimSize:]
        input_size = product(native_input_dims)
        weight_dim = layer['weights'].shape

        weights_size = product(weight_dim[0:4])

        self.macs = weight_dim[0]*weight_dim[1]*weight_dim[3]*input_size/layer.get('groups', 1)

        self.memory_cpu += self.calc_alignedbuffer_size(weights_size, SIZEOF_FLOAT)
        self.memory_fxp += self.calc_alignedbuffer_size(weights_size, SIZEOF_UINT8_T)
        self.param_count = weights_size + (weight_dim[3] * layer.get('groups', 1))  # weights_shape + bias_shape

    def extract_dropout(self, layer):
        self.add_parm("keep", layer["keep"])

    def extract_dynamicconvolutional(self, layer):
        self.add_parm("padding x", layer['padx'])
        self.add_parm("padding y", layer['pady'])
        if 'padding_mode' in layer:
            self.add_parm("padding mode", padding_mode_name(layer['padding_mode']))
        self.add_parm("stride x", layer['stridex'])
        self.add_parm("stride y", layer['stridey'])
        if 'dilationx' in layer:
            self.add_parm("dilation x", layer['dilationx'])
        if 'dilationy' in layer:
            self.add_parm("dilation y", layer['dilationy'])
        weights = layer['weights']
        self.add_parm("num filters", weights.shape[3])
        self.add_parm("kernel", "%dx%d" %  weights.shape[0:2])

        if 'groups' in layer:
            self.add_parm("groups", layer['groups'])

        self.param_count = product(weights.shape)
        self.param_count += weights.shape[3] # biases
        native_input_dims = self.input_dims[0][-self.nativeDimSize:]
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]

        # calculate size of expanded buffer
        conv_image_width = weights.shape[0] * weights.shape[1] * native_input_dims[2]
        rounding_amount = 32;
        expanded_stride = ((conv_image_width + rounding_amount - 1) / rounding_amount) * rounding_amount;
        expanded_height = native_output_dims[0] * native_output_dims[1];
        expanded_buffer_size = expanded_height * expanded_stride

        # calculate size of weight array
        weights_size = product(weights.shape[0:4])

        # = filter size * number of filter positions
        self.macs = product(weights.shape[0:3])*product(native_output_dims)

        self.memory_cpu += self.calc_alignedbuffer_size(expanded_buffer_size, SIZEOF_FLOAT) + self.calc_alignedbuffer_size(weights_size, SIZEOF_FLOAT)
        self.memory_fxp += self.calc_alignedbuffer_size(expanded_buffer_size, SIZEOF_UINT8_T) + self.calc_alignedbuffer_size(weights_size, SIZEOF_UINT8_T)

    def extract_elementwise_op(self, layer):
        self.add_parm("operation", layer["op_name"])

    def extract_elementwise_binary_op(self, layer):
        self.add_parm("operation", layer["op_name"])
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_elementwise_select(self, layer):
        self.add_parm("operation", layer["op_name"])

    def extract_elementwise_unary_op(self, layer):
        self.add_parm("operation", layer["op_name"])

    def extract_extract_glimpse(self, layer):
        self.add_parm("glimpse_width", layer["glimpse_width"])
        self.add_parm("glimpse_height", layer["glimpse_height"])
        self.add_parm("centered", layer["centered"])
        self.add_parm("normalized", layer["normalized"])
        self.add_parm("uniform_noise", layer["uniform_noise"])

    def extract_fully_connected(self, layer):
        self.param_count = layer['bias'].shape[0]
        for weights in layer['weights_list']:
            weights_size = product(weights.shape)

            self.param_count += weights_size
            self.macs += weights_size

            self.memory_cpu += self.calc_alignedbuffer_size(weights_size, SIZEOF_FLOAT)
            self.memory_fxp += self.calc_alignedbuffer_size(weights_size, SIZEOF_UINT8_T)

    def extract_l2_norm(self, layer):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims) * 2  # 2 since summation is squared

    def extract_matmul(self, layer):
        self.param_count = layer['bias'].shape[0]
        # macs is m * n * k, where m and n are outer dims after transpose and k is the inner common dim
        k = self.input_dims[0][-1]
        if layer['transpose_a']:
            k = self.input_dims[0][-2]
        self.macs = product(self.input_dims[0][:-2]) * product(self.input_dims[1][:-2]) * k
        self.add_parm("transpose_a", layer['transpose_a'])
        self.add_parm("transpose_b", layer['transpose_b'])

    def extract_gate_tx_gru(self, layer):
        self.add_parm("gate activation", neuron_name(layer['gate_activation']))
        self.add_parm("rec activation", neuron_name(layer['rec_activation']))
        self.add_parm("rec gate activation", neuron_name(layer['rec_gate_activation']))
        self.add_parm("weight shape", layer['gate_weights']['forward_input_to_state'].shape)
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)*2

    def extract_generate_proposals(self, layer):
        self.add_parm("spatial scale", layer['spatial_scale'])
        self.add_parm("pre nms top N", layer['pre_nms_top_n'])
        self.add_parm("post nms top N", layer['post_nms_top_n'])
        self.add_parm("nms thresh", layer['nms_thresh'])
        self.add_parm("min size", layer['min_size'])
        self.add_parm("correct transform coords", layer['correct_transform_coords'])

    def extract_image_projective_transform(self, layer):
        self.add_parm("interpolation", layer['interpolation'])

    def extract_local_response_norm(self, layer):
        self.extract_cmrn(layer)
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)*layer['window_size']*layer['window_size']

    def extract_lstm(self, layer):
        x_weights = layer['x_gate_weights']
        self.add_parm("x weights", x_weights.shape)
        self.param_count += product(x_weights.shape)
        h_weights = layer['h_gate_weights']
        self.add_parm("h weights", h_weights.shape)
        self.param_count += product(h_weights.shape)
        bias = layer['gate_biases']
        self.param_count += product(bias.shape)
        self.add_parm("biases",    bias.shape)

        if 'x_static_gate_weights' in layer:
            x_static_weights = layer['x_static_gate_weights']
            self.param_count += product(x_static_weights.shape)
            self.add_parm("x_static weights", x_static_weights.shape)

        self.add_parm("backward",  layer['backward'])
        input_features = self.input_dims[0][-1]
        output_features = self.output_dims_list[0][-1]
        steps = self.output_dims_list[0][-2]
        self.macs = 4*input_features*output_features*steps
        self.macs += 4*output_features*output_features*steps
        self.macs += 3*output_features*steps

    def extract_neuron(self, layer):

        self.add_parm("a", layer['a'])
        self.add_parm("b", layer['b'])
        self.add_parm("min_clamp", layer['min_clamp'])
        self.add_parm("max_clamp", layer['max_clamp'])
        self.add_parm("func", neuron_name(layer['func']))

    def extract_onehot(self, layer):

        self.add_parm("depth", layer['depth'])
        self.add_parm("axis", layer['axis'])
        self.add_parm("on_value", layer['on_value'])
        self.add_parm("off_value", layer['off_value'])

    def extract_pooling(self, layer):
        self.add_parm("pool size x", "%s" %(layer['pool_size_x']))
        self.add_parm("pool size y", "%s" %(layer['pool_size_y']))
        self.add_parm("stride x","%s" %(layer['pool_stride_x']))
        self.add_parm("stride y","%s" %(layer['pool_stride_y']))
        self.add_parm("padding x", "%s" %(layer['pad_x']))
        self.add_parm("padding y", "%s" %(layer['pad_y']))
        if layer['pool_type'] == modeltools.POOL_MAX:
            pool_type = "POOL_MAX"
        else:
            pool_type = "POOL_AVG"
            native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
            self.macs = product(native_output_dims) * layer['pool_size_x'] * layer['pool_size_y']
        self.add_parm("pool_type", pool_type)

    def extract_prelu(self, layer):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)
        self.param_count = product(layer["coeff"].shape)

    def extract_proposal(self, layer):
        self.add_parm("feat_stride", "%s" %( layer['feat_stride']))
        self.add_parm("scales", "%s" %( layer['scales']))
        self.add_parm("ratios", "%s" %( layer['ratios']))
        self.add_parm("anchor_base_size", "%s" %( layer['anchor_base_size']))
        self.add_parm("min_bbox_size", "%s" %( layer['min_bbox_size']))
        self.add_parm("max_num_proposals", "%s" %( layer['max_num_proposals']))
        self.add_parm("max_num_rois", "%s" %( layer['max_num_rois']))
        self.add_parm("iou_threshold_nms", "%s" %( layer['iou_threshold_nms']))

    def extract_power(self, layer):
        self.add_parm("scale", "%s" %( layer['scale']))
        self.add_parm("shift", "%s" %( layer['shift']))
        self.add_parm("power", "%s" %( layer['power']))

    def extract_reduce(self, layer):

        self.add_parm("operation", "%s" %( layer['op_name']))

    def extract_roialign(self, layer):
        def add(parm):
            self.add_parm(parm, layer[parm])
        add('spatial_scale')
        add('pooled_h')
        add('pooled_w')
        add('sampling_ratio')
        add('implode_batch')
        if layer['implode_batch']:
            add('tiled_batch_h')
            add('tiled_batch_w')
            add('batch_pad_h')
            add('batch_pad_w')
            add('padvalue')
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_roipooling(self, layer):
        self.add_parm("pooled size", "%sx%s" %( layer['pooled_size_w'],layer['pooled_size_h']))
        self.add_parm("spatial scale", "%s" %( layer['spatial_scale']))
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_scaling(self, layer):
        self.add_parm("pad_value", layer['pad'])
        self.add_parm("maintain_aspect_ratio", layer['maintain_aspect_ratio'])
        self.add_parm("align_corners", layer['align_corners'])
        self.add_parm("half_pixel_centers", layer['half_pixel_centers'])
        if 'resize_mode' in layer:
            self.add_parm("resize_mode", resize_mode_name(layer['resize_mode']))
        if 'scale_height' in layer:
            self.add_parm("scale_height", layer['scale_height'])
        if 'scale_width' in layer:
            self.add_parm("scale_width", layer['scale_width'])
        if layer['resize_mode'] == modeltools.RESIZE_BILINEAR:
            native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
            self.macs = product(native_output_dims)

    def extract_user_defined(self, layer):
        self.add_parm("blob_size", len(layer['blob']))

    def extract_udo(self, layer):
        for attr in layer.keys():
            parm = layer[attr]
            # done this way to avoid numpy import
            if parm.__class__.__name__ == 'ndarray':
                # set to 80 because that is an acceptable readable length for terminals
                # TODO: maybe this shouldn't even be printed
                if len(str(parm)) > 80:
                    self.add_parm(attr + "_type", 'np.ndarray')
                    self.add_parm(attr + '_dims', list(parm.shape))
                    parm = 'too big to print'
                else:
                    parm = str(parm)
            if not hasattr(self, attr):
                self.add_parm(attr, parm)

    def extract_scale(self, layer):
        self.add_parm("axis", layer['axis'])
        self.add_parm("num_axes", layer['num_axes'])
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)
        weights = layer['weights']
        self.param_count = weights.shape[0] * 2  # weights + bias (same shape of channel for each)

    def extract_slice(self, layer):
        self.add_parm("slice_axis", layer['slice_axis'])
        self.add_parm("slice_points", layer['slice_points'])

    def extract_strided_slice(self, layer):
        self.add_parm("begin", layer['begin'])
        self.add_parm("end", layer['end'])
        self.add_parm("strides", layer['strides'])
        self.add_parm("shrink_axis_mask", layer['shrink_axis_mask'])
        self.add_parm("begin_mask", layer['begin_mask'])
        self.add_parm("end_mask", layer['end_mask'])
        self.add_parm("new_axis_mask", layer['new_axis_mask'])

    def extract_pad(self, layer):
        mode_name = padding_mode_name(layer['padding_mode'])
        self.add_parm("paddings", layer['paddings'])
        self.add_parm("mode", mode_name)
        if mode_name == 'constant':
            self.add_parm("constant_values", layer['constant_values'])
        if mode_name == 'reflect':
            self.add_parm("constant_values", 0)

    def extract_argmax(self, layer):
        self.add_parm("op", layer['op'])
        self.add_parm("axis", layer['axis'])

    def extract_reduction(self, layer):
        self.add_parm("axes", layer['axes'])
        self.add_parm("keep_dims", layer['keep_dims'])

    def extract_prod(self, layer):
        self.add_parm("axes", layer['axes'])
        self.add_parm("keep_dims", layer['keep_dims'])

    def extract_tile(self, layer):
        self.add_parm("multiples", layer['multiples'])

    def extract_permute(self, layer):
        self.add_parm("permute_order", layer['permute_order'])
        native_output_dims = self.output_dims_list[0][:]

    def extract_ssd_detection_output(self, layer):
        self.add_parm("num_ classes", layer['num_classes'])
        self.add_parm("share_location", layer['share_location'])
        self.add_parm("background_label_id", layer['background_label_id'])
        self.add_parm("nms_top_k", layer['nms_top_k'])
        self.add_parm("nms_eta", layer['nms_eta'])
        self.add_parm("nms_threshold", layer['nms_threshold'])
        self.add_parm("keep_top_k", layer['keep_top_k'])
        self.add_parm("variance_encoded_in_target", layer['variance_encoded_in_target'])
        self.add_parm("confidence_threshold", layer['confidence_threshold'])

        if 'prior_box_code_type' in layer:
            self.add_parm("prior box type", prior_box_name(layer['prior_box_code_type']))

    def extract_detection_output(self, layer):
        if 'delta_scaling_factors' in layer:
            self.add_parm("delta_scaling_factors", layer['delta_scaling_factors'])
        self.add_parm("confidence_threshold", layer['confidence_threshold'])
        self.add_parm("iou_threshold", layer['iou_threshold'])
        self.add_parm("nms_type", nms_type_name(layer['nms_type']))
        self.add_parm("background_class_idx", layer['background_class_idx'])
        self.add_parm("use_bg_in_nms", layer['use_bg_in_nms'])
        self.add_parm("output_background", layer['output_background'])
        self.add_parm("share_location", layer['share_location'])
        self.add_parm("nms_eta", layer['nms_eta'])
        self.add_parm("detection_limit", layer['detection_limit'])
        self.add_parm("keep_top_k", layer["keep_top_k"])

    def extract_concatenation(self, layer):
        self.add_parm("axis", layer['axis'])

    def extract_pixel_shuffle(self, layer):
        self.add_parm('upscale factor', layer['upscale_factor'])

    def extract_crop_and_resize(self, layer):
        self.add_parm("crop_height", layer["crop_height"])
        self.add_parm("crop_width", layer["crop_width"])
        self.add_parm("interpolation_method", layer["interpolation_method"])
        self.add_parm("extrapolation_value", layer["extrapolation_value"])

    def extract_moments(self, layer):
        self.add_parm("axes", layer['axes'])
        self.add_parm("keep_dims", layer['keep_dims'])

    def extract_gather(self, layer):
        self.add_parm("axis", layer['axis'])

    def extract_pack(self, layer):
        self.add_parm("axis", layer['axis'])

    def extract_unpack(self, layer):
        self.add_parm("axis", layer['axis'])
        self.add_parm("num outputs", layer['num'])

    def extract_space_to_depth(self, layer):
        self.add_parm('downscale factor', layer['downscale_factor'])


class ModelInfo(object):
    def __init__(self):
        self.model = modeltools.Model()
        self.model_filename = ""
        self.rows = []
        self.layer_mapping = {}
        self.total_params = 0
        self.total_macs = 0
        self.total_memory_cpu = 0
        self.total_memory_fxp = 0
        self.total_layer_types = set()

    def load(self, input_file_name):
        self.model.load(input_file_name)
        self.model_filename = input_file_name

    def extract_model_info(self, input_file_name):
        self.load(input_file_name)

        layers = self.model.get_layers()
        for layer in layers:
            row = LayerRow(layer, self.rows, self.model)
            self.rows.append(row)
            self.total_params += row.get_num_params()
            self.total_macs += row.get_macs()
            self.total_memory_cpu += row.get_memory_cpu()
            self.total_memory_fxp += row.get_memory_fxp()
            layer_type = str(layer['type']).upper()
            if layer_type == "NEURON":
                self.total_layer_types.add(layer_type + "_" + str(neuron_name(layer['func'])).upper())
            elif layer_type == "SCALING" and 'resize_mode' in layer:
                self.total_layer_types.add(layer_type + "_" + resize_mode_name(layer['resize_mode']).upper())
            elif layer_type == "ELEMENTWISE_OP" or layer_type == "ELEMENTWISE_BINARY_OP" or \
                 layer_type == "ELEMENTWISE_UNARY_OP" or layer_type == "ELEMENTWISE_SELECT" or \
                 layer_type == "REDUCE":
                self.total_layer_types.add(layer_type + "_" + str(layer["op_name"]).upper())
            else:
                self.total_layer_types.add(str(layer['type']).upper())
        return self.rows

    def dump_info(self, show_memory_usage, input_file_name, output_file_name):

        """
        Dump information about the given DLC file.

        If output_file_name is not None, then the info is written as CSV to that file.
        If it is none, then the dump is printed to the screen.
        """

        self.extract_model_info(input_file_name)

        global csv_file_flag
        csv_content = []
        if output_file_name is not None:
            csv_file_flag = True

        print_value('DLC info for: ' + os.path.abspath(input_file_name), csv_content)

        headers = ["Id", "Name", "Type", "Inputs", "Outputs", "Out Dims", "Runtimes", "Parameters"]
        col_sizes = [1+len(header) for header in headers]

        for row in self.rows:
            if row.param_count > 0:
                row.add_parm( "param count", get_si_notation(row.param_count, self.total_params))

            if row.macs > 0:
                row.add_parm( "MACs per inference", get_si_notation(row.macs, self.total_macs))

            if row.layer_affinity != 'UNSET':
                row.add_parm( "Layer Affinity", row.layer_affinity)
            if show_memory_usage:
                if row.get_memory_cpu() > 0:
                    row.add_parm( "Memory needed", get_binary_si_notation(row.get_memory_cpu(), self.total_memory_cpu))

        for row in self.rows:
            col_sizes[0] = max(col_sizes[0], 1+row.id_width())
            col_sizes[1] = max(col_sizes[1], 1+row.name_width())
            col_sizes[2] = max(col_sizes[2], 1+row.type_width())
            col_sizes[3] = max(col_sizes[3], 1+row.input_width())
            col_sizes[4] = max(col_sizes[4], 1+row.output_width())
            col_sizes[5] = max(col_sizes[5], 1+row.output_dims_width())
            col_sizes[6] = max(col_sizes[6], 1+row.runtimes_width())
            col_sizes[7] = max(col_sizes[7], 1+row.parms_width())

        (model_version, total_params_str, total_macs_str, converter_command, quantizer_command,
         converter_version, model_copyright) = self.get_meta_data(self.total_params, self.total_macs, input_file_name)
        memory_data = self.get_memory_data(self.total_memory_cpu, self.total_memory_fxp, self.total_params, show_memory_usage)

        print_value(model_version, csv_content)
        print_value(model_copyright, csv_content)
        total_size = 2+2*len(col_sizes)-1+sum(col_sizes)
        print_value('-'*total_size)
        print_row(headers, col_sizes, csv_content)
        print_value('-'*total_size)

        for row in self.rows:
            row.dump(self.model, col_sizes, csv_content)
        print_value('-'*total_size)
        print_value("Note: The supported runtimes column assumes a processor target of Snapdragon 835 (8998)", csv_content)
        print_value("Key : A:AIP\n      D:DSP\n      G:GPU\n      C:CPU\n", csv_content)

        # Add unique layer types to Meta data printing for dlc-info
        layer_types = "Layers used by DLC: {}".format(", ".join(sorted(self.total_layer_types)))
        print_value(total_params_str + '\n' + total_macs_str + '\n' + converter_command + '\n' + quantizer_command+ '\n' + converter_version +
                    "\n" + layer_types, csv_content)

        print_value('Est. Steady-State Memory Needed to Run: ' + '\n'.join(memory_data), csv_content)

        print_value(('-' * total_size) + '\n')

        if self.is_aix_enabled():
            try:
                aix_records = self.get_aix_records()
                if aix_records:
                    self.dump_aix_info(aix_records, csv_content)
            except Exception as e:
                raise Exception("Error Querying for AIP Records in model:", self.model_filename, e)

        if self.is_backend_cache_enabled():
            try:
                backend_cache_records = self.get_backend_cache_records()
                if backend_cache_records:
                    self.dump_cache_info(backend_cache_records, csv_content)
            except Exception as e:
                raise Exception("Error Querying for Backend Cache Records in model:", self.model_filename, e)

        if output_file_name is not None:
            try:
                with open(output_file_name, 'w') as csv_file:
                    writer = csv.writer(csv_file)
                    for d in csv_content:
                        writer.writerow(d)

            except IOError:
                print("IOError: Cannot open CSV file " + output_file_name, file=sys.stderr)
                sys.exit(-1)

    def dump_runtime_info(self):
        count = 1
        for row in self.rows:
            print("Layer %2d\t" % count + str(row.valid_cpu) + " | " + str(row.valid_gpu) + " | " + str(row.valid_dsp) + " | " + str(row.valid_aip))
            print("-" * 60)
            count = count + 1

    def dump_cache_info(self, cache_records, csv_content):
        """
            Prints Aix table with each record and its corresponding meta information
        @param aix_records: dictionary of the aix records and their meta info
        """

        print_value("\nCache Info:", csv_content)
        headers = ["Cache Record Name", "snpe_version", "record_version", "backend_record_search_key",
                   "backend_type", "record_size", "Subnets"]
        col_sizes = [1 + len(header) for header in headers]

        max_col_size = 25
        for s in range(0, len(col_sizes)):
            col_sizes[s] = max(col_sizes[s], max_col_size)

        subnet_col_max_size = 40
        col_sizes[-1] = max(col_sizes[-1], subnet_col_max_size)  # to account for long buffer names

        total_size = 2 + 2 * len(col_sizes) - 1 + sum(col_sizes)
        print_value('-'*total_size)
        print_row(headers, col_sizes, csv_content)
        print_value('-'*total_size)
        for cache_record_name, cache_meta_info in cache_records.items():
            cache_meta_list = [cache_record_name]  # add the record name column

            # add everything after name but before Subnets(since Subnets have further info)
            for i in range(1, len(headers) - 1):
                cache_meta_list.append(cache_meta_info[headers[i]])

            subnet_col = "num_of_subnets: " + str(cache_meta_info['num_of_subnets'])
            cache_meta_list.append(subnet_col)
            print_row(cache_meta_list, col_sizes, csv_content)

            # Add subnets meta info for record
            if cache_meta_info['compatibility']:
                for j in range(0, cache_meta_info['num_of_subnets']):
                    subnet_name = "subnet_" + str(j)
                    print_row(["", "", "", "", "", subnet_name + ":"], col_sizes, csv_content)
                    # note: separated if cases for start/end ids so that they get printed one after the other for
                    #        better visual. Python was ordering them randomly even if OrderedDict was used.
                    if "start_layer_Id" in cache_meta_info[subnet_name].keys():
                        subnet_col = "start_layer_Id: " + str(cache_meta_info[subnet_name]["start_layer_Id"])
                        print_row(["", "", "", "", "", "  " + subnet_col], col_sizes, csv_content)
                        cache_meta_info[subnet_name].pop("start_layer_Id")
                    if "end_layer_Id" in cache_meta_info[subnet_name].keys():
                        subnet_col = "end_layer_Id: " + str(cache_meta_info[subnet_name]["end_layer_Id"])
                        print_row(["", "", "", "", "", "  " + subnet_col], col_sizes, csv_content)
                        cache_meta_info[subnet_name].pop("end_layer_Id")
                    for subnet_key, subnet_value in cache_meta_info[subnet_name].items():
                        subnet_col = subnet_key + ": "
                        if isinstance(subnet_value, list):
                            print_row(["", "", "", "", "", "  " + subnet_col], col_sizes, csv_content)
                            for value in subnet_value:
                                value = "    " + value  # indent for visual
                                print_row(["", "", "", "", "", value], col_sizes, csv_content)
                        else:
                            subnet_col += str(subnet_value)
                            print_row(["", "", "", "", "", "  " + subnet_col], col_sizes, csv_content)
            else:
                # print warning if record is not compatible with current version of snpe
                err_msg = '    Warning: This record (' + cache_record_name + ') is incompatible with the latest version of SNPE.'
                remaining_col_len = total_size - len(err_msg) - 2  # to find where to put the trailing '|' for row
                # print error for record indented and in red color
                print_value('|' + "\033[91m{}\033[00m" .format(err_msg) + (" " * remaining_col_len) + '|', csv_content)

    def dump_aix_info(self, aix_records, csv_content):
        """
            Prints Aix table with each record and its corresponding meta information
        @param aix_records: dictionary of the aix records and their meta info
        """

        print_value("\nAIP Info:", csv_content)
        headers = ["AIP Record Name", "nnc_version", "record_version", "hta_blob_id", "record_size", "Subnets"]
        col_sizes = [1 + len(header) for header in headers]

        max_col_size = 25
        for s in range(0, len(col_sizes)):
            col_sizes[s] = max(col_sizes[s], max_col_size)

        subnet_col_max_size = 40
        col_sizes[-1] = max(col_sizes[-1], subnet_col_max_size)  # to account for long buffer names

        total_size = 2 + 2 * len(col_sizes) - 1 + sum(col_sizes)
        print_value('-'*total_size)
        print_row(headers, col_sizes, csv_content)
        print_value('-'*total_size)
        for aix_record_name, aix_meta_info in aix_records.items():
            aix_meta_list = [aix_record_name]  # add the record name column

            # add everything after name but before Subnets(since Subnets have further info)
            for i in range(1, len(headers) - 1):
                aix_meta_list.append(aix_meta_info[headers[i]])

            subnet_col = "num_of_subnets: " + str(aix_meta_info['num_of_subnets'])
            aix_meta_list.append(subnet_col)
            print_row(aix_meta_list, col_sizes, csv_content)

            # Add subnets meta info for record
            if aix_meta_info['compatibility']:
                for j in range(0, aix_meta_info['num_of_subnets']):
                    subnet_name = "subnet_" + str(j)
                    print_row(["", "", "", "", "", subnet_name + ":"], col_sizes, csv_content)
                    # note: separated if cases for start/end ids so that they get printed one after the other for
                    #        better visual. Python was ordering them randomly even if OrderedDict was used.
                    if "start_layer_Id" in aix_meta_info[subnet_name].keys():
                        subnet_col = "start_layer_Id: " + str(aix_meta_info[subnet_name]["start_layer_Id"])
                        print_row(["", "", "", "", "", "  " + subnet_col], col_sizes, csv_content)
                        aix_meta_info[subnet_name].pop("start_layer_Id")
                    if "end_layer_Id" in aix_meta_info[subnet_name].keys():
                        subnet_col = "end_layer_Id: " + str(aix_meta_info[subnet_name]["end_layer_Id"])
                        print_row(["", "", "", "", "", "  " + subnet_col], col_sizes, csv_content)
                        aix_meta_info[subnet_name].pop("end_layer_Id")
                    for subnet_key, subnet_value in aix_meta_info[subnet_name].items():
                        subnet_col = subnet_key + ": "
                        if isinstance(subnet_value, list):
                            print_row(["", "", "", "", "", "  " + subnet_col], col_sizes, csv_content)
                            for value in subnet_value:
                                value = "    " + value  # indent for visual
                                print_row(["", "", "", "", "", value], col_sizes, csv_content)
                        else:
                            subnet_col += str(subnet_value)
                            print_row(["", "", "", "", "", "  " + subnet_col], col_sizes, csv_content)
            else:
                # print warning if record is not compatible with current version of snpe
                err_msg = '    Warning: This record(' + aix_record_name + ') is incompatible with the latest version of SNPE.'
                remaining_col_len = total_size - len(err_msg) - 2  # to find where to put the trailing '|' for row
                # print error for record indented and in red color
                print_value('|' + "\033[91m{}\033[00m" .format(err_msg) + (" " * remaining_col_len) + '|', csv_content)

    def read_converter_command(self, dlc_file_path):
        archive = zipfile.ZipFile(dlc_file_path, 'r')
        if 'dlc.metadata' in archive.namelist():
            meta = archive.read('dlc.metadata').decode()
            for k, v in [line.split('=', 1) for line in meta.split('\n') if len(line) > 0]:
                if k == 'converter-command':
                    return v
        return 'N/A'

    def read_quantizer_command(self, dlc_file_path):
        archive = zipfile.ZipFile(dlc_file_path, 'r')
        if 'dlc.metadata' in archive.namelist():
            meta = archive.read('dlc.metadata').decode()
            for k, v in [line.split('=', 1) for line in meta.split('\n') if len(line) > 0]:
                if k == 'quantizer-command':
                    return v
        return 'N/A'

    def is_aix_enabled(self):
        return self.model.is_aix_enabled()

    def is_backend_cache_enabled(self):
        return self.model.is_backend_cache_enabled()

    def is_aix_record_present(self):
        archive = zipfile.ZipFile(self.model_filename, 'r')
        archive_filelist = archive.filelist
        for fileinfo in archive_filelist:
            if "aip" in fileinfo.filename or "hta" in fileinfo.filename or "aix" in fileinfo.filename:
                return True
        return False

    def get_aix_records(self):
        return self.model.get_aix_records()

    def get_backend_cache_records(self):
        return self.model.get_backend_cache_records()

    def read_converter_version(self, dlc_file_path):
        archive = zipfile.ZipFile(dlc_file_path, 'r')
        if 'dlc.metadata' in archive.namelist():
            meta = archive.read('dlc.metadata').decode()
            for k, v in [line.split('=', 1) for line in meta.split('\n') if len(line) > 0]:
                if k == 'converter-version':
                    return v
        return 'N/A'

    def get_layer_mapping(self):

        if self.layer_mapping == {}:
            layers = self.model.get_layers()
            for layer in layers:
                row  = LayerRow(layer, self.rows, self.model)
                self.layer_mapping[row.id] = row.name

        return self.layer_mapping

    def get_model_copyright(self):
        return self.model.get_model_copyright()

    def get_meta_data(self, total_params, total_macs, input_file_name):

        model_version = 'Model Version: %s' %self.model.get_model_version()
        total_params = 'Total parameters: %d (%d MB assuming single precision float)' %(total_params, total_params*4/(1024*1024))
        total_macs = 'Total MACs per inference: %s' %get_si_notation(total_macs, total_macs)
        converter_command = 'Converter command: {}'.format(self.read_converter_command(input_file_name))
        quantizer_command = 'Quantizer command: {}'.format(self.read_quantizer_command(input_file_name))
        converter_version = 'DLC created with converter version: {}'.format(self.read_converter_version(input_file_name))
        model_copyright = 'Model Copyright:{}'.format(self.get_model_copyright())

        return model_version, total_params, total_macs, converter_command, quantizer_command, converter_version, model_copyright

    def get_memory_data(self, total_memory_cpu, total_memory_fxp, total_params, show_full_usage):

        # sizes of fixed allocations in SNPE, measured with Massif
        symphony_fixed_allocs = 23068672
        hogl_fixed_allocs = 67141632 + 3145728

        mem_linux_cpu = total_memory_cpu + hogl_fixed_allocs + symphony_fixed_allocs
        mem_linux_fxp = total_memory_fxp + hogl_fixed_allocs + symphony_fixed_allocs
        mem_android_cpu = total_memory_cpu + symphony_fixed_allocs

        lines = []

        if show_full_usage:

            # print full usage data
            lines.append('') # start on new line
            lines.append('- On Linux CPU: %s' %get_binary_si_notation(mem_linux_cpu, None))
            lines.append('- On Linux CPU Quantized to 8-bit: %s' %get_binary_si_notation(mem_linux_fxp, None))
            lines.append('- On Android CPU: %s' %get_binary_si_notation(mem_android_cpu, None))
        else:

            # print abridged usage data
            lines.append('%s' %get_binary_si_notation(total_memory_cpu, None))

        return lines

    def get_input_dims(self):
        row = self.rows[0]
        return row.outputs_string(0)

    def get_total_macs(self):
        return self.total_macs

    def get_total_params(self):
        return self.total_params

    def types_info(self):
        name_and_type = {}
        for row in self.rows:
            name_and_type.update({row.name:row.type})
        return name_and_type

    def types_info_by_id(self):
        id_and_type = {}
        for row in self.rows:
            id_and_type.update({row.id: row.type})
        return id_and_type

    def layer_names_by_id(self):
        id_and_type = {}
        for row in self.rows:
            id_and_type.update({row.id: row.name})
        return id_and_type

    def ids_layer(self):
        name_and_id = {}
        for row in self.rows:
            name_and_id.update({row.name: row.id})
        return name_and_id

    def params_info(self):
        name_and_parm = OrderedDict()
        for row in self.rows:
            m = max(len(row.get_parm_list()), len(row.get_input_list()))
            m = max(m,len(row.get_output_list()))
            parms = []
            for i in range(0,m-1):
                parms.append(row.get_parm(i))
            name_and_parm.update({row.name:parms})
        return name_and_parm

    def dims_info(self):
        name_and_dims = OrderedDict()
        for row in self.rows:
            output_dims = ['x'.join(map(str, dim)) for dim in row.output_dims_list]
            name_and_dims.update({row.name: output_dims})
        return name_and_dims

    def weights_info(self):
        name_and_weights = OrderedDict()
        for row in self.rows:
            layer = row.layer
            name_and_weights.update({row.name: layer.get('weights')})
        return name_and_weights

    def get_output_names(self):
        output_names = OrderedDict()
        for row in self.rows:
            output_names.update({row.name: row.get_output_list()})
        return output_names
