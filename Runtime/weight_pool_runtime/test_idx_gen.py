# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:40:46 2021

@author: shuru
"""

import os
import math
import numpy as np
import torch

def write_c_common_header(f):
    #f.write("{}\n\n".format(LICENSE))
    #f.write(self.tensor_flow_reference_version)
    f.write("#pragma once\n")

def write_c_header_offsets(f, prefix):
    f.write("#define {}_INPUT_OFFSET {}\n".format(prefix, input_zero_point))
    f.write("#define {}_OUTPUT_OFFSET {}\n".format(prefix, output_zero_point))

def write_c_config_header(write_common_parameters=True):
    filename = self.config_data

    self.generated_header_files.append(filename)
    filepath = self.headers_dir + filename

    prefix = self.testdataset.upper()

    print("Writing C header with config data {}...".format(filepath))
    with open(filepath, "w+") as f:
        self.write_c_common_header(f)
        if (write_common_parameters):
            f.write("#define {}_OUT_CH {}\n".format(prefix, self.output_ch))
            f.write("#define {}_IN_CH {}\n".format(prefix, self.input_ch))
            f.write("#define {}_INPUT_W {}\n".format(prefix, self.x_input))
            f.write("#define {}_INPUT_H {}\n".format(prefix, self.y_input))
            f.write("#define {}_DST_SIZE {}\n".format(prefix, self.x_output * self.y_output * self.output_ch
                                                      * self.batches))
            f.write("#define {}_INPUT_SIZE {}\n".format(prefix, self.x_input * self.y_input * self.input_ch))
            if self.relu6:
                f.write("#define {}_OUT_ACTIVATION_MIN {}\n".format(prefix, 0))
                f.write("#define {}_OUT_ACTIVATION_MAX {}\n".format(prefix, 6))
            else:
                f.write("#define {}_OUT_ACTIVATION_MIN {}\n".format(prefix, self.out_activation_min))
                f.write("#define {}_OUT_ACTIVATION_MAX {}\n".format(prefix, self.out_activation_max))
            f.write("#define {}_INPUT_BATCHES {}\n".format(prefix, self.batches))
    #self.format_output_file(filepath)

def generate_c_array(name, array, datatype="q7_t", const="const "):
    if not os.path.exists(headers_dir):
        os.makedirs(headers_dir)

    w = None
    if type(array) is list:
        w = array
        size = len(array)
    else:
        w = array
        w = w.ravel()
        #size = tf.size(array)
        size = np.size(array)
    filename = name + "_data.h"
    filepath = headers_dir + filename

    generated_header_files.append(filename)

    print("Generating C header {}...".format(filepath))
    print(filepath)
    with open(filepath, "w+") as f:
        write_c_common_header(f)
        f.write("#include <stdint.h>\n\n")
        f.write(const + datatype + " " + testdataset + '_' + name + "[%d] =\n{\n" % size)
        for i in range(size - 1):
            f.write("  %d,\n" % w[i])
        f.write("  %d\n" % w[size - 1])
        f.write("};\n")
    #self.format_output_file(filepath)
    
def network_idx_gen(name, idx_list, datatype="uint_8", const="const "):
    if not os.path.exists(headers_dir):
        os.makedirs(headers_dir)

    layer_cnt = 0
    filename = name + "_data.h"
    filepath = headers_dir + filename

    generated_header_files.append(filename)

    print("Generating C header {}...".format(filepath))
    print(filepath)
    with open(filepath, "w+") as f:
        write_c_common_header(f)
        f.write("#include <stdint.h>\n\n")
        for array in idx_list:
            layer_cnt = layer_cnt + 1
            w = None
            if type(array) is list:
                w = array
                size = len(array)
            else:
                w = array
                w = w.ravel()
                #size = tf.size(array)
                size = np.size(array)
                
            f.write(const + datatype + " " + name + '_' + "layer_" + str(layer_cnt) + "[%d] =\n{\n" % size)
            for i in range(size - 1):
                f.write("  %d,\n" % w[i])
            f.write("  %d\n" % w[size - 1])
            f.write("};\n")
    #self.format_output_file(filepath)
    
def array_gen(len, max):
    return np.random.randint(max, size = len)

def array_gen_sim(len, max):
    result = np.zeros((len))
    #implemnt something here
    return result



generated_header_files = []    
headers_dir = './index_data/'
input_zero_point = 0
output_zero_point = 0
testdataset = 'full_network_index_data'
lut_size = 32
layer_1 = array_gen(128*128,lut_size)
network_info = [layer_1]

#generate_c_array('index', test_array, datatype='uint8_t')
network_idx_gen('benchmark_idx', network_info, datatype='uint8_t')