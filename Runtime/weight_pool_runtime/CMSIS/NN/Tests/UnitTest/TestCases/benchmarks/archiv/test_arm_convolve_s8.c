/*
 * Copyright (C) 2010-2020 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#define LUT_SIZE 32
#define FW_GRAN 9 //granularity of fixed weight, for fixing 3*3 kernel it should be 9
#define LUT_ENTRY 512 //should be equal to 2^(FW_GRAN)


#include <stdlib.h>
#include "lut_utiles.h"
#include <math.h>

//#include <arm_nnfunctions.h>
#include "D:\research\fixedweightNN\CMSIS_NN\Include\arm_nnfunctions.h"
//#include <unity.h>

/*
#include "../TestData/basic/test_data.h"
#include "../TestData/conv_1_x_n_1/test_data.h"
#include "../TestData/conv_1_x_n_2/test_data.h"
#include "../TestData/conv_1_x_n_3/test_data.h"
#include "../TestData/conv_2/test_data.h"
#include "../TestData/conv_3/test_data.h"
#include "../TestData/conv_4/test_data.h"
#include "../TestData/conv_out_activation/test_data.h"
#include "../TestData/stride2pad1/test_data.h"
*/
#include "../Utils/validate.h"
//#include "../TestData/conv_large/test_data.h"
#include "../TestData/conv_small/test_data.h"
//#include "../TestData/conv_large_2/test_data.h"
/*
void basic_arm_convolve_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[BASIC_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = basic_biases;
    const q7_t *kernel_data = basic_weights;
    const q7_t *input_data = basic_input;
    const q7_t *output_ref = basic_output_ref;
    const int32_t output_ref_size = BASIC_DST_SIZE;

    input_dims.n = BASIC_INPUT_BATCHES;
    input_dims.w = BASIC_INPUT_W;
    input_dims.h = BASIC_INPUT_H;
    input_dims.c = BASIC_IN_CH;
    filter_dims.w = BASIC_FILTER_X;
    filter_dims.h = BASIC_FILTER_Y;
    output_dims.w = BASIC_OUTPUT_W;
    output_dims.h = BASIC_OUTPUT_H;
    output_dims.c = BASIC_OUT_CH;

    conv_params.padding.w = BASIC_PAD_X;
    conv_params.padding.h = BASIC_PAD_Y;
    conv_params.stride.w = BASIC_STRIDE_X;
    conv_params.stride.h = BASIC_STRIDE_Y;

    conv_params.input_offset = BASIC_INPUT_OFFSET;
    conv_params.output_offset = BASIC_OUTPUT_OFFSET;
    conv_params.activation.min = BASIC_OUT_ACTIVATION_MIN;
    conv_params.activation.max = BASIC_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)basic_output_mult;
    quant_params.shift = (int32_t *)basic_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_status result = arm_convolve_s8(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims,
                                        input_data,
                                        &filter_dims,
                                        kernel_data,
                                        &bias_dims,
                                        bias_data,
                                        &output_dims,
                                        output);

    free(ctx.buf);
    //TEST_ASSERT_EQUAL(expected, result);
    //TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    free(ctx.buf);
    //TEST_ASSERT_EQUAL(expected, result);
    //TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void stride2pad1_arm_convolve_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[STRIDE2PAD1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = stride2pad1_biases;
    const q7_t *kernel_data = stride2pad1_weights;
    const q7_t *input_data = stride2pad1_input;
    const q7_t *output_ref = stride2pad1_output_ref;
    const int32_t output_ref_size = STRIDE2PAD1_DST_SIZE;

    input_dims.n = STRIDE2PAD1_INPUT_BATCHES;
    input_dims.w = STRIDE2PAD1_INPUT_W;
    input_dims.h = STRIDE2PAD1_INPUT_H;
    input_dims.c = STRIDE2PAD1_IN_CH;
    filter_dims.w = STRIDE2PAD1_FILTER_X;
    filter_dims.h = STRIDE2PAD1_FILTER_Y;
    output_dims.w = STRIDE2PAD1_OUTPUT_W;
    output_dims.h = STRIDE2PAD1_OUTPUT_H;
    output_dims.c = STRIDE2PAD1_OUT_CH;

    conv_params.padding.w = STRIDE2PAD1_PAD_X;
    conv_params.padding.h = STRIDE2PAD1_PAD_Y;
    conv_params.stride.w = STRIDE2PAD1_STRIDE_X;
    conv_params.stride.h = STRIDE2PAD1_STRIDE_Y;

    conv_params.input_offset = STRIDE2PAD1_INPUT_OFFSET;
    conv_params.output_offset = STRIDE2PAD1_OUTPUT_OFFSET;
    conv_params.activation.min = STRIDE2PAD1_OUT_ACTIVATION_MIN;
    conv_params.activation.max = STRIDE2PAD1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)stride2pad1_output_mult;
    quant_params.shift = (int32_t *)stride2pad1_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_status result = arm_convolve_s8(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims,
                                        input_data,
                                        &filter_dims,
                                        kernel_data,
                                        &bias_dims,
                                        bias_data,
                                        &output_dims,
                                        output);

    free(ctx.buf);
    //TEST_ASSERT_EQUAL(expected, result);
    //TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    free(ctx.buf);
    //TEST_ASSERT_EQUAL(expected, result);
    //TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_2_arm_convolve_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[CONV_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = conv_2_biases;
    const q7_t *kernel_data = conv_2_weights;
    const q7_t *input_data = conv_2_input;
    const q7_t *output_ref = conv_2_output_ref;
    const int32_t output_ref_size = CONV_2_DST_SIZE;

    input_dims.n = CONV_2_INPUT_BATCHES;
    input_dims.w = CONV_2_INPUT_W;
    input_dims.h = CONV_2_INPUT_H;
    input_dims.c = CONV_2_IN_CH;
    filter_dims.w = CONV_2_FILTER_X;
    filter_dims.h = CONV_2_FILTER_Y;
    output_dims.w = CONV_2_OUTPUT_W;
    output_dims.h = CONV_2_OUTPUT_H;
    output_dims.c = CONV_2_OUT_CH;

    conv_params.padding.w = CONV_2_PAD_X;
    conv_params.padding.h = CONV_2_PAD_Y;
    conv_params.stride.w = CONV_2_STRIDE_X;
    conv_params.stride.h = CONV_2_STRIDE_Y;

    conv_params.input_offset = CONV_2_INPUT_OFFSET;
    conv_params.output_offset = CONV_2_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_2_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_2_output_mult;
    quant_params.shift = (int32_t *)conv_2_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_status result = arm_convolve_s8(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims,
                                        input_data,
                                        &filter_dims,
                                        conv_2_weights,
                                        &bias_dims,
                                        bias_data,
                                        &output_dims,
                                        output);

    free(ctx.buf);
    //TEST_ASSERT_EQUAL(expected, result);
    //TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));

    
    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    free(ctx.buf);
    
    //TEST_ASSERT_EQUAL(expected, result);
    //TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_3_arm_convolve_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[CONV_3_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = conv_3_biases;
    const q7_t *kernel_data = conv_3_weights;
    const q7_t *input_data = conv_3_input;
    const q7_t *output_ref = conv_3_output_ref;
    const int32_t output_ref_size = CONV_3_DST_SIZE;

    input_dims.n = CONV_3_INPUT_BATCHES;
    input_dims.w = CONV_3_INPUT_W;
    input_dims.h = CONV_3_INPUT_H;
    input_dims.c = CONV_3_IN_CH;
    filter_dims.w = CONV_3_FILTER_X;
    filter_dims.h = CONV_3_FILTER_Y;
    output_dims.w = CONV_3_OUTPUT_W;
    output_dims.h = CONV_3_OUTPUT_H;
    output_dims.c = CONV_3_OUT_CH;

    conv_params.padding.w = CONV_3_PAD_X;
    conv_params.padding.h = CONV_3_PAD_Y;
    conv_params.stride.w = CONV_3_STRIDE_X;
    conv_params.stride.h = CONV_3_STRIDE_Y;

    conv_params.input_offset = CONV_3_INPUT_OFFSET;
    conv_params.output_offset = CONV_3_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_3_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_3_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_3_output_mult;
    quant_params.shift = (int32_t *)conv_3_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_status result = arm_convolve_s8(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims,
                                        input_data,
                                        &filter_dims,
                                        conv_3_weights,
                                        &bias_dims,
                                        bias_data,
                                        &output_dims,
                                        output);

    free(ctx.buf);
    //TEST_ASSERT_EQUAL(expected, result);
    //TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    free(ctx.buf);
    //TEST_ASSERT_EQUAL(expected, result);
    //TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}
*/
/*
void conv_4_arm_convolve_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[CONV_4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = conv_4_biases;
    const q7_t *kernel_data = conv_4_weights;
    const q7_t *input_data = conv_4_input;
    const q7_t *output_ref = conv_4_output_ref;
    const int32_t output_ref_size = CONV_4_DST_SIZE;

    input_dims.n = CONV_4_INPUT_BATCHES;
    input_dims.w = CONV_4_INPUT_W;
    input_dims.h = CONV_4_INPUT_H;
    input_dims.c = CONV_4_IN_CH;
    filter_dims.w = CONV_4_FILTER_X;
    filter_dims.h = CONV_4_FILTER_Y;
    output_dims.w = CONV_4_OUTPUT_W;
    output_dims.h = CONV_4_OUTPUT_H;
    output_dims.c = CONV_4_OUT_CH;

    conv_params.padding.w = CONV_4_PAD_X;
    conv_params.padding.h = CONV_4_PAD_Y;
    conv_params.stride.w = CONV_4_STRIDE_X;
    conv_params.stride.h = CONV_4_STRIDE_Y;

    conv_params.input_offset = CONV_4_INPUT_OFFSET;
    conv_params.output_offset = CONV_4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_4_output_mult;
    quant_params.shift = (int32_t *)conv_4_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_status result = arm_convolve_s8(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims,
                                        input_data,
                                        &filter_dims,
                                        conv_4_weights,
                                        &bias_dims,
                                        bias_data,
                                        &output_dims,
                                        output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}


void conv_1_x_n_1_arm_convolve_s8(void)
{
    const arm_status expected = ARM_MATH_SIZE_MISMATCH;
    q7_t output[CONV_1_X_N_1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = conv_1_x_n_1_biases;
    const q7_t *kernel_data = conv_1_x_n_1_weights;
    const q7_t *input_data = conv_1_x_n_1_input;
    const q7_t *output_ref = conv_1_x_n_1_output_ref;
    const int32_t output_ref_size = CONV_1_X_N_1_DST_SIZE;

    input_dims.n = CONV_1_X_N_1_INPUT_BATCHES;
    input_dims.w = CONV_1_X_N_1_INPUT_W;
    input_dims.h = CONV_1_X_N_1_INPUT_H;
    input_dims.c = CONV_1_X_N_1_IN_CH;
    filter_dims.w = CONV_1_X_N_1_FILTER_X;
    filter_dims.h = CONV_1_X_N_1_FILTER_Y;
    output_dims.w = CONV_1_X_N_1_OUTPUT_W;
    output_dims.h = CONV_1_X_N_1_OUTPUT_H;
    output_dims.c = CONV_1_X_N_1_OUT_CH;

    conv_params.padding.w = CONV_1_X_N_1_PAD_X;
    conv_params.padding.h = CONV_1_X_N_1_PAD_Y;
    conv_params.stride.w = CONV_1_X_N_1_STRIDE_X;
    conv_params.stride.h = CONV_1_X_N_1_STRIDE_Y;

    conv_params.input_offset = CONV_1_X_N_1_INPUT_OFFSET;
    conv_params.output_offset = CONV_1_X_N_1_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_1_X_N_1_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_1_X_N_1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_1_x_n_1_output_mult;
    quant_params.shift = (int32_t *)conv_1_x_n_1_output_shift;

    int32_t buf_size = arm_convolve_1_x_n_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_status result = arm_convolve_1_x_n_s8(&ctx,
                                              &conv_params,
                                              &quant_params,
                                              &input_dims,
                                              input_data,
                                              &filter_dims,
                                              kernel_data,
                                              &bias_dims,
                                              bias_data,
                                              &output_dims,
                                              output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);

    buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_s8(&ctx,
                             &conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output);
    free(ctx.buf);
    TEST_ASSERT_EQUAL(ARM_MATH_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_1_x_n_2_arm_convolve_s8(void)
{
    const arm_status expected = ARM_MATH_SIZE_MISMATCH;
    q7_t output[CONV_1_X_N_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = conv_1_x_n_2_biases;
    const q7_t *kernel_data = conv_1_x_n_2_weights;
    const q7_t *input_data = conv_1_x_n_2_input;
    const q7_t *output_ref = conv_1_x_n_2_output_ref;
    const int32_t output_ref_size = CONV_1_X_N_2_DST_SIZE;

    input_dims.n = CONV_1_X_N_2_INPUT_BATCHES;
    input_dims.w = CONV_1_X_N_2_INPUT_W;
    input_dims.h = CONV_1_X_N_2_INPUT_H;
    input_dims.c = CONV_1_X_N_2_IN_CH;
    filter_dims.w = CONV_1_X_N_2_FILTER_X;
    filter_dims.h = CONV_1_X_N_2_FILTER_Y;
    output_dims.w = CONV_1_X_N_2_OUTPUT_W;
    output_dims.h = CONV_1_X_N_2_OUTPUT_H;
    output_dims.c = CONV_1_X_N_2_OUT_CH;

    conv_params.padding.w = CONV_1_X_N_2_PAD_X;
    conv_params.padding.h = CONV_1_X_N_2_PAD_Y;
    conv_params.stride.w = CONV_1_X_N_2_STRIDE_X;
    conv_params.stride.h = CONV_1_X_N_2_STRIDE_Y;

    conv_params.input_offset = CONV_1_X_N_2_INPUT_OFFSET;
    conv_params.output_offset = CONV_1_X_N_2_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_1_X_N_2_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_1_X_N_2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_1_x_n_2_output_mult;
    quant_params.shift = (int32_t *)conv_1_x_n_2_output_shift;

    int32_t buf_size = arm_convolve_1_x_n_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_status result = arm_convolve_1_x_n_s8(&ctx,
                                              &conv_params,
                                              &quant_params,
                                              &input_dims,
                                              input_data,
                                              &filter_dims,
                                              kernel_data,
                                              &bias_dims,
                                              bias_data,
                                              &output_dims,
                                              output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);

    buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_s8(&ctx,
                             &conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output);
    free(ctx.buf);
    TEST_ASSERT_EQUAL(ARM_MATH_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_1_x_n_3_arm_convolve_s8(void)
{
    const arm_status expected = ARM_MATH_SIZE_MISMATCH;
    q7_t output[CONV_1_X_N_3_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = conv_1_x_n_3_biases;
    const q7_t *kernel_data = conv_1_x_n_3_weights;
    const q7_t *input_data = conv_1_x_n_3_input;
    const q7_t *output_ref = conv_1_x_n_3_output_ref;
    const int32_t output_ref_size = CONV_1_X_N_3_DST_SIZE;

    input_dims.n = CONV_1_X_N_3_INPUT_BATCHES;
    input_dims.w = CONV_1_X_N_3_INPUT_W;
    input_dims.h = CONV_1_X_N_3_INPUT_H;
    input_dims.c = CONV_1_X_N_3_IN_CH;
    filter_dims.w = CONV_1_X_N_3_FILTER_X;
    filter_dims.h = CONV_1_X_N_3_FILTER_Y;
    output_dims.w = CONV_1_X_N_3_OUTPUT_W;
    output_dims.h = CONV_1_X_N_3_OUTPUT_H;
    output_dims.c = CONV_1_X_N_3_OUT_CH;

    conv_params.padding.w = CONV_1_X_N_3_PAD_X;
    conv_params.padding.h = CONV_1_X_N_3_PAD_Y;
    conv_params.stride.w = CONV_1_X_N_3_STRIDE_X;
    conv_params.stride.h = CONV_1_X_N_3_STRIDE_Y;

    conv_params.input_offset = CONV_1_X_N_3_INPUT_OFFSET;
    conv_params.output_offset = CONV_1_X_N_3_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_1_X_N_3_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_1_X_N_3_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_1_x_n_3_output_mult;
    quant_params.shift = (int32_t *)conv_1_x_n_3_output_shift;

    int32_t buf_size = arm_convolve_1_x_n_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_status result = arm_convolve_1_x_n_s8(&ctx,
                                              &conv_params,
                                              &quant_params,
                                              &input_dims,
                                              input_data,
                                              &filter_dims,
                                              kernel_data,
                                              &bias_dims,
                                              bias_data,
                                              &output_dims,
                                              output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);

    buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_s8(&ctx,
                             &conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output);
    free(ctx.buf);
    TEST_ASSERT_EQUAL(ARM_MATH_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_out_activation_arm_convolve_s8(void)
{
    const arm_status expected = ARM_MATH_SIZE_MISMATCH;
    q7_t output[CONV_OUT_ACTIVATION_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = conv_out_activation_biases;
    const q7_t *kernel_data = conv_out_activation_weights;
    const q7_t *input_data = conv_out_activation_input;
    const q7_t *output_ref = conv_out_activation_output_ref;
    const int32_t output_ref_size = CONV_OUT_ACTIVATION_DST_SIZE;

    input_dims.n = CONV_OUT_ACTIVATION_INPUT_BATCHES;
    input_dims.w = CONV_OUT_ACTIVATION_INPUT_W;
    input_dims.h = CONV_OUT_ACTIVATION_INPUT_H;
    input_dims.c = CONV_OUT_ACTIVATION_IN_CH;
    filter_dims.w = CONV_OUT_ACTIVATION_FILTER_X;
    filter_dims.h = CONV_OUT_ACTIVATION_FILTER_Y;
    output_dims.w = CONV_OUT_ACTIVATION_OUTPUT_W;
    output_dims.h = CONV_OUT_ACTIVATION_OUTPUT_H;
    output_dims.c = CONV_OUT_ACTIVATION_OUT_CH;

    conv_params.padding.w = CONV_OUT_ACTIVATION_PAD_X;
    conv_params.padding.h = CONV_OUT_ACTIVATION_PAD_Y;
    conv_params.stride.w = CONV_OUT_ACTIVATION_STRIDE_X;
    conv_params.stride.h = CONV_OUT_ACTIVATION_STRIDE_Y;

    conv_params.input_offset = CONV_OUT_ACTIVATION_INPUT_OFFSET;
    conv_params.output_offset = CONV_OUT_ACTIVATION_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_OUT_ACTIVATION_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_OUT_ACTIVATION_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_out_activation_output_mult;
    quant_params.shift = (int32_t *)conv_out_activation_output_shift;

    int32_t buf_size = arm_convolve_1_x_n_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_status result = arm_convolve_1_x_n_s8(&ctx,
                                              &conv_params,
                                              &quant_params,
                                              &input_dims,
                                              input_data,
                                              &filter_dims,
                                              kernel_data,
                                              &bias_dims,
                                              bias_data,
                                              &output_dims,
                                              output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);

    buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_s8(&ctx,
                             &conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output);
    free(ctx.buf);
    TEST_ASSERT_EQUAL(ARM_MATH_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}
*/

uint8_t * lut_data_gen(){
  static const int total_entry = LUT_SIZE*LUT_ENTRY;
  uint8_t data_arr[total_entry];
  for(int i=0; i<total_entry; i++){
    data_arr[i] = (uint8_t)rand();
  }
  return data_arr; 
}

uint8_t * lut_index_gen(int n_kernels){
  //number of datas need to same as number of filters
  uint8_t index_arr[n_kernels];
  for(int i=0; i<n_kernels; i++){
    index_arr[i] = (uint8_t)rand();
  }
  return index_arr;
}

void conv_fw_convolve_128(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[CONV_LARGE_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = conv_large_2_biases;
    const q7_t *kernel_data = conv_large_2_weights;
    const q7_t *input_data = conv_large_2_input;
    const q7_t *output_ref = conv_large_2_output_ref;
    const int32_t output_ref_size = CONV_LARGE_2_DST_SIZE;

    input_dims.n = CONV_LARGE_2_INPUT_BATCHES;
    input_dims.w = CONV_LARGE_2_INPUT_W;
    input_dims.h = CONV_LARGE_2_INPUT_H;
    input_dims.c = CONV_LARGE_2_IN_CH;
    filter_dims.w = CONV_LARGE_2_FILTER_X;
    filter_dims.h = CONV_LARGE_2_FILTER_Y;
    output_dims.w = CONV_LARGE_2_OUTPUT_W;
    output_dims.h = CONV_LARGE_2_OUTPUT_H;
    output_dims.c = CONV_LARGE_2_OUT_CH;

    conv_params.padding.w = CONV_LARGE_2_PAD_X;
    conv_params.padding.h = CONV_LARGE_2_PAD_Y;
    conv_params.stride.w = CONV_LARGE_2_STRIDE_X;
    conv_params.stride.h = CONV_LARGE_2_STRIDE_Y;

    conv_params.input_offset = CONV_LARGE_2_INPUT_OFFSET;
    conv_params.output_offset = CONV_LARGE_2_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_LARGE_2_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_LARGE_2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_large_2_output_mult;
    quant_params.shift = (int32_t *)conv_large_2_output_shift;

    uint8_t *lut_data;
    uint8_t *lut_index;

    lut_data = lut_data_gen();
    lut_index = lut_index_gen(output_dims.c * input_dims.c);

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);

  /*
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_status result_1 = fw_conv_v1(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims,
                                        input_data,
                                        &filter_dims,
                                        kernel_pool_lut,
                                        kernel_coeff,
                                        &bias_dims,
                                        bias_data,
                                        &output_dims,
																				kpool_base_addr,
                                        output);
    free(ctx.buf);

    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_status result_2 = fw_conv_v2_loop_reorder(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims,
                                        input_data,
                                        &filter_dims,
                                        kernel_pool_lut,
                                        kernel_coeff,
                                        &bias_dims,
                                        bias_data,
                                        &output_dims,
																				kpool_base_addr,
                                        output);
    free(ctx.buf);

    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_status result_3 = fw_conv_v3_loop_reorder_full(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims,
                                        input_data,
                                        &filter_dims,
                                        kernel_pool_lut,
                                        kernel_coeff,
                                        &bias_dims,
                                        bias_data,
                                        &output_dims,
																				kpool_base_addr,
                                        output);
    free(ctx.buf);
    */
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    arm_status result_arm = arm_convolve_s8(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims,
                                        input_data,
                                        &filter_dims,
                                        kernel_data,
                                        &bias_dims,
                                        bias_data,
                                        &output_dims,
                                        output);

    free(ctx.buf);

    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    arm_status result_lut_1 = lut_conv_v1_nocoeff(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims,
                                        input_data,
                                        &filter_dims,
                                        lut_index,
                                        &bias_dims,
                                        bias_data,
                                        &output_dims,
                                        lut_data,
                                        output);

    free(ctx.buf);

    //TEST_ASSERT_EQUAL(expected, result);
    //TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    /*
    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    free(ctx.buf);
    */
    //TEST_ASSERT_EQUAL(expected, result);
    //TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}


void conv_fw_convolve_32(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[CONV_LARGE_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = conv_large_biases;
    const q7_t *kernel_data = conv_large_weights;
    const q7_t *input_data = conv_large_input;
    const q7_t *output_ref = conv_large_output_ref;
    const int32_t output_ref_size = CONV_LARGE_DST_SIZE;

    input_dims.n = CONV_LARGE_INPUT_BATCHES;
    input_dims.w = CONV_LARGE_INPUT_W;
    input_dims.h = CONV_LARGE_INPUT_H;
    input_dims.c = CONV_LARGE_IN_CH;
    filter_dims.w = CONV_LARGE_FILTER_X;
    filter_dims.h = CONV_LARGE_FILTER_Y;
    output_dims.w = CONV_LARGE_OUTPUT_W;
    output_dims.h = CONV_LARGE_OUTPUT_H;
    output_dims.c = CONV_LARGE_OUT_CH;

    conv_params.padding.w = CONV_LARGE_PAD_X;
    conv_params.padding.h = CONV_LARGE_PAD_Y;
    conv_params.stride.w = CONV_LARGE_STRIDE_X;
    conv_params.stride.h = CONV_LARGE_STRIDE_Y;

    conv_params.input_offset = CONV_LARGE_INPUT_OFFSET;
    conv_params.output_offset = CONV_LARGE_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_LARGE_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_LARGE_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_large_output_mult;
    quant_params.shift = (int32_t *)conv_large_output_shift;

    uint8_t *lut_data;
    uint8_t *lut_index;

    lut_data = lut_data_gen();
    lut_index = lut_index_gen(output_dims.c * input_dims.c);

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);


    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    arm_status result_arm = arm_convolve_s8(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims,
                                        input_data,
                                        &filter_dims,
                                        kernel_data,
                                        &bias_dims,
                                        bias_data,
                                        &output_dims,
                                        output);

    free(ctx.buf);

    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    arm_status result_lut_1 = lut_conv_v1_nocoeff(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims,
                                        input_data,
                                        &filter_dims,
                                        lut_index,
                                        &bias_dims,
                                        bias_data,
                                        &output_dims,
                                        lut_data,
                                        output);

    free(ctx.buf);

}

void conv_fw_convolve_8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[CONV_SMALL_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = conv_small_biases;
    const q7_t *kernel_data = conv_small_weights;
    const q7_t *input_data = conv_small_input;
    const q7_t *output_ref = conv_small_output_ref;
    const int32_t output_ref_size = CONV_SMALL_DST_SIZE;

    input_dims.n = CONV_SMALL_INPUT_BATCHES;
    input_dims.w = CONV_SMALL_INPUT_W;
    input_dims.h = CONV_SMALL_INPUT_H;
    input_dims.c = CONV_SMALL_IN_CH;
    filter_dims.w = CONV_SMALL_FILTER_X;
    filter_dims.h = CONV_SMALL_FILTER_Y;
    output_dims.w = CONV_SMALL_OUTPUT_W;
    output_dims.h = CONV_SMALL_OUTPUT_H;
    output_dims.c = CONV_SMALL_OUT_CH;

    conv_params.padding.w = CONV_SMALL_PAD_X;
    conv_params.padding.h = CONV_SMALL_PAD_Y;
    conv_params.stride.w = CONV_SMALL_STRIDE_X;
    conv_params.stride.h = CONV_SMALL_STRIDE_Y;

    conv_params.input_offset = CONV_SMALL_INPUT_OFFSET;
    conv_params.output_offset = CONV_SMALL_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_SMALL_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_SMALL_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_small_output_mult;
    quant_params.shift = (int32_t *)conv_small_output_shift;

    uint8_t *lut_data;
    uint8_t *lut_index;

    lut_data = lut_data_gen();
    lut_index = lut_index_gen(output_dims.c * input_dims.c);

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);


    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    arm_status result_arm = arm_convolve_s8(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims,
                                        input_data,
                                        &filter_dims,
                                        kernel_data,
                                        &bias_dims,
                                        bias_data,
                                        &output_dims,
                                        output);

    free(ctx.buf);

    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    arm_status result_lut_1 = lut_conv_v1_nocoeff(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims,
                                        input_data,
                                        &filter_dims,
                                        lut_index,
                                        &bias_dims,
                                        bias_data,
                                        &output_dims,
                                        lut_data,
                                        output);

    free(ctx.buf);

}

int main(){
while(1){
	
	//conv_2_arm_convolve_s8();

	conv_fw_convolve_8();
	//int apple = 5;
	
}


return 0;
}