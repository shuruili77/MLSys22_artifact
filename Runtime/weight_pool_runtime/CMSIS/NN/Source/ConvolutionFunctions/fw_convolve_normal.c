//#include "arm_nnfunctions.h"
#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\CMSIS\NN\Include\arm_nnfunctions.h"
//#include "arm_nnsupportfunctions.h"
#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\CMSIS\NN\Include\arm_nnsupportfunctions.h"

arm_status fw_conv_v1(const cmsis_nn_context *ctx,
                           const cmsis_nn_conv_params *conv_params,
                           const cmsis_nn_per_channel_quant_params *quant_params,
                           const cmsis_nn_dims *input_dims,
                           const q7_t *input_data,
                           const cmsis_nn_dims *filter_dims,
                           const uint8_t *filter_idx,
                           const q7_t *filter_coeff,
                           const cmsis_nn_dims *bias_dims,
                           const int32_t *bias_data,
                           const cmsis_nn_dims *output_dims,
                           const uint32_t base_address,
                           q7_t *output_data)
{
    (void)bias_dims;
    q15_t *buffer_a = (q15_t *)ctx->buf;

    const uint16_t input_batches = input_dims->n;
    const uint16_t input_x = input_dims->w;
    const uint16_t input_y = input_dims->h;
    const uint16_t input_ch = input_dims->c;
    const uint16_t kernel_x = filter_dims->w;
    const uint16_t kernel_y = filter_dims->h;
    const uint16_t output_x = output_dims->w;
    const uint16_t output_y = output_dims->h;
    const uint16_t output_ch = output_dims->c;

    const uint16_t pad_x = conv_params->padding.w;
    const uint16_t pad_y = conv_params->padding.h;
    const uint16_t stride_x = conv_params->stride.w;
    const uint16_t stride_y = conv_params->stride.h;

    const int32_t input_offset = conv_params->input_offset;
    const int32_t out_offset = conv_params->output_offset;
    const int32_t out_activation_min = conv_params->activation.min;
    const int32_t out_activation_max = conv_params->activation.max;
    int32_t *output_mult = quant_params->multiplier;
    int32_t *output_shift = quant_params->shift;

    int i_batch;
    for (i_batch = 0; i_batch < input_batches; i_batch++)
    {
        /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
        (void)buffer_a;
        int32_t i_out_ch, i_out_y, i_out_x, i_input_ch, i_ker_y, i_ker_x;
        int32_t conv_out;
        uint32_t filter_address_cur; //index to access filter from filter pool, which is the actual address of the selected filter/*

        for (i_out_ch = 0; i_out_ch < output_ch; i_out_ch++)
        {
            for (i_out_y = 0; i_out_y < output_y; i_out_y++)
            {
                for (i_out_x = 0; i_out_x < output_x; i_out_x++)
                {
                    conv_out = 0;

                    const int32_t base_idx_y = stride_y * i_out_y - pad_y;
                    const int32_t base_idx_x = stride_x * i_out_x - pad_x;

                    const int32_t ker_y_start = MAX(0, -base_idx_y);
                    const int32_t ker_x_start = MAX(0, -base_idx_x);

                    const int32_t ker_y_end = MIN(kernel_y, input_y - base_idx_y);
                    const int32_t ker_x_end = MIN(kernel_x, input_x - base_idx_x);

                    for (i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                    {
                        for (i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                        {   
                            const int32_t filter_pixel = i_ker_y*3 + i_ker_x;
                            const int32_t in_row = base_idx_y + i_ker_y;
                            const int32_t in_col = base_idx_x + i_ker_x;
                            for (i_input_ch = 0; i_input_ch < input_ch; i_input_ch++)
                            {
                                filter_address_cur = base_address + filter_idx[i_out_ch+i_input_ch];//stupid way to access the index, repeat access for every kernel pixel, need to change loop order 
                                char* filter_pointer = (char*)(filter_address_cur + filter_pixel); //access the filter value by reading memory
                                conv_out +=
                                    (input_data[(in_row * input_x + in_col) * input_ch + i_input_ch] + input_offset) * *filter_pointer;
                                    conv_out = conv_out * filter_coeff[i_out_ch + i_input_ch];//multiply by coeffecients, actually this is the worst way of doing this
                            }
                        }
                    }
                    //conv_out = conv_out * filter_coeff[i_out_ch + i_input_ch];//multiply by coeffecients, actually doesn't compataitable with filter weights
                    if (bias_data)
                    {
                        conv_out += bias_data[i_out_ch];
                    }
                    conv_out = arm_nn_requantize(conv_out, output_mult[i_out_ch], output_shift[i_out_ch]);
                    conv_out += out_offset;
                    conv_out = MAX(conv_out, out_activation_min);
                    conv_out = MIN(conv_out, out_activation_max);
                    output_data[i_out_ch + (i_out_y * output_x + i_out_x) * output_ch] = (int8_t)conv_out;
                }
            }
        }
        /* Advance to the next batch */
        input_data += (input_x * input_y * input_ch);
        output_data += (output_x * output_y * output_ch);
    }

    /* Return to application */
    return ARM_MATH_SUCCESS;
}

arm_status fw_conv_v2_loop_reorder(const cmsis_nn_context *ctx,
                           const cmsis_nn_conv_params *conv_params,
                           const cmsis_nn_per_channel_quant_params *quant_params,
                           const cmsis_nn_dims *input_dims,
                           const q7_t *input_data,
                           const cmsis_nn_dims *filter_dims,
                           const uint8_t *filter_idx,
                           const q7_t *filter_coeff,
                           const cmsis_nn_dims *bias_dims,
                           const int32_t *bias_data,
                           const cmsis_nn_dims *output_dims,
                           const uint32_t base_address,
                           q7_t *output_data)
{
    (void)bias_dims;
    q15_t *buffer_a = (q15_t *)ctx->buf;

    const uint16_t input_batches = input_dims->n;
    const uint16_t input_x = input_dims->w;
    const uint16_t input_y = input_dims->h;
    const uint16_t input_ch = input_dims->c;
    const uint16_t kernel_x = filter_dims->w;
    const uint16_t kernel_y = filter_dims->h;
    const uint16_t output_x = output_dims->w;
    const uint16_t output_y = output_dims->h;
    const uint16_t output_ch = output_dims->c;

    const uint16_t pad_x = conv_params->padding.w;
    const uint16_t pad_y = conv_params->padding.h;
    const uint16_t stride_x = conv_params->stride.w;
    const uint16_t stride_y = conv_params->stride.h;

    const int32_t input_offset = conv_params->input_offset;
    const int32_t out_offset = conv_params->output_offset;
    const int32_t out_activation_min = conv_params->activation.min;
    const int32_t out_activation_max = conv_params->activation.max;
    int32_t *output_mult = quant_params->multiplier;
    int32_t *output_shift = quant_params->shift;

    int i_batch;
    for (i_batch = 0; i_batch < input_batches; i_batch++)
    {
        /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
        (void)buffer_a;
        int32_t i_out_ch, i_out_y, i_out_x, i_input_ch, i_ker_y, i_ker_x;
        int32_t conv_out;
        uint32_t filter_address_cur; //index to access filter from filter pool, which is the actual address of the selected filter/*

        for (i_out_ch = 0; i_out_ch < output_ch; i_out_ch++)
        {
            for (i_out_y = 0; i_out_y < output_y; i_out_y++)
            {
                for (i_out_x = 0; i_out_x < output_x; i_out_x++)
                {
                    conv_out = 0;

                    const int32_t base_idx_y = stride_y * i_out_y - pad_y;
                    const int32_t base_idx_x = stride_x * i_out_x - pad_x;

                    const int32_t ker_y_start = MAX(0, -base_idx_y);
                    const int32_t ker_x_start = MAX(0, -base_idx_x);

                    const int32_t ker_y_end = MIN(kernel_y, input_y - base_idx_y);
                    const int32_t ker_x_end = MIN(kernel_x, input_x - base_idx_x);

                    for (i_input_ch = 0; i_input_ch < input_ch; i_input_ch++)
                    {
                        filter_address_cur = base_address + filter_idx[i_out_ch+i_input_ch];//stupid way to access the index, repeat access for every kernel pixel, need to change loop order 
                        q7_t coeff = filter_coeff[i_out_ch + i_input_ch];

                        for (i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                        {
                            const int32_t in_row = base_idx_y + i_ker_y;
                            for (i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                            {   
                                
                                const int32_t in_col = base_idx_x + i_ker_x;
                                
                                char* filter_pointer = (char*)(filter_address_cur + i_ker_y*3 + i_ker_x); //access the filter value by reading memory
                                conv_out +=
                                    (input_data[(in_row * input_x + in_col) * input_ch + i_input_ch] + input_offset) * *filter_pointer;
                              
                            }
                        }
                        conv_out = conv_out * filter_coeff[i_out_ch + i_input_ch];//multiply by coeffecients for each 2d kernel, repeat this for each filter channel
                    }
                    
                    if (bias_data)
                    {
                        conv_out += bias_data[i_out_ch];
                    }
                    conv_out = arm_nn_requantize(conv_out, output_mult[i_out_ch], output_shift[i_out_ch]);
                    conv_out += out_offset;
                    conv_out = MAX(conv_out, out_activation_min);
                    conv_out = MIN(conv_out, out_activation_max);
                    output_data[i_out_ch + (i_out_y * output_x + i_out_x) * output_ch] = (int8_t)conv_out;
                }
            }
        }
        /* Advance to the next batch */
        input_data += (input_x * input_y * input_ch);
        output_data += (output_x * output_y * output_ch);
    }

    /* Return to application */
    return ARM_MATH_SUCCESS;
}

arm_status fw_conv_v3_loop_reorder_full(const cmsis_nn_context *ctx,
                           const cmsis_nn_conv_params *conv_params,
                           const cmsis_nn_per_channel_quant_params *quant_params,
                           const cmsis_nn_dims *input_dims,
                           const q7_t *input_data,
                           const cmsis_nn_dims *filter_dims,
                           const uint8_t *filter_idx,
                           const q7_t *filter_coeff,
                           const cmsis_nn_dims *bias_dims,
                           const int32_t *bias_data,
                           const cmsis_nn_dims *output_dims,
                           const uint32_t base_address,
                           q7_t *output_data)
{
    //This version make input channel outside of the x and y dimension loop
    (void)bias_dims;
    q15_t *buffer_a = (q15_t *)ctx->buf;

    const uint16_t input_batches = input_dims->n;
    const uint16_t input_x = input_dims->w;
    const uint16_t input_y = input_dims->h;
    const uint16_t input_ch = input_dims->c;
    const uint16_t kernel_x = filter_dims->w;
    const uint16_t kernel_y = filter_dims->h;
    const uint16_t output_x = output_dims->w;
    const uint16_t output_y = output_dims->h;
    const uint16_t output_ch = output_dims->c;

    const uint16_t pad_x = conv_params->padding.w;
    const uint16_t pad_y = conv_params->padding.h;
    const uint16_t stride_x = conv_params->stride.w;
    const uint16_t stride_y = conv_params->stride.h;

    const int32_t input_offset = conv_params->input_offset;
    const int32_t out_offset = conv_params->output_offset;
    const int32_t out_activation_min = conv_params->activation.min;
    const int32_t out_activation_max = conv_params->activation.max;
    int32_t *output_mult = quant_params->multiplier;
    int32_t *output_shift = quant_params->shift;

    int i_batch;
    for (i_batch = 0; i_batch < input_batches; i_batch++)
    {
        /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
        (void)buffer_a;
        int32_t i_out_ch, i_out_y, i_out_x, i_input_ch, i_ker_y, i_ker_x;
        int32_t conv_out;
        uint32_t filter_address_cur; //index to access filter from filter pool, which is the actual address of the selected filter/*

        for (i_out_ch = 0; i_out_ch < output_ch; i_out_ch++)
        {
            for (i_input_ch = 0; i_input_ch < input_ch; i_input_ch++)
            {
                filter_address_cur = base_address + filter_idx[i_out_ch+i_input_ch];//stupid way to access the index, repeat access for every kernel pixel, need to change loop order 
                q7_t coeff = filter_coeff[i_out_ch + i_input_ch];
                for (i_out_y = 0; i_out_y < output_y; i_out_y++)
                {
                    for (i_out_x = 0; i_out_x < output_x; i_out_x++)
                    {
                        conv_out = 0;

                        const int32_t base_idx_y = stride_y * i_out_y - pad_y;
                        const int32_t base_idx_x = stride_x * i_out_x - pad_x;

                        const int32_t ker_y_start = MAX(0, -base_idx_y);
                        const int32_t ker_x_start = MAX(0, -base_idx_x);

                        const int32_t ker_y_end = MIN(kernel_y, input_y - base_idx_y);
                        const int32_t ker_x_end = MIN(kernel_x, input_x - base_idx_x);

                        for (i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                        {
                            const int32_t in_row = base_idx_y + i_ker_y;
                            for (i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                            {   
                                
                                const int32_t in_col = base_idx_x + i_ker_x;
                                
                                char* filter_pointer = (char*)(filter_address_cur + i_ker_y*3 + i_ker_x); //access the filter value by reading memory
                                conv_out +=
                                    (input_data[(in_row * input_x + in_col) * input_ch + i_input_ch] + input_offset) * *filter_pointer;
                              
                            }
                        }
                        conv_out = conv_out * filter_coeff[i_out_ch + i_input_ch];//multiply by coeffecients for each 2d kernel, repeat this for each filter channel
                        
                        if (bias_data)
                        {
                            conv_out += bias_data[i_out_ch];
                        }
                        conv_out = arm_nn_requantize(conv_out, output_mult[i_out_ch], output_shift[i_out_ch]);
                        conv_out += out_offset;
                        conv_out = MAX(conv_out, out_activation_min);
                        conv_out = MIN(conv_out, out_activation_max);
                        output_data[i_out_ch + (i_out_y * output_x + i_out_x) * output_ch] = (int8_t)conv_out;
                    }
                }

            }
        }
        /* Advance to the next batch */
        input_data += (input_x * input_y * input_ch);
        output_data += (output_x * output_y * output_ch);
    }

    /* Return to application */
    return ARM_MATH_SUCCESS;
}
