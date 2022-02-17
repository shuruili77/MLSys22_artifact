# MLSys22_artifact for Bit-serial Weight Pools
Codes for MLSys 2022 artifact evaluation. 
The experiment codes contain two parts, one is for runtime evaluation and the other is for neural network accruacy evaluation.

## Runtime evaluation
### Hardware requirements
Two microcontrollers are used:
- STM32F207ZG
- STM32F103RB

### Software requirements
- Keil uVision 5.34 (may require license) or similar IDE (need to have program timer)
- STLink Driver (for connecting the microcontroller)
- STM32CubeMX (for generating driver libraries and initialization codes, free to use)
- Python 3.5+ (for input data and index data generation scripts)
- Tensorflow 2 (for input data generation, not required if using pre-generated data in this repo)

### Useful paths
- data generation scripts: Runtime/weight_pool_runtime
- generated test data: Runtime/weight_pool_runtime/TestData_fullnetwork
- generated index data: Runtime/weight_pool_runtime/index_data
- generated weight pool data: Runtime/weight_pool_runtime/lut_zdim64_data.h
- weight-pool based convolution source codes: Runtime/weight_pool_runtime/CMSIS/NN/Source/ConvolutionFunctions/lut_convolve_zdim.c
- Testbenches: /Runtime/weight_pool_runtime/CMSIS/NN/Tests/UnitTest/TestCases/benchmarks

###  Workflow and usage

#### Data generation
**All test data required (described below) to evaluate the networks reported in the paper are already provided. Data generation is not needed for verifying the results. So this step can be skipped if you want to use pre-generated data.**

The first step of the workflow is to generate input data for the testbenches. The scripts for data generation are located in Runtime/weight_pool_runtime. 

There are three types of data need to be generated: neural network data (layer inputs, bias, network configuration, etc.), weight index data (for the proposed weight-pool networks, neural network weights are replaced with indices pointing to the weight pool array) and The actual weight pool data.

For the neural network data, the generating scripts are data_gen_generic.py and data_gen_mobilenet.py. Use data_gen_generic.py for all networks except for MobileNet and use data_gen_mobilenet.py for MobileNet. The network parameters are already defined in the scripts. When using the generic script, need to modify the 'networkname' and 'selected_network' parameter to the target neural network model. The neural network model parameters are already defined in the script (defined just before the 'networkname' parameter). The default path for generated neural network data is './TestData_fullnetwork/'. Note the neural network data generation scripts is derived from the scripts in ARM CMSIS library and requires Tensorflow to execute. 

For the weight index data, the generating scripts are idx_gen_generic.py and idx_gen_mobilenet.py. Similarly, use idx_gen_generic.py for all networks except MobileNet and use idx_gen_mobilenet.py for MobileNet. When using the generic script, need to modify the 'networkname' and 'network' parameter to the target neural network model. The default path for generated index data is './index_data'. Tensorflow is not used for generating index data, only Numpy is used.

For the actual weight pool data, it can be a random C array with N entries, where N is 256 * WEIGHT_POOL_SIZE. The data can be random because it won't affect the runtime. 

If you want to test on another CNN apart from the ones reported in the paper or just a single layer, you can add the network configuration in idx_gen_generic.py and data_gen_generic.py as python lists. The format is provided in the scripts. Then modify the network name in the scripts and run the scripts to generate the data. 

To run the scripts, type `python xxx.py` in terminal, where xxx.py is the script name you want to run. 

#### Setting up the benchmark
After generating test data, the next step is to setting up corresponding testbenches for runtime evaluation. Testbench for all the networks reported in the paper are provided in '/Runtime/weight_pool_runtime/CMSIS/NN/Tests/UnitTest/TestCases/benchmarks'. The testbench files are used as the entry to the program (main() function). All generated data are included in the testbench as C headers. The microcontroller initialization codes (for STM32F207ZG) are also included in the testbenches. **For STM32F207ZG evaluation you don't need to modify anything unless you want to test on other networks.** You need to write a testbench yourself to test on other networks, and the format can refer to existing benchmarks.

For STM32F103RB evaluation, you need to replace the initialization codes (`SystemClock_Config()` function) to the one for F103RB. The initialization codes can be generated from STM32CubeMX. 

##### STM32CubeMX tutorial
STM32CubeMX is needed to generate initialization codes if you want to test on STM32F103RB. It can also add corresponding driver library to Keil uVision's runtime environment so you don't need to manually import the driver library.

First open the program and start a new project. Then select the corresponding model in the MCU selecter. Then open the Clock configuration tab and set the HCLK to the maximum frequency of the microcontrollers (120 MHz for F207ZG and 72 MHz for F103RB) and press enter, the program will automatically find solution to it. Then click the generate code button to generate the initialization code and required driver files. More detailed tutorials on STM32CubeMX can be found online, for example [this one](https://www.stmicroelectronics.com.cn/content/ccc/resource/training/technical/product_training/group0/76/8c/01/d7/28/d0/4c/e7/STM32G0-Ecosystem-STM32CubeMX-Tool/files/STM32G0-Ecosystem-STM32CubeMX-Tool.pdf/_jcr_content/translations/en.STM32G0-Ecosystem-STM32CubeMX-Tool.pdf).

#### Parameter configuration
Once the test network is determined and required test data is generated, the next step is to set the correct parameters for experiments. In the paper two types of experiment results are reported (Section 5.4.1 and 5.4.2), which are the impact of activation bitwidth and full network benchmark with different weight pool size and activation bitwidth. There are two parameters can be adjusted, namely the activation bitwidth and weight pool size. Both parameters are defined in the weight pool convolution source code: 'Runtime/weight_pool_runtime/CMSIS/NN/Source/ConvolutionFunctions/lut_convolve_zdim.c'. The parameters are defined as C macros in the beginning of the code. *LUT_PREC* defines the activation bitwidth and *LUT_SIZE* defines the weight pool size. Make sure to set these two parameters to correct values before compiling the codes. 

#### Compliation
Before testing on microcontrollers, the codes need to be complied first. The instructions here are for Keil uVision IDE. First create a project and set the device to the correct microcontroller under STMicroelectronics. For run-time environment, select CMSIS-CORE, Device-Startup and Device-STM32Cube HAL-(Common, Cortex, GPIO, RCC). If you are not using STM32CubeMX then you need to manually add the driver files into the source group. Then add everything under '/Runtime/weight_pool_runtime/CMSIS/NN/Source' to the source group. The next step is add the testbench to the source group and only one testbench should be added to the source group to avoid conflicts. 
Then open the 'Options' setting for the targer (right click target), and open the C/C++(AC6) tab to to set the optimization level to **O2**. Finally, build the project and the binary will be generated and ready to test on actual microcontrollers. 

#### Testing
The final step is to test the complied code on the target microcontroller. First install the STLink Driver ([click here to download](https://www.st.com/en/development-tools/stsw-link009.html)) and connect the microcontroller to the host. 

Then open the Options setting again to configure the debugger and target. First open the target tab and set the Xtal to the maximum freqeuncy of the target microcontrollers (120 MHz for F207ZG and 72 MHz for F103RB). Then open the Debug tab and on the top-right click use debugger and in the drop-down menu select ST-Link Debugger. Then click the Settings button on the right of the drop down menu, and open the trace tab to set the Core Clock frequency to the maximum frequency (same as before). 

After finishing the configuration, the next step is to start the debug session. The program will be loaded into the microcontroller and start execution. The runtime is measured by the built-in timer in uVision's debugger and layer-wise runtime can be measured by setting breakpoints accordingly. 

#### Expected results
Key results of runtime evaluation is shown is table 7 of the main paper. The results can be reproduced using the description above.

## Accuracy evaluation
### Hardware requirements
- Large-VRAM GPU preferred for faster training and inference

### Software requirements
- Pytorch 1.9.0
- Python 3.6.1 (Python 3.5+ should work)
- kmeans-pytorch (for k-means clustering of pretrained weights, to install: `pip install kmeans-pytorch`)

###  Useful paths
- Scripts for training uncompressed neural networks and generating weight pools: 'Accuracy\accuracy_codes\original_model_training'
- Scripts for training and evaluating weight pool networks: 'Accuracy\accuracy_codes\fw_training'

### Network and datasets
In the paper five neural networks and two datasets are evaluated. The five neural networks are Resnet-mlperf, Resnet-10, Resnet-14, MobileNet-v2 and TinyConv. Network definition can be found in the corresponding training scripts under  'Accuracy\accuracy_codes\original_model_training'. Two datasets are CIFAR-10 and QuickDraw-100. 
#### CIFAR-10
This dataset can be automatically downloaded when running the training scripts.
#### QuickDraw-100
We used the instructions in this [GitHub repo](https://github.com/XJay18/QuickDraw-pytorch) to download and generate the QuickDraw dataset for Pytorch. Since we use 100 categories, when generating the dataset the "-c" parameter should be set to 100. 

### Workflow and usage
#### Uncompressed network training and weight pool generation
The first step is to train uncompressed network to generate the 'pre-trained' weights that will be further used to generate weight pools. After training the next step is to generate the weight pool for this network by applying K-means clustering on the trained weights. The uncompressed model training and weight pool generation is combined into a single script. You just need to run the corresponding python scipt under 'Accuracy\accuracy_codes\original_model_training' to generate uncompressed weights and weight pools. 

**Make sure to change the 'PATH' variable to the path you want the neural network weights to be stored and change the 'output_path' variable to the path you want to stored the generated weight pool. **

Before running the scripts, you need to install Pytorch and *kmeans-pytorch* library (check software requirements).

#### Weight pool network training

