# MLSys22_artifact for Bit-serial Weight Pools
Code for MLSys 2022 artifact evaluation 
The experiment codes contains two parts, one is for runtime evaluation and the other is for neural network accruacy evaluation.

## Runtime evaluation
### Hardware requirements
Two microcontrollers are used:
- STM32F207ZG
- STM32F103RB

### Software requirements
Bold text means the software requirement is mandatory, others (not in bold) are optional.
- **Keil uVision 5.34 (may require license) or similar IDE (need to have program timer)**
- STM32CubeMX (for generating driver libraries and initialization codes, free to use, not required to reproduce results for STM32F207GZ)
- Python 3.5+ (for input data and index data generation scripts)
- Tensorflow 2 (for input data generation, not required if using pre-generated data in this repo)

### Useful paths
- data generation scripts: Runtime/weight_pool_runtime
- generated test data: Runtime/weight_pool_runtime/TestData_fullnetwork
- generated index data: Runtime/weight_pool_runtime/index_data
- generated weight pool data: Runtime/weight_pool_runtime/lut_zdim64_data.h
- weight-pool based convolution source codes: Runtime/weight_pool_runtime/CMSIS/NN/Source/ConvolutionFunctions/lut_convolve_zdim.c
- Testbenches: /Runtime/weight_pool_runtime/CMSIS/NN/Tests/UnitTest/TestCases/benchmarks

### Workflow and usage
#### Data generation
**All test data required (described below) to evaluate the networks reported in the paper are already provided. Data generation is not needed for verifying the results. So this step can be skipped if you want to use pre-generated data.**

The first step of the workflow is to generate input data for the testbenches. The scripts for data generation are located in Runtime/weight_pool_runtime. 

There are three types of data need to be generated: neural network data (layer inputs, bias, network configuration, etc.), weight index data (for the proposed weight-pool networks, neural network weights are replaced with indices pointing to the weight pool array) and The actual weight pool data.

For the neural network data, the generating scripts are data_gen_generic.py and data_gen_mobilenet.py. Use data_gen_generic.py for all networks except for MobileNet and use data_gen_mobilenet.py for MobileNet. The network parameters are already defined in the scripts. When using the generic script, need to modify the 'networkname' and 'selected_network' parameter to the target neural network model. The neural network model parameters are already defined in the script (defined just before the 'networkname' parameter). The default path for generated neural network data is './TestData_fullnetwork/'. Note the neural network data generation scripts is derived from the scripts in ARM CMSIS library and requires Tensorflow to execute. 

For the weight index data, the generating scripts are idx_gen_generic.py and idx_gen_mobilenet.py. Similarly, use idx_gen_generic.py for all networks except MobileNet and use idx_gen_mobilenet.py for MobileNet. When using the generic script, need to modify the 'networkname' and 'network' parameter to the target neural network model. The default path for generated index data is './index_data'. Tensorflow is not used for generating index data, only Numpy is used.

For the actual weight pool data, it can be a random C array with N entries, where N is 256 * WEIGHT_POOL_SIZE. The data can be random because it won't affect the runtime. 

#### Parameter configuration
Once the test network is determined and required test data is generated, the next step is to set the correct parameters for experiments. 

#### Compliation and testing
