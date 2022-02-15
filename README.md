# MLSys22_artifact for Bit-serial Weight Pools
Code for MLSys 2022 artifact evaluation 
The experiment codes contains two parts, one is for runtime evaluation and the other is for neural network accruacy evaluation.

## Runtime evaluation
### Hardware requirements
Two microcontrollers are used:
- STM32F207ZG
- STM32F103RB

### Software requirements
- Keil uVision 5.34 (may require license) or similar IDE (need to have program timer)
- STM32CubeMX (for generating driver libraries and initialization codes, free to use, not required to reproduce results for STM32F207GZ)
- Tensorflow 2 (for input data generation, not required if using pre-generated data in this repo)

### Useful paths
- data generation scripts: Runtime/weight_pool_runtime
- generated test data: Runtime/weight_pool_runtime/TestData_fullnetwork
- generated idex data: Runtime/weight_pool_runtime/index_data
- weight-pool based convolution source codes: Runtime/weight_pool_runtime/CMSIS/NN/Source/ConvolutionFunctions/lut_convolve_zdim.c
- Testbenches: /Runtime/weight_pool_runtime/CMSIS/NN/Tests/UnitTest/TestCases/benchmarks

### Workflow 
#### Data generation
The first step of the workflow is to generate input data for the testbenches. The scripts for data generation are located in Runtime/weight_pool_runtime. 

There are three types of data need to be generated: neural network data (layer inputs, bias, network configuration, etc.), weight index data (for the proposed weight-pool networks, neural network weights are replaced with indices pointing to the weight pool array) and The actual weight pool.

