# CUDA Exponential Integral

## 1. Implementation Overview

The core functionality of the project involves the computation of E_n(x) across a matrix defined by orders n and sample values x.

### 1.1. Central Processing Unit (CPU) Implementation

The CPU-based computation is found within the exponentialIntegralCpu function, located in src/main.cpp. This function executes the E_n(x) calculation serially for each (order, sample) pair. The numerical method employed is adaptive: for x > 1, a continued fraction expansion is utilized due to its favorable convergence properties in this domain, whereas for x <= 1, a power series expansion is applied.

### 1.2. Graphics Processing Unit (GPU) Implementation

The CUDA implementation, contained within src/gpu.cu, leverages the massively parallel architecture to accelerate the E_n(x) computations.

#### 1.2.1. Kernel Design (exponential_integral_kernel)

A two-dimensional CUDA kernel, exponential_integral_kernel, is launched to perform the parallel computations. The grid and block dimensions are configured such that individual threads are mapped to unique (order, sample) pairs, facilitating fine-grained parallelism inherent to the problem structure:

- The global thread index in the x-dimension, calculated as blockIdx.x * blockDim.x + threadIdx.x, corresponds to the sample index.
- The block index in the y-dimension, blockIdx.y, corresponds to the order index (n - 1, as orders are 1-indexed).
- The core computational logic within each thread is executed by the exponential_integral_device_logic function, which mirrors the numerical methods (continued fraction and power series expansions) employed in the CPU version.

#### 1.2.2. Data Management and Execution Flow (batch_exponential_integral_gpu)

The host function batch_exponential_integral_gpu orchestrates the sequence of operations required for GPU execution:

- **Device Memory Allocation**: Memory is allocated on the GPU for the input sample array (d_samples) and the output results array (d_results).
- **Host to Device Data Transfer**: The host_samples array is copied from host memory to the allocated device memory (d_samples).
- **Kernel Invocation**: The exponential_integral_kernel is launched on the GPU.
- **Device to Host Data Transfer**: The computed d_results array is copied from device memory back to the host memory (host_results_gpu).

#### 1.2.3. CUDA Stream Implementation Strategy

For scenarios where num_streams > 1, the batch_exponential_integral_gpu function is adapted to facilitate concurrent execution of operations through CUDA streams:

- **Stream Initialization**: An array comprising num_streams CUDA streams is created.
- **Comprehensive Memory Allocation**: Device memory sufficient for the entire dataset of samples and results is allocated prior to stream operations. It is important to note that the timing reported for this phase in multi-stream configurations, as per the application's output, also encompasses the host-side std::vector::resize operation for host_results_gpu.
- **Data Chunking**: The total set of num_samples is partitioned into num_streams discrete chunks.
- **Asynchronous Operations Loop**: A loop iterates num_streams times, performing the following operations for each chunk i:

    - The i $^\text{th}$ chunk of sample data is transferred asynchronously from host to device memory via cudaMemcpyAsync, associated with streams[i].
    - The exponential_integral_kernel is launched asynchronously on streams[i], configured to process only the samples within the current chunk. The sample_offset kernel parameter ensures that results are written to their correct global positions within the d_results array.
    - The computed results corresponding to the current chunk are transferred asynchronously from device to host memory using cudaMemcpy2DAsync on streams[i]. The use of cudaMemcpy2DAsync is imperative for correctly managing the memory pitch of the two-dimensional result data during the transfer of partial (chunk-wise) results across all orders.

- **Global Synchronization**: Upon enqueuing all asynchronous operations for all streams, cudaDeviceSynchronize() is called to ensure completion.

The overlapped time, as reported for multi-stream executions, quantifies the duration from the initiation of the first asynchronous operation to the completion of the cudaDeviceSynchronize() call. This metric, therefore, covrers the concurrently executed host to device transfers, kernel computations, and device to host transfers.

## 2. Compilation and Execution Instructions

### 2.1. Compilation

The project utilizes a Makefile for streamlined compilation.

- **Single-Precision (FP32)**: To compile the application for single-precision floating-point calculations (default), execute:

    ```bash
    make
    ```

- **Double-Precision (FP64)**: To compile for double-precision floating-point calculations, use the type variable:

    ```bash
    make type=double
    ```

- **Important Note on Switching Precision**: Both single-precision and double-precision compilations produce an executable named main. It is crucial to clean the project before switching between precision modes. To ensure a clean build when changing precision:

    ```bash
    make clean
    make type=double
    ```

### 2.2. Execution

The compiled program, main, can be executed from the command-line with various options.

- **Basic Execution**:

    ```bash
    ./main [options]
    ```

- **Command-Line Options**: A comprehensive list of available command-line options and their descriptions can be displayed by executing the program with the -h flag:

    ```bash
    ./main -h
    ```

- **Example Execution**: To run a simulation with 20 000 orders, 20 000 samples, a CUDA block size of 256, using 2 CUDA streams, and enabling detailed timing output:

    ```bash
    ./main -n 20000 -m 20000 -l 256 -s 2 -t
    ```