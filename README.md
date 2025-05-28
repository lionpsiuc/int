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

## 3. Performance Results and Empirical Discussion

All performance benchmarks were conducted using a numerical tolerance of 1.00e-30 and a maximum iteration limit of 2 000 000 000 for the series and continued fraction expansions.

### 3.1. Single-Precision (FP32) Analysis: Impact of Block Size (Single Stream Configuration)

Table 1: FP32 Performance Characteristics versus CUDA Block Size (Single Stream)

| Samples (m) | Orders (n) | Block Size | CPU Time (s) | GPU Time (s) | Overall Speedup | Computational Speedup (CPU/Kernel) |
| :---------- | :--------- | :--------- | :----------- | :----------- | :-------------- | :------------------------- |
| 5000 | 5000 | 64 | 0.830028 | 0.313314 | 2.65x | 613.90x |
| 5000 | 5000 | 128 | 0.827914 | 0.308188 | 2.69x | 616.11x |
| 5000 | 5000 | 256 | 0.831409 | 0.315410 | 2.64x | 610.28x |
| 5000 | 5000 | 512 | 0.831866 | 0.307637 | 2.70x | 583.93x |
| 8192 | 8192 | 64 | 2.161306 | 0.394488 | 5.48x | 655.20x |
| 8192 | 8192 | 128 | 2.154336 | 0.395814 | 5.44x | 653.78x |
| 8192 | 8192 | 256 | 2.149745 | 0.396265 | 5.43x | 641.09x |
| 8192 | 8192 | 512 | 2.142700 | 0.390511 | 5.49x | 609.96x |
| 16384 | 16384 | 64 | 8.083513 | 0.783341 | 10.32x | 665.54x |
| 16384 | 16384 | 128 | 8.180818 | 0.785967 | 10.41x | 684.35x |
| 16384 | 16384 | 256 | 8.099629 | 0.794310 | 10.20x | 672.38x |
| 16384 | 16384 | 512 | 8.064396 | 0.792281 | 10.18x | 633.94x |
| 20000 | 20000 | 64 | 11.611526 | 1.036951 | 11.20x | 662.23x |
| 20000 | 20000 | 128 | 11.570643 | 1.031079 | 11.22x | 667.04x |
| 20000 | 20000 | 256 | 11.566506 | 1.036720 | 11.16x | 662.49x |
| 20000 | 20000 | 512 | 11.585226 | 1.046460 | 11.07x | 639.94x |

#### Discussion (FP32, Single Stream)

- **Overall Speedup** : The GPU implementation consistently surpasses the CPU in performance, with observed speedups increasing with increasing problem dimensions (e.g., from approximately 2.7x for a 5000 x 5000 matrix to around 11.2x for a 20000 x 20000 matrix). This trend indicates favorable scalability.
- **Computational Speedup**: The computational speedup, defined as the ratio of CPU execution time to GPU kernel execution time, is exceptionally high (frequently exceeding 600x).
- **Block Size Optimization**: A CUDA block size of 128 threads per block generally yields optimal or near-optimal overall speedup. Nevertheless, the performance differentials across the tested block sizes (64, 128, 256, 512) are typically marginal, suggesting that the GPU's scheduling mechanism effectively manages thread execution for this specific kernel.
- **Performance Bottleneck Identification**: The substantial discrepancy between the overall speedup and the computational speedup identifies data transfer operations (both host to device and device to host, with device to host being particularly prominent as per detailed timing output) as the predominant performance bottleneck. These transfers consume a significant fraction of the total GPU processing time (e.g., upwards of 70% for n = m = 20000 configurations). The kernel execution phase itself makes up a minor portion of the total elapsed time.

### 3.2. Double-Precision (FP64) Analysis: Impact of Block Size (Single Stream Configuration)

Table 2: FP64 Performance Characteristics versus CUDA Block Size (Single Stream)

| Samples (m) | Orders (n) | Block Size | CPU Time (s) | GPU Time (s) | Overall Speedup | Computational Speedup (CPU/Kernel) |
| :---------- | :--------- | :--------- | :----------- | :----------- | :-------------- | :------------------------- |
| 5000 | 5000 | 64 | 1.861321 | 0.408556 | 4.56x | 34.27x |
| 5000 | 5000 | 128 | 1.867001 | 0.411974 | 4.53x | 34.35x |
| 5000 | 5000 | 256 | 1.866690 | 0.414952 | 4.50x | 34.35x |
| 5000 | 5000 | 512 | 1.859838 | 0.416926 | 4.46x | 34.20x |
| 8192 | 8192 | 64 | 4.716279 | 0.657103 | 7.18x | 33.83x |
| 8192 | 8192 | 128 | 4.705355 | 0.655978 | 7.17x | 34.05x |
| 8192 | 8192 | 256 | 4.712366 | 0.663831 | 7.10x | 33.90x |
| 8192 | 8192 | 512 | 4.726948 | 0.661695 | 7.14x | 34.10x |
| 16384 | 16384 | 64 | 17.678456 | 1.766623 | 10.01x | 36.80x |
| 16384 | 16384 | 128 | 17.663640 | 1.762805 | 10.02x | 37.45x |
| 16384 | 16384 | 256 | 17.697822 | 1.737309 | 10.19x | 38.26x |
| 16384 | 16384 | 512 | 17.621897 | 1.744598 | 10.10x | 38.04x |
| 20000 | 20000 | 64 | 25.674295 | 2.452560 | 10.47x | 38.38x |
| 20000 | 20000 | 128 | 25.715911 | 2.446812 | 10.51x | 38.78x |
| 20000 | 20000 | 256 | 29.099138 | 2.455238 | 11.85x | 43.10x |
| 20000 | 20000 | 512 | 25.707505 | 2.435693 | 10.55x | 38.58x |

#### Discussion (FP64, Single Stream):

- **Overall Speedup**: Consistent with FP32 observations, the overall speedup for FP64 computations demonstrates scalability with problem size, ranging from approximately 4.5x to between 10.5x and 11.8x.
- **Computational Speedup**: The computational speedup is considerably lower than that observed for FP32, typically ranging between 33x and 43x. This reduction is attributable to the lower native throughput of FP64 arithmetic units on the NVIDIA GeForce 2080 SUPER.
- **Optimal Block Size**: The optimal block size exhibits greater variability for FP64 computations compared to FP32. Smaller block dimensions (e.g., 64) tend to be more favorable for smaller configurations, whereas larger block dimensions (e.g., 256, 512) demonstrate improved performance for larger problem instances.
- **Bottleneck and FP64 Computational Cost**: Data transfer persists as the principal performance constraint. However, the GPU computation phase for FP64 operations constitutes a larger proportion of the total GPU execution time (e.g., approximately 13% for 5000 x 5000, increasing to around 27% for 20000 x 20000) relative to FP32. This reflects the inherently higher computational expense associated with double-precision arithmetic, which is obvious.

### 3.3. Comparative Analysis: FP32 versus FP64 (Single Stream Configuration)

FP32 computations achieve substantially higher computational speedups, a consequence of GPU architectures being predominantly optimized for single-precision arithmetic. The overall speedup is also generally superior for FP32, although this advantage diminishes for very large datasets where data transfer times become overwhelmingly dominant for both precision levels. FP64 operations are intrinsically more time-consuming and impose double the memory bandwidth requirements, thereby affecting both computational and data transfer efficiencies.

### 3.4. Evaluation of CUDA Stream Concurrency (n = m = 20000, l = 256)

The efficacy of CUDA streams in overlapping data transfers with kernel execution was assessed.

Table 3: FP32 Stream Performance Characteristics (n = m = 20000, l = 256)

| Streams | CPU Time (s) | GPU Total (s) | Overall Speedup | GPU Allocation (s) | GPU Overlapped (s) | % Overlapped of Total |
| :------ | :----------- | :------------ | :-------------- | :-------------- | :----------------- | :------------- |
| 1 | 11.566506 | 1.036720 | 11.16x | 0.00023 (device) | 0.77444 (sequential) | 74.70% |
| 2 | 11.570466 | 1.033592 | 11.19x | 0.500061 (host and device) | 0.277902 | 26.89% |
| 3 | 11.581276 | 1.036450 | 11.17x | 0.501630 (host and device) | 0.280830 | 27.10% |
| 4 | 11.580983 | 1.052205 | 11.01x | 0.504100 (host and device) | 0.281999 | 26.80% |
| 5 | 11.598100 | 1.038126 | 11.17x | 0.501830 (host and device) | 0.277946 | 26.77% |

FP32 Single-Stream Note: GPU Allocation (s) pertains to device memory allocation only. GPU Overlapped (s) is the sum of sequential host to device, computation, and device to host times from detailed single-stream output. For multi-stream configurations (i.e., streams > 1), GPU Allocation (s) includes host-side vector resize operations.

Table 4: FP64 Stream Performance Characteristics (n=m=20000, l=256)

| Streams | CPU Time (s) | GPU Total (s) | Overall Speedup | GPU Allocation (s) | GPU Overlapped (s) | % Overlapped of Total |
| :------ | :----------- | :------------ | :-------------- | :-------------- | :----------------- | :------------- |
| 1 | 29.099138 | 2.455238 | 11.85x | 0.00025 (device) | 2.19129 (sequential) | 89.25% |
| 2 | 25.705160 | 2.451828 | 10.48x | 1.004327 (host and device) | 1.185583 | 48.35% |
| 3 | 25.724427 | 2.468948 | 10.42x | 1.005048 (host and device) | 1.197168 | 48.49% |
| 4 | 25.715040 | 2.478240 | 10.38x | 1.005166 (host and device) | 1.202894 | 48.54% |
| 5 | 25.666575 | 2.482904 | 10.34x | 1.015859 (host and device) | 1.198634 | 48.28% |

FP64 Single-Stream Note: GPU Allocation (s) pertains to device memory. GPU Overlapped (s) is the sum of host to device, computation, and device to host times. For multi-stream (i.e., streams > 1), GPU Allocation includes host vector resize. Note the variation in CPU Time (s) between single-stream and multi-stream FP64 benchmarks.

#### Discussion of CUDA Stream Effectiveness

- **Operational Overlap Confirmation**: A significant reduction in the GPU Overlapped (s) time is observed when transitioning from a single stream to two streams for both FP32 (0.77s to approximately 0.28s) and FP64 (2.19s to approximately 1.19s). This empirically validates that asynchronous host to device transfers, kernel executions, and device to host transfers are indeed being executed concurrently.
- **Impact on Total GPU Time**: The use of multiple streams did not translate into a substantial reduction in the total GPU execution time. In the FP32 case, employing two streams yielded a marginal improvement over a single stream, whereas additional streams provided no further  benefit and, in some instances, slightly degraded performance. For FP64 computations, the total GPU time remained largely static or exhibited a minor increase with an increasing number of streams.
- **Interpretation of GPU Allocation (s) in Multi-Stream Context**: The GPU Allocation (s) time reported for multi-stream configurations is considerably larger. This is attributable to the timing methodology in gpu.cu for num_streams > 1, which incorporates the duration of the host-side host_results_gpu.resize() operation. For the n = m = 20000 problem size, this host-side memory allocation for a substantial vector (1.6 GB for FP32, 3.2 GB for FP64) represents a non-negligible overhead and significantly influences the reported GPU Allocation (s) obscuring potential improvements in device-side allocation efficiency.
- **Definition of Computational Speedup (CPU/Kernel) with Streams**: The Computational Speedup (CPU/Kernel) metric, as reported by the application for multi-stream scenarios, is derived from CPU Time (s) / GPU Overlapped (s). Given that the GPU Overlapped (s) encompasses data transfers in addition to kernel computations, this metric is not directly analogous to the single-stream ational Speedup (CPU/Kernel), which is based on CPU Time (s) / Kernel Execution Time (s).
- **Factors Limiting Overall Benefit from Streams**:

    - **Inclusion of Host-Side Operations in Timing**: The incorporation of host-side vector::resize operations within the timed GPU Allocation (s) phase for multi-stream runs confounds the assessment of pure GPU-side performance enhancements.
    - **PCIe Bandwidth Constraints**: The application is inherently data-intensive, involving the transfer of large data volumes. The total available PCIe bus bandwidth constitutes a fundamental performance limitation. While streams can facilitate more effective concurrent utilization of copy engines and compute resources, they cannot augment the intrinsic maximum throughput of the bus.
    - **Kernel Compute Intensity Relative to Data Transfer**: For FP32 computations, the kernel execution is extremely rapid. Consequently, the temporal window available for overlapping operations is predominantly dictated by the duration of data transfers. In the FP64 context, the kernel is computationally more demanding, thereby offering a greater opportunity for data transfer latencies to be masked by computation, an effect reflected in the more pronounced relative reduction of the GPU Overlapped (s) time.
    - **Stream Management Overhead**: The creation, management, and synchronization of multiple CUDA streams introduce a degree of operational overhead. For this specific workload, particularly when considering the inclusion of substantial host-side allocation within the timed measurements, this overhead may counteract the benefits of concurrency beyond a small number of streams.

#### Conclusion on Stream Utilization

CUDA streams demonstrably enable the concurrent execution of asynchronous data transfers and kernel computations. However, for this particular application, which is characterized by substantial data transfer volumes and (particularly in the FP32 case) very rapid kernel execution relative to these transfer times, the resultant end-to-end performance improvement in terms of total GPU execution time is marginal.

## 4. Overall Conclusion

The GPU-accelerated implementation presented herein achieves substantial overall speedups, reaching up to approximately 11-12x that of the serial CPU counterpart, particularly when processing large-scale datasets. Single-precision (FP32) arithmetic executed on the GPU exhibits markedly higher computational throughput compared to double-precision (FP64) operations. The principal factor constraining performance is identified as the data transfer latency between the host system and the GPU device. Although CUDA streams facilitate operational overlap, their application in this data-intensive workload does not yield significant reductions in total execution time. This outcome is likely attributable to the saturation of PCIe bandwidth and the confounding influence of including host-side memory operations within the timed sections for multi-stream configurations.