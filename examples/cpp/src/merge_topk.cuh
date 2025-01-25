#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <chrono>
#include <iomanip>

// CUDA error checking
#define CHECK_CUDA(call) do {                                             \
	cudaError_t err = call;                                              \
	if (err != cudaSuccess) {                                            \
		std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__    \
					<< ": " << cudaGetErrorString(err) << std::endl;       \
		exit(1);                                                         \
	}                                                                    \
} while(0)

namespace topk_impl {

constexpr uint32_t BLOCK_SIZE = 256;
constexpr uint32_t WARP_SIZE = 32;

// Warp-level minimum finding
__device__ __forceinline__ void warp_reduce_min(
	float& dist, uint32_t& idx, const uint32_t warp_tid) {
	#pragma unroll
	for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
		float other_dist = __shfl_down_sync(0xffffffff, dist, offset);
		uint32_t other_idx = __shfl_down_sync(0xffffffff, idx, offset);
		if (other_dist < dist || (other_dist == dist && other_idx < idx)) {
			dist = other_dist;
			idx = other_idx;
		}
	}
}

// Optimized kernel
__global__ void kern_topk_merge_optimized(
	const uint32_t topK,
	const uint32_t n_queries,
	const float* __restrict__ input_distances,    // [2*n_queries, topK]
	const uint32_t* __restrict__ input_neighbors, // [2*n_queries, topK]
	float* __restrict__ output_distances,         // [n_queries, topK]
	uint32_t* __restrict__ output_neighbors      // [n_queries, topK]
) {
	const uint32_t query_idx = blockIdx.x;
	if (query_idx >= n_queries) return;

	extern __shared__ char shared_mem[];
	float* shared_distances = reinterpret_cast<float*>(shared_mem);
	uint32_t* shared_neighbors = reinterpret_cast<uint32_t*>(shared_mem + (2 * topK * sizeof(float)));

	const uint32_t tid = threadIdx.x;
	const uint32_t warp_tid = tid % WARP_SIZE;
	const uint32_t total_k = topK * 2;

	// Load data into shared memory
	for (uint32_t i = tid; i < total_k; i += blockDim.x) {
		if (i < topK) {
			shared_distances[i] = input_distances[query_idx * 2 * topK + i];
			shared_neighbors[i] = input_neighbors[query_idx * 2 * topK + i];
		} else {
			shared_distances[i] = input_distances[(query_idx * 2 + 1) * topK + (i - topK)];
			shared_neighbors[i] = input_neighbors[(query_idx * 2 + 1) * topK + (i - topK)];
		}
	}
	__syncthreads();

	// Process topK elements in parallel
	for (uint32_t k = 0; k < topK; k++) {
		float min_dist = INFINITY;
		uint32_t min_idx = 0;

		// Each thread finds its local minimum
		for (uint32_t i = tid; i < total_k; i += blockDim.x) {
			float dist = shared_distances[i];
			if (dist < min_dist) {
				min_dist = dist;
				min_idx = i;
			}
		}

		// Warp-level reduction
		warp_reduce_min(min_dist, min_idx, warp_tid);

		// Write results and mark as processed
		if (tid == 0) {
			output_distances[query_idx * topK + k] = min_dist;
			output_neighbors[query_idx * topK + k] = shared_neighbors[min_idx];
			shared_distances[min_idx] = INFINITY;
		}
		__syncthreads();
	}
}

} // namespace topk_impl

void merge_search_results(
	const uint32_t topk,
	const uint32_t n_queries,
	const float* distances,
	const uint32_t* neighbors,
	float* output_distances,
	uint32_t* output_neighbors,
	cudaStream_t stream
) {
	const uint32_t shared_mem_size = (2 * topk * sizeof(float)) + (2 * topk * sizeof(uint32_t));
	
	dim3 block(topk_impl::BLOCK_SIZE);
	dim3 grid(n_queries);

	topk_impl::kern_topk_merge_optimized<<<grid, block, shared_mem_size, stream>>>(
		topk,
		n_queries,
		distances,
		neighbors,
		output_distances,
		output_neighbors
	);
	
	CHECK_CUDA(cudaGetLastError());
}

// void test_merge_search_results() {
// 	// Test parameters
// 	const uint32_t topk = 10;
// 	const uint32_t n_queries = 5000;
// 	const uint32_t total_queries = 2 * n_queries;
	
// 	std::cout << "Running benchmark with " << n_queries << " queries, topk=" << topk << std::endl;
	
// 	// Allocate host memory
// 	float* h_distances = nullptr;
// 	uint32_t* h_neighbors = nullptr;
// 	CHECK_CUDA(cudaMallocHost(&h_distances, total_queries * topk * sizeof(float)));
// 	CHECK_CUDA(cudaMallocHost(&h_neighbors, total_queries * topk * sizeof(uint32_t)));
	
// 	// Initialize test data
// 	for (uint32_t i = 0; i < total_queries; ++i) {
// 		for (uint32_t j = 0; j < topk; ++j) {
// 			h_distances[i * topk + j] = static_cast<float>(j) + (i % 2) * 0.5f;
// 			h_neighbors[i * topk + j] = j + i * topk;
// 		}
// 	}

// 	// Display first 4 neighbors input data
// 	std::cout << "\nInput data:" << std::endl;
// 	for (uint32_t i = 0; i < 4; ++i) {
// 		std::cout << "\nQuery " << i/2 << " Part " << i%2 << " (First 4):" << std::endl;
// 		for (uint32_t j = 0; j < 4; ++j) {
// 			std::cout << std::fixed << std::setprecision(1)
// 					 << "  [" << j << "] Distance: " << h_distances[i * topk + j] 
// 					 << ", Neighbor: " << h_neighbors[i * topk + j] << std::endl;
// 		}
// 	}
	
// 	// Allocate device memory
// 	float *d_distances = nullptr, *d_output_distances = nullptr;
// 	uint32_t *d_neighbors = nullptr, *d_output_neighbors = nullptr;
	
// 	CHECK_CUDA(cudaMalloc(&d_distances, total_queries * topk * sizeof(float)));
// 	CHECK_CUDA(cudaMalloc(&d_neighbors, total_queries * topk * sizeof(uint32_t)));
// 	CHECK_CUDA(cudaMalloc(&d_output_distances, n_queries * topk * sizeof(float)));
// 	CHECK_CUDA(cudaMalloc(&d_output_neighbors, n_queries * topk * sizeof(uint32_t)));
	
// 	// Copy input data to device
// 	CHECK_CUDA(cudaMemcpy(d_distances, h_distances, total_queries * topk * sizeof(float), cudaMemcpyHostToDevice));
// 	CHECK_CUDA(cudaMemcpy(d_neighbors, h_neighbors, total_queries * topk * sizeof(uint32_t), cudaMemcpyHostToDevice));
	
// 	// Create stream
// 	cudaStream_t stream;
// 	CHECK_CUDA(cudaStreamCreate(&stream));
	
// 	// Warm-up runs
// 	std::cout << "\nPerforming warm-up runs..." << std::endl;
// 	for (int i = 0; i < 10; i++) {
// 		merge_search_results(topk, n_queries, d_distances, d_neighbors, 
// 							 d_output_distances, d_output_neighbors, stream);
// 	}
// 	CHECK_CUDA(cudaStreamSynchronize(stream));
	
// 	// Benchmark timing
// 	const int num_iterations = 100;
// 	std::cout << "Running " << num_iterations << " iterations for timing..." << std::endl;
	
// 	auto start = std::chrono::high_resolution_clock::now();
	
// 	for (int i = 0; i < num_iterations; i++) {
// 		merge_search_results(topk, n_queries, d_distances, d_neighbors, 
// 							 d_output_distances, d_output_neighbors, stream);
// 	}
// 	CHECK_CUDA(cudaStreamSynchronize(stream));
	
// 	auto end = std::chrono::high_resolution_clock::now();
// 	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
// 	double avg_time = duration.count() / (double)num_iterations;
	
// 	std::cout << "Average execution time: " << std::fixed << std::setprecision(2) 
// 				<< avg_time << " microseconds" << std::endl;
// 	std::cout << "Throughput: " << std::scientific << std::setprecision(4) 
// 				<< (n_queries * 1e6) / avg_time << " queries/second" << std::endl;
	
// 	// Get results
// 	float* h_output_distances = nullptr;
// 	uint32_t* h_output_neighbors = nullptr;
// 	CHECK_CUDA(cudaMallocHost(&h_output_distances, n_queries * topk * sizeof(float)));
// 	CHECK_CUDA(cudaMallocHost(&h_output_neighbors, n_queries * topk * sizeof(uint32_t)));
	
// 	CHECK_CUDA(cudaMemcpy(h_output_distances, d_output_distances, n_queries * topk * sizeof(float), cudaMemcpyDeviceToHost));
// 	CHECK_CUDA(cudaMemcpy(h_output_neighbors, d_output_neighbors, n_queries * topk * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	
// 	// Print first two merged query results
// 	std::cout << "\nFinal merged results:" << std::endl;
// 	for (uint32_t i = 0; i < 2; ++i) {
// 		std::cout << "\nMerged Query " << i << ":" << std::endl;
// 		for (uint32_t j = 0; j < topk; ++j) {
// 			std::cout << std::fixed << std::setprecision(1)
// 					 << "  [" << j << "] Distance: " << h_output_distances[i * topk + j] 
// 					 << ", Neighbor: " << h_output_neighbors[i * topk + j] << std::endl;
// 		}
// 	}
	
// 	// Verify results
// 	bool success = true;
// 	for (uint32_t i = 0; i < n_queries && success; ++i) {
// 		for (uint32_t j = 1; j < topk; ++j) {
// 			if (h_output_distances[i * topk + j] < h_output_distances[i * topk + j - 1]) {
// 				std::cout << "\nError: Results not sorted for query " << i 
// 						 << " at position " << j 
// 						 << " (" << h_output_distances[i * topk + j] 
// 						 << " < " << h_output_distances[i * topk + j - 1] << ")" << std::endl;
// 				success = false;
// 				break;
// 			}
// 		}
// 	}
	
// 	// Cleanup
// 	CHECK_CUDA(cudaStreamDestroy(stream));
// 	CHECK_CUDA(cudaFreeHost(h_distances));
// 	CHECK_CUDA(cudaFreeHost(h_neighbors));
// 	CHECK_CUDA(cudaFreeHost(h_output_distances));
// 	CHECK_CUDA(cudaFreeHost(h_output_neighbors));
// 	CHECK_CUDA(cudaFree(d_distances));
// 	CHECK_CUDA(cudaFree(d_neighbors));
// 	CHECK_CUDA(cudaFree(d_output_distances));
// 	CHECK_CUDA(cudaFree(d_output_neighbors));
	
// 	if (success) {
// 		std::cout << "\nAll tests passed successfully!" << std::endl;
// 	}
// }

// int main() {
// 	test_merge_search_results();
// 	return 0;
// }
