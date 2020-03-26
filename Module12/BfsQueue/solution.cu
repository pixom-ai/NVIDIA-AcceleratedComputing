
#include <stdio.h>
#include <wb.h>

#define BLOCK_SIZE 512
// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 2048

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Global queuing stub
__global__ void gpu_global_queuing_kernel(
    int *nodePtrs, int *nodeNeighbors, int *nodeVisited,
    int *currLevelNodes, int *nextLevelNodes,
    const unsigned int numCurrLevelNodes, int *numNextLevelNodes) {

  //@@ Insert Global Queuing Code Here

  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Loop over all nodes in the curent level
  for (unsigned int idx = tid; idx < numCurrLevelNodes;
       idx += gridDim.x * blockDim.x) {
    const unsigned int node = currLevelNodes[idx];
    // Loop over all neighbors of the node
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1];
         ++nbrIdx) {
      const unsigned int neighbor = nodeNeighbors[nbrIdx];
      // If the neighbor hasn't been visited yet
      const unsigned int visited = atomicExch(&(nodeVisited[neighbor]), 1);
      if (!visited) {
        // Add it to the global queue (already marked in the exchange)
        const unsigned int gQIdx = atomicAdd(numNextLevelNodes, 1);
        nextLevelNodes[gQIdx]    = neighbor;
      }
    }
  }
}

// Block queuing stub
__global__ void gpu_block_queuing_kernel(
    int *nodePtrs, int *nodeNeighbors, int *nodeVisited,
    int *currLevelNodes, int *nextLevelNodes,
    const unsigned int numCurrLevelNodes, int *numNextLevelNodes) {

  //@@ INSERT KERNEL CODE HERE

  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numCurrLevelNodes_reg = numCurrLevelNodes;

  // Initialize shared memory queue
  __shared__ int bQueue[BQ_CAPACITY];
  __shared__ int bQueueCount, gQueueStartIdx;
  if (threadIdx.x == 0) {
    bQueueCount = 0;
  }
  __syncthreads();

  // Loop over all nodes in the curent level
  for (unsigned int idx = tid; idx < numCurrLevelNodes_reg;
       idx += gridDim.x * blockDim.x) {
    const unsigned int node = currLevelNodes[idx];
    // Loop over all neighbors of the node
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1];
         ++nbrIdx) {
      const unsigned int neighbor = nodeNeighbors[nbrIdx];
      // If the neighbor hasn't been visited yet
      const unsigned int visited = atomicExch(&(nodeVisited[neighbor]), 1);
      if (!visited) {
        // Add it to the block queue
        const unsigned int bQueueIdx = atomicAdd(&bQueueCount, 1);
        if (bQueueIdx < BQ_CAPACITY) {
          bQueue[bQueueIdx] = neighbor;
        } else { // If full, add it to the global queue
          bQueueCount                  = BQ_CAPACITY;
          const unsigned int gQueueIdx = atomicAdd(numNextLevelNodes, 1);
          nextLevelNodes[gQueueIdx]    = neighbor;
        }
      }
    }
  }
  __syncthreads();

  // Allocate space for block queue to go into global queue
  if (threadIdx.x == 0) {
    gQueueStartIdx = atomicAdd(numNextLevelNodes, bQueueCount);
  }
  __syncthreads();

  // Store block queue in global queue
  for (unsigned int bQueueIdx = threadIdx.x; bQueueIdx < bQueueCount;
       bQueueIdx += blockDim.x) {
    nextLevelNodes[gQueueStartIdx + bQueueIdx] = bQueue[bQueueIdx];
  }
}

// Host function for global queuing invocation
void gpu_global_queuing(int *nodePtrs, int *nodeNeighbors,
                        int *nodeVisited, int *currLevelNodes,
                        int *nextLevelNodes,
                        unsigned int numCurrLevelNodes,
                        int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(
      nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
      numCurrLevelNodes, numNextLevelNodes);
}

// Host function for block queuing invocation
void gpu_block_queuing(int *nodePtrs, int *nodeNeighbors, int *nodeVisited,
                       int *currLevelNodes, int *nextLevelNodes,
                       unsigned int numCurrLevelNodes,
                       int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(
      nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
      numCurrLevelNodes, numNextLevelNodes);
}

int main(int argc, char *argv[]) {
  // Variables
  int numNodes;
  int *nodePtrs_h;
  int *nodeNeighbors_h;
  int *nodeVisited_h;
  int numTotalNeighbors_h;
  int *currLevelNodes_h;
  int *nextLevelNodes_h;
  int numCurrLevelNodes;
  int numNextLevelNodes_h;
  int *nodePtrs_d;
  int *nodeNeighbors_d;
  int *nodeVisited_d;
  int *currLevelNodes_d;
  int *nextLevelNodes_d;
  int *numNextLevelNodes_d;

  enum Mode { GPU_GLOBAL_QUEUE = 2, GPU_BLOCK_QUEUE };

  wbArg_t args = wbArg_read(argc, argv);
  Mode mode    = (Mode)wbImport_flag(wbArg_getInputFile(args, 0));

  nodePtrs_h =
      (int *)wbImport(wbArg_getInputFile(args, 1), &numNodes, "Integer");
  nodeNeighbors_h = (int *)wbImport(wbArg_getInputFile(args, 2),
                                    &numTotalNeighbors_h, "Integer");

  nodeVisited_h =
      (int *)wbImport(wbArg_getInputFile(args, 3), &numNodes, "Integer");
  currLevelNodes_h = (int *)wbImport(wbArg_getInputFile(args, 4),
                                     &numCurrLevelNodes, "Integer");

  // (do not modify) Datasets should be consistent
  if (nodePtrs_h[numNodes] != numTotalNeighbors_h) {
    wbLog(ERROR, "Datasets are inconsistent! Please report this.");
  }

  // (do not modify) Prepare next level containers (i.e. output variables)
  numNextLevelNodes_h = 0;
  nextLevelNodes_h    = (int *)malloc((numNodes) * sizeof(int));

  wbLog(TRACE, "# Modes = ", mode);
  wbLog(TRACE, "# Nodes = ", numNodes);
  wbLog(TRACE, "# Total Neighbors = ", numTotalNeighbors_h);
  wbLog(TRACE, "# Current Level Nodes = ", numCurrLevelNodes);

  // (do not modify) Allocate device variables --------------------------

  wbLog(TRACE, "Allocating device variables...");

  wbCheck(cudaMalloc((void **)&nodePtrs_d, (numNodes + 1) * sizeof(int)));
  wbCheck(cudaMalloc((void **)&nodeVisited_d, numNodes * sizeof(int)));
  wbCheck(cudaMalloc((void **)&nodeNeighbors_d,
                     nodePtrs_h[numNodes] * sizeof(int)));
  wbCheck(cudaMalloc((void **)&currLevelNodes_d,
                     numCurrLevelNodes * sizeof(int)));
  wbCheck(cudaMalloc((void **)&numNextLevelNodes_d, sizeof(int)));
  wbCheck(
      cudaMalloc((void **)&nextLevelNodes_d, (numNodes) * sizeof(int)));
  wbCheck(cudaDeviceSynchronize());

  // (do not modify) Copy host variables to device --------------------

  wbLog(TRACE, "Copying data from host to device...");

  wbCheck(cudaMemcpy(nodePtrs_d, nodePtrs_h, (numNodes + 1) * sizeof(int),
                     cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(nodeVisited_d, nodeVisited_h, numNodes * sizeof(int),
                     cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(nodeNeighbors_d, nodeNeighbors_h,
                     nodePtrs_h[numNodes] * sizeof(int),
                     cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(currLevelNodes_d, currLevelNodes_h,
                     numCurrLevelNodes * sizeof(int),
                     cudaMemcpyHostToDevice));
  wbCheck(cudaMemset(numNextLevelNodes_d, 0, sizeof(int)));
  wbCheck(cudaDeviceSynchronize());

  // (do not modify) Launch kernel ----------------------------------------

  printf("Launching kernel ");

  if (mode == GPU_GLOBAL_QUEUE) {
    wbLog(INFO, "(GPU with global queuing)...");
    gpu_global_queuing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d,
                       currLevelNodes_d, nextLevelNodes_d,
                       numCurrLevelNodes, numNextLevelNodes_d);
    wbCheck(cudaDeviceSynchronize());
  } else if (mode == GPU_BLOCK_QUEUE) {
    wbLog(INFO, "(GPU with block and global queuing)...");
    gpu_block_queuing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d,
                      currLevelNodes_d, nextLevelNodes_d,
                      numCurrLevelNodes, numNextLevelNodes_d);
    wbCheck(cudaDeviceSynchronize());
  } else {
    wbLog(ERROR, "Invalid mode!\n");
    exit(0);
  }

  // (do not modify) Copy device variables from host ----------------------

  wbLog(INFO, "Copying data from device to host...");

  wbCheck(cudaMemcpy(&numNextLevelNodes_h, numNextLevelNodes_d,
                     sizeof(int), cudaMemcpyDeviceToHost));
  wbCheck(cudaMemcpy(nextLevelNodes_h, nextLevelNodes_d,
                     numNodes * sizeof(int), cudaMemcpyDeviceToHost));
  wbCheck(cudaMemcpy(nodeVisited_h, nodeVisited_d, numNodes * sizeof(int),
                     cudaMemcpyDeviceToHost));
  wbCheck(cudaDeviceSynchronize());

  // (do not modify) Verify correctness
  // -------------------------------------
  // Only check that the visited nodes match the reference implementation

  wbSolution(args, nodeVisited_h, numNodes);

  // (do not modify) Free memory
  // ------------------------------------------------------------
  free(nodePtrs_h);
  free(nodeVisited_h);
  free(nodeNeighbors_h);
  free(currLevelNodes_h);
  free(nextLevelNodes_h);
  wbCheck(cudaFree(nodePtrs_d));
  wbCheck(cudaFree(nodeVisited_d));
  wbCheck(cudaFree(nodeNeighbors_d));
  wbCheck(cudaFree(currLevelNodes_d));
  wbCheck(cudaFree(numNextLevelNodes_d));
  wbCheck(cudaFree(nextLevelNodes_d));

  return 0;
}
