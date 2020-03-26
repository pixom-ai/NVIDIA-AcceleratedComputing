#include "assert.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "wb.h"

static char *base_dir;

typedef enum { CPU = 1, GPU_GLOBAL_QUEUE, GPU_BLOCK_QUEUE } Mode;

static void write_data(char *file_name, unsigned int *data,
                       unsigned int len) {
  FILE *handle = fopen(file_name, "w");
  fprintf(handle, "%d\n", len);
  for (unsigned int i = 0; i < len; ++i) {
    fprintf(handle, "%d\n", data[i]);
  }

  fflush(handle);
  fclose(handle);
}

static void write_flag(char *file_name, unsigned int data) {
  FILE *handle = fopen(file_name, "w");
  fprintf(handle, "%d\n", data);

  fflush(handle);
  fclose(handle);
}

void setupProblem(unsigned int numNodes, unsigned int maxNeighborsPerNode,
                  unsigned int **nodePtrs_h,
                  unsigned int **nodeNeighbors_h,
                  unsigned int **nodeVisited_h,
                  unsigned int **nodeVisited_ref,
                  unsigned int **currLevelNodes_h,
                  unsigned int **nextLevelNodes_h,
                  unsigned int **numCurrLevelNodes_h,
                  unsigned int **numNextLevelNodes_h) {

  // Initialize node pointers
  *nodePtrs_h =
      (unsigned int *)malloc((numNodes + 1) * sizeof(unsigned int));
  *nodeVisited_h = (unsigned int *)malloc(numNodes * sizeof(unsigned int));
  *nodeVisited_ref =
      (unsigned int *)malloc(numNodes * sizeof(unsigned int));
  (*nodePtrs_h)[0] = 0;
  for (unsigned int node = 0; node < numNodes; ++node) {
    const unsigned int numNeighbors = rand() % (maxNeighborsPerNode + 1);
    (*nodePtrs_h)[node + 1]         = (*nodePtrs_h)[node] + numNeighbors;
    (*nodeVisited_h)[node] = (*nodeVisited_ref)[node] = 0;
  }

  // Initialize neighbors
  const unsigned int totalNeighbors = (*nodePtrs_h)[numNodes];
  *nodeNeighbors_h =
      (unsigned int *)malloc(totalNeighbors * sizeof(unsigned int));
  for (unsigned int neighborIdx = 0; neighborIdx < totalNeighbors;
       ++neighborIdx) {
    (*nodeNeighbors_h)[neighborIdx] = rand() % numNodes;
  }

  // Initialize current level
  *numCurrLevelNodes_h = (unsigned int *)malloc(sizeof(unsigned int));
  **numCurrLevelNodes_h =
      numNodes / 10; // Let level contain 10% of all nodes
  *currLevelNodes_h = (unsigned int *)malloc((**numCurrLevelNodes_h) *
                                             sizeof(unsigned int));
  for (unsigned int idx = 0; idx < **numCurrLevelNodes_h; ++idx) {
    // Find a node that's not visited yet
    unsigned node;
    do {
      node = rand() % numNodes;
    } while ((*nodeVisited_h)[node]);
    (*currLevelNodes_h)[idx] = node;
    (*nodeVisited_h)[node] = (*nodeVisited_ref)[node] = 1;
  }

  // Prepare next level containers (i.e. output variables)
  *numNextLevelNodes_h  = (unsigned int *)malloc(sizeof(unsigned int));
  **numNextLevelNodes_h = 0;
  *nextLevelNodes_h =
      (unsigned int *)malloc((numNodes) * sizeof(unsigned int));
}

void compute(unsigned int numNodes, unsigned int *nodePtrs,
             unsigned int *nodeNeighbors, unsigned int *nodeVisited,
             unsigned int *nodeVisited_ref, unsigned int *currLevelNodes,
             unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
             unsigned int *numNextLevelNodes) {

  // Initialize reference
  unsigned int numNextLevelNodes_ref = 0;
  unsigned int *nextLevelNodes_ref =
      (unsigned int *)malloc((numNodes) * sizeof(unsigned int));

  // Compute reference out
  // Loop over all nodes in the curent level
  for (unsigned int idx = 0; idx < *numCurrLevelNodes; ++idx) {
    unsigned int node = currLevelNodes[idx];
    // Loop over all neighbors of the node
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1];
         ++nbrIdx) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      // If the neighbor hasn't been visited yet
      if (!nodeVisited_ref[neighbor]) {
        // Mark it and add it to the queue
        nodeVisited_ref[neighbor]                 = 1;
        nextLevelNodes_ref[numNextLevelNodes_ref] = neighbor;
        ++numNextLevelNodes_ref;
      }
    }
  }
}

static void create_dataset(const int datasetNum, const int numNodes,
                           const int maxNeighborsPerNode, Mode mode) {

  const char *dir_name =
      wbDirectory_create(wbPath_join(base_dir, datasetNum));

  // graph structure and current level (input data)
  char *mode_file_name           = wbPath_join(dir_name, "input0.raw");
  char *nodePtr_file_name        = wbPath_join(dir_name, "input1.raw");
  char *nodeNeighbors_file_name  = wbPath_join(dir_name, "input2.raw");
  char *nodeVisited_file_name    = wbPath_join(dir_name, "input3.raw");
  char *currLevelNodes_file_name = wbPath_join(dir_name, "input4.raw");

  // next level nodes (output data)
  char *output_file_name = wbPath_join(dir_name, "output.raw");

  unsigned int *nodePtrs_h;
  unsigned int *nodeNeighbors_h;
  unsigned int *nodeVisited_h;
  unsigned int *nodeVisited_ref; // Needed for reference checking
  unsigned int *currLevelNodes_h;
  unsigned int *nextLevelNodes_h;
  unsigned int *numCurrLevelNodes_h;
  unsigned int *numNextLevelNodes_h;

  setupProblem(numNodes, maxNeighborsPerNode, &nodePtrs_h,
               &nodeNeighbors_h, &nodeVisited_h, &nodeVisited_ref,
               &currLevelNodes_h, &nextLevelNodes_h, &numCurrLevelNodes_h,
               &numNextLevelNodes_h);

  compute(numNodes, nodePtrs_h, nodeNeighbors_h, nodeVisited_h,
          nodeVisited_ref, currLevelNodes_h, nextLevelNodes_h,
          numCurrLevelNodes_h, numNextLevelNodes_h);

  write_data(nodePtr_file_name, nodePtrs_h, numNodes + 1);
  write_data(nodeNeighbors_file_name, nodeNeighbors_h,
             nodePtrs_h[numNodes]);
  write_data(nodeVisited_file_name, nodeVisited_h, numNodes);
  write_data(currLevelNodes_file_name, currLevelNodes_h,
             *numCurrLevelNodes_h);
  write_flag(mode_file_name, mode);
  write_data(output_file_name, nodeVisited_ref, numNodes);

  free(nodePtrs_h);
  free(nodeNeighbors_h);
  free(nodeVisited_h);
  free(nodeVisited_ref);
  free(currLevelNodes_h);
  free(nextLevelNodes_h);
  free(numCurrLevelNodes_h);
  free(numNextLevelNodes_h);
}

int main() {
  base_dir = wbPath_join(wbDirectory_current(), "BfsQueue", "Dataset");
  create_dataset(0, 1024, 2, GPU_GLOBAL_QUEUE);
  create_dataset(1, 4097, 13, GPU_GLOBAL_QUEUE);
  create_dataset(2, 20000, 10, GPU_GLOBAL_QUEUE);
  create_dataset(3, 1024, 2, GPU_BLOCK_QUEUE);
  create_dataset(4, 4097, 13, GPU_BLOCK_QUEUE);
  create_dataset(5, 20000, 10, GPU_BLOCK_QUEUE);
  return 0;
}
