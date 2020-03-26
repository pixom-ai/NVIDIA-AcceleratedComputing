---
title: Breadth-First Search Queue
author: GPU Teaching Kit -- Accelerated Computing
module: 12
---

# Objective

The purpose of this lab is to understand hierarchical queuing in the context of the breadth first search algorithm as an example. You will implement a single iteration of breadth first search that takes a set of nodes in the current level (also called wave-front) as input and outputs the set of nodes belonging to the next level. You will implement two kernels:
* A simple version with global queuing
* An optimized version that uses shared-memory queuing.

# Instructions

The graph structure is stored in the following way:

* `numNodes` - the total number of nodes in the graph
* `nodePtrs` - an array of length `numNodes`. Each entry is a pointer into
 `numNeighbors`, described below.
* `nodeNeighbors` - an array whose length is the total number of neighbors each
node has. `nodeNeighbors[nodePtrs[node]]` to `nodeNeighbors[nodePtrs[node+1]]`
describes the neighbors of node `node`.

The kernels take these structures as inputs, as well as a list of nodes in the current level, for which all of the neighbors must be visited.

* `currLevelNodes` - an array of nodes for which neighbors must be visited
* `numCurrLevelNodes` - the size of the previous array
* `visitedNodes` - an array that describes which nodes have already been visited in the BFS

The kernels will need to produce the following outputs

* `nextLevelNodes` - an array of neighbor nodes.
* `numNextLevelNodes` - the number of neighbors
* `visitedNodes` - the nodes that have been visited by the end of the iteration. Note that this is the same array as the input. This is the output value that WebGPU will compare for correctness of your implementation.

Sequential pseudocode for the kernel is:

    // Loop over all nodes in the current level
    for idx = 0..numCurrLevelNodes
      node = currLevelNodes[idx];
      // Loop over all neighbors of the node
      for(nbrIdx = nodePtrs[node]..nodePtrs[node + 1];
        neighbor = nodeNeighbors[nbrIdx];
        // If the neighbor hasn't been visited yet
        if !nodeVisited[neighbor]
          // Mark it and add it to the queue
          nodeVisited[neighbor] = 1;
          nextLevelNodes[*numNextLevelNodes] = neighbor;
          ++(*numNextLevelNodes);


An empty stub for the kernels is provided. All you need to do is correctly implement the kernel code.

# Test Datasets
The first three datasets invoke the Global Queue kernel. The second three invoke the Block Queue Kernel

# Local Setup Instructions

The most recent version of source code for this lab along with the build-scripts can be found on the [Bitbucket repository](LINKTOLAB). A description on how to use the [CMake](https://cmake.org/) tool in along with how to build the labs for local development found in the [README](LINKTOREADME) document in the root of the repository.

The executable generated as a result of compiling the lab can be run using the following command:

```{.bash}
./BfsQueue_Template -e <expected.raw> \
  -i <input0.raw>,<input1.raw>,<input2.raw>,<input3.raw>,<input4.raw> -t integral_vector
```

where `<expected.raw>` is the expected output, `<input.raw>` is the input dataset. The datasets can be generated using the dataset generator built as part of the compilation process.
