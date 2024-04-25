# example

## Layer Purpose

This is a very simple layer that demonstrates how to write a layer to trace an OpenCL API, in this case `clGetPlatformIDs`.
This example layer was heavily inspired by Brice Videau's (@Kerilk's) [presentation](https://youtu.be/QUKhspUEh00) and [sample code](https://github.com/Kerilk/OpenCL-Layers-Tutorial) from [IWOCL](https://www.iwocl.org/) 2021.
Because it is so simple, this is a good layer to verify that the correct version of the OpenCL ICD loader is installed and that your environment is correctly setup to build and use layers.

Please see the presentatation for more information about how layers work in general, and how this layer works specifically.

## Key APIs and Concepts

The most important concepts to understand from this sample are the functions to query the properties of a layer and to install the dispatch table for the layer.
The contents of these functions will be similar for all other layers.

```c
clGetLayerInfo
clInitLayer
```
