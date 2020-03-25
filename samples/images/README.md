# Image Samples

This directory contains samples demonstrating use of OpenCL image memory objects.
Unlike OpenCL buffer memory objects, OpenCL image memory objects have a specified format and dimensionality.
Within kernels, images are represented as opaque handles, and are accessed using specialized read and write functions.

## Summary of Image Samples

* [enumimageformats](./00_enumimageformats): Queries and prints the supported image formats for a device.
