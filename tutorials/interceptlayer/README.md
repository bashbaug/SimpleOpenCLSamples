# Using the Intercept Layer for OpenCL Applications

## Tutorial Purpose

This tutorial demonstrates common usages of the [Intercept Layer for OpenCL Applications](https://github.com/intel/opencl-intercept-layer) to debug and optimize an OpenCL program.
The initial version of this program is syntactically correct and compiles, but it crashes, has bugs, and it is slow!
In this tutorial we will fix the bugs and likely improve the application performance.
When we are done we will generate a cool [sine-based Julia Set](http://paulbourke.net/fractals/sinjulia/) fractal.

![Sin-Based Julia Set Image](sinjulia.png)

This tutorial has multiple parts, with each part building upon the previous part.
Try to solve each part on your own, but if you get stuck solutions are provided for each part.

* Part 0: [Building and Running the Tutorial](part0.md)
* Part 1: [Fixing an OpenCL Error](part1.md)
* Part 2: [Fixing an OpenCL Program Build Error](part2.md)
* Part 3: [Fixing an OpenCL Map Error](part3.md)
* Part 4: [Profiling and Improving Performance](part4.md)
* Part 5: [Improving Performance More](part5.md)
* Part 6: [Final Words and Additional Things to Try](part6.md)

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `-i <number>` | 16 | Specify the number of iterations to execute.
| `--gwx <number>` | TBD! | Specify the global work size to execute, in the X direction.  This also determines the width of the generated image.
| `--gwy <number>` | TBD! | Specify the global work size to execute, in the Y direction.  This also determines the height of the generated image.
| `--lwx <number>` | 0 | Specify the local work-group size in the X direction.  If either local work-group size is zero a `NULL` local work-group size is used.
| `--lwy <number>` | 0 | Specify the local work-group size in the Y direction.  If either local work-group size is zero a `NULL` local work-group size is used.
