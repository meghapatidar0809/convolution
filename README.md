# Convolution Implementations

### Overview
The convolution uses a stride of 2, zero padding, and assumes average pooling. The input data and filter weights are randomly initialized. The main goal is to implement and compare the two convolution methods.

![](convolution/image.jpg)

### General Features
- **Naive 7-Loop Implementation**: This implementation uses a straightforward, nested-loop structure to perform the convolution. The loops iterate through the output feature maps, channels, height, and width, as well as the filter's height and width. This method is easy to understand but less performant for large-scale operations.

- **Flattened Implementation**: This approach reframes the convolution as a matrix multiplication. The input feature maps are "flattened" into a matrix and the convolution filters are also reshaped. The convolution is then performed as a single matrix multiplication, which can be highly optimized by modern libraries. This method is generally more efficient for hardware accelerators.

