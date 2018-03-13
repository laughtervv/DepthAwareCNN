#include "depthconv_cuda_kernel.h"

#include <cstdio>

#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename DType>
__device__ DType get_gradient_weight(DType argmax_h, DType argmax_w,
                                     const int h, const int w, const int height,
                                     const int width) {

  if (argmax_h < 0 || argmax_h > height || argmax_w < 0 || argmax_w > width) {
    // empty
    return 0;
  }

  argmax_h = max(argmax_h, (DType)0.0f);
  argmax_w = max(argmax_w, (DType)0.0f);

  int argmax_h_low = (int)argmax_h;
  int argmax_w_low = (int)argmax_w;
  int argmax_h_high;
  int argmax_w_high;
  if (argmax_h_low >= height - 1) {
    argmax_h_high = argmax_h_low = height - 1;
    argmax_h = (DType)argmax_h_low;
  } else {
    argmax_h_high = argmax_h_low + 1;
  }
  if (argmax_w_low >= width - 1) {
    argmax_w_high = argmax_w_low = width - 1;
    argmax_w = (DType)argmax_w_low;
  } else {
    argmax_w_high = argmax_w_low + 1;
  }
  DType weight = 0;
  if (h == argmax_h_low) {
    if (w == argmax_w_low) {
      weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
    } else if (w == argmax_w_high) {
      weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
    }
  } else if (h == argmax_h_high) {
    if (w == argmax_w_low) {
      weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
    } else if (w == argmax_w_high) {
      weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
    }
  }
  return weight;
}

template <typename DType>
__global__ void depthconv_im2col_gpu_kernel(
    const int n, const DType *data_im, const DType *data_depth,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int height_col,
    const int width_col, DType *data_col) {
  // CxHxW --> (khxkw)x(CxHxW) 
  CUDA_KERNEL_LOOP(index, n) {
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int c_im = (index / width_col) / height_col;
    const int c_col = c_im * kernel_h * kernel_w;


    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    DType *data_col_ptr = data_col + (c_col * height_col + h_col) * width_col + w_col;
    const DType *data_im_ptr = data_im + (c_im * height + h_in) * width + w_in;
    const DType *data_depth_ptr = data_depth + h_in * width + w_in;
    DType Di = 0.;
    bool valid = true;
    if ((h_in + dilation_h * (kernel_h - 1) / 2)>=0 &&
         w_in  + dilation_w * (kernel_w - 1) / 2 >= 0 &&
         (h_in + dilation_h * (kernel_h - 1) / 2) < height &&
         w_in  + dilation_w * (kernel_w - 1) / 2 < width)
        Di = data_depth[(h_in + dilation_h * (kernel_h - 1) / 2) * width + w_in  + dilation_w * (kernel_w - 1) / 2];
    else
        valid = false;
    //const DType Di = data_depth[(h_in + (kernel_h - 1) / 2 + dilation_h - 1) * width + (w_in + (kernel_w - 1) / 2 + dilation_w - 1)]; 

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        DType val = static_cast<DType>(0);
        DType Dval = static_cast<DType>(0);
        const int h_im = h_in + i * dilation_h;
        const int w_im = w_in + j * dilation_w;
        if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
          const int map_h = i * dilation_h;
          const int map_w = j * dilation_w;
          val = data_im_ptr[map_h * width + map_w];
          if (valid)
            Dval = data_depth_ptr[map_h * width + map_w];
	  //printf("%f,%d\n",Dval,h_in * width + w_in+map_h * width + map_w - ((h_in + (kernel_h - 1) / 2 + dilation_h - 1) * width + (w_in + (kernel_w - 1) / 2 + dilation_w - 1)));
          // printf("Di-Dval: %f, %f\n", Di, Dval);
	  // if (exp(-abs(Di - Dval))<0.2)
	  //	printf("Di-Dval: %f\n", exp(-abs(Di - Dval)));
          val *= exp(-abs(Di - Dval));
        }
        *data_col_ptr = val;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

template <typename DType>
void depthconv_im2col(cudaStream_t stream, const DType *data_im, const DType *data_depth, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w, DType *data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // Launch
  depthconv_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0,
                                 stream>>>(
      num_kernels, data_im, data_depth, height, width, ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col, width_col, data_col);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in depthconv_im2col: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
  }
}

template void depthconv_im2col<float>(
    cudaStream_t stream, const float *data_im, const float *data_depth, 
    const int channels, const int height, const int width, const int ksize_h,
    const int ksize_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w, float *data_col);

/*template void depthconv_im2col<double>(
    cudaStream_t stream, const double *data_im, const double *data_depth,
    const int channels, const int height, const int width, const int ksize_h,
    const int ksize_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w, double *data_col);*/

template <typename DType>
__global__ void depthconv_col2im_gpu_kernel(
    const int n, const DType *data_col, const DType *data_depth,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w, const int height_col,
    const int width_col, DType *grad_im) {
  CUDA_KERNEL_LOOP(index, n) {
    for (int ii = 0; ii < kernel_h * kernel_w; ii++){
      int ii_index = ii + index * kernel_h * kernel_w;
      const int j = (ii_index / width_col / height_col) % kernel_w;
      const int i = (ii_index / width_col / height_col / kernel_w) % kernel_h;
      const int c = ii_index / width_col / height_col / kernel_w / kernel_h;
      // compute the start and end of the output

      int w_out = ii_index % width_col;
      int h_out = (ii_index / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;

      //const DType cur_inv_h_data = h_in + i * dilation_h;
      //const DType cur_inv_w_data = w_in + j * dilation_w;

      const DType cur_top_grad = data_col[ii_index];
      const int cur_h = h_in + i * dilation_h;//(int)cur_inv_h_data;
      const int cur_w = w_in + j * dilation_w;//(int)cur_inv_w_data;

       DType Di = 0.;
      bool valid = true;
      if ((h_in + dilation_h * (kernel_h - 1) / 2)>=0 &&
             w_in  + dilation_w * (kernel_w - 1) / 2 >= 0 &&
             (h_in + dilation_h * (kernel_h - 1) / 2) < height &&
             w_in  + dilation_w * (kernel_w - 1) / 2 < width)
            Di = data_depth[(h_in + dilation_h * (kernel_h - 1) / 2) * width + w_in  + dilation_w * (kernel_w - 1) / 2];
      else
            valid = false;
//      const DType Di = data_depth[(h_in + dilation_h * (kernel_h - 1) / 2) * width + w_in  + dilation_w * (kernel_w - 1) / 2];
      //const DType Di = data_depth[(h_in + (kernel_h - 1) / 2 + dilation_h - 1) * width + w_in  + (kernel_w - 1) / 2 + dilation_w - 1];
      //printf("%d\n",(h_in + dilation_h * (kernel_h - 1) / 2) * width + w_in  + dilation_w * (kernel_w - 1) / 2);
      //data_depth[cur_h * width + cur_w];
      // data_depth[(h_in + (kernel_h - 1) / 2 + dilation_h - 1) * width + w_in  + (kernel_w - 1) / 2 + dilation_w - 1];
      

      int cur_bottom_grad_pos =
          (c * height + cur_h) * width + cur_w;
      int cur_bottom_depth_pos=
          (cur_h) * width + cur_w;
      //printf("%d,%d,%d,%d\n",i,j,((h_in + dilation_h * (kernel_h - 1) / 2) * width + w_in  + dilation_w * (kernel_w - 1) / 2-cur_bottom_depth_pos),dilation_h);
      //printf("%d\n",((h_in + dilation_h * (kernel_h - 1) / 2) * width + w_in  + dilation_w * (kernel_w - 1) / 2-cur_bottom_depth_pos));
      DType Dval = 0.;
      if (valid)
        Dval = data_depth[cur_bottom_depth_pos];
      if (cur_h >= 0 && cur_h < height && cur_w  >= 0 &&
              cur_w  < width)
        atomicAdd(grad_im + cur_bottom_grad_pos, cur_top_grad * exp(-abs(Di - Dval)));
      
    }
  }
}

template <typename DType>
void depthconv_col2im(cudaStream_t stream, const DType *data_col,
                       const DType *data_depth, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w, DType *grad_im) {

  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // int channel_per_depthconv_group = channels / depthconv_group;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  depthconv_col2im_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0,
                                 stream>>>(
      num_kernels, data_col, data_depth, channels, height, width, ksize_h,
      ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      height_col, width_col, grad_im);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in depthconv_col2im: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
  }
}

template void depthconv_col2im<float>(
    cudaStream_t stream, const float *data_col, const float *data_depth,
    const int channels, const int height, const int width, const int ksize_h,
    const int ksize_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w, float *grad_im);

/*template void depthconv_col2im<double>(
    cudaStream_t stream, const double *data_col, const double *data_depth,
    const int channels, const int height, const int width, const int ksize_h,
    const int ksize_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w, double *grad_im);*/
