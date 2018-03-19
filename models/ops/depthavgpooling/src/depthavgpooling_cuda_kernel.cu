#include "depthavgpooling_cuda_kernel.h"

#include <cstdio>

#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename Dtype, typename Acctype>
__global__ void AvePoolForward_kernel(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const bottom_data_depth,
    const int channels, const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data, Dtype* const depth_weight_count) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    Dtype pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Acctype aveval = Acctype(0);
    const Dtype* const bottom_slice = bottom_data + (n * channels + c) * height * width;

    int ih = (hstart + hend) / 2;
    int iw = (wstart + wend) / 2;
    Acctype Di = bottom_data_depth[ih * width + iw];
    Acctype divcount = 0.;

    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        Acctype Dval = bottom_data_depth[h * width + w];
        Acctype weight_val = exp(-abs(Di - Dval));
//        divcount += weight_val;
        pool_size -= (1. - weight_val);
        aveval += (bottom_slice[h * width + w] * weight_val);
      }
    }
    depth_weight_count[ih * width + iw] = pool_size;//divcount;
    top_data[index] = Dtype(aveval / pool_size);//((hend - hstart) * (wend - wstart)));//ScalarConvert<Acctype, Dtype>::to(aveval / ((hend - hstart) * (wend - wstart)));
  }
}

void AvePoolForward(cudaStream_t stream, const int count,
    const float* const input_data, const float* const input_depth_data, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    float* const top_data, float* const depth_weight_count) {

  AvePoolForward_kernel<float, double>
      <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>(
        count, input_data, input_depth_data,
        channels, height, width, pooled_height, pooled_width,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, top_data, depth_weight_count);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in AvePoolFoward: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
  }
}


template <typename Dtype, typename Acctype>
__global__ void AvePoolBackward_kernel(const int nthreads, const Dtype* const top_diff, const Dtype* const bottom_data_depth, const Dtype* const depth_weight_count,
    const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Acctype gradient = Acctype(0);
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;

    bool valid = true;
    Acctype Dval = 0.;
    if (h<0 || h> height || w<0 ||w>width)
      valid = false;
    else
      Dval = bottom_data_depth[h * width + w];

    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        Dtype weight_count = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, height);
        wend = min(wend, width);

        int ih = (hstart + hend) / 2;
        int iw = (wstart + wend) / 2;
        Acctype Di = bottom_data_depth[ih * width + iw];
        Acctype weight_val = 1.;//
        if(valid && depth_weight_count[ih * width + iw]==0){
            weight_val = exp(-abs(Di - Dval));
            weight_count = depth_weight_count[ih * width + iw];
        }

        gradient += top_diff_slice[ph * pooled_width + pw] * weight_val / weight_count;//((hend - hstart) * (wend - wstart));
      }
    }
    bottom_diff[index] = Dtype(gradient);//ScalarConvert<Acctype, Dtype>::to(gradient);
  }
}



void AvePoolBackward(cudaStream_t stream, const int count, const float* const gradOutput,const float* const input_depth,const float* const depth_weight_count,
    const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    float* const bottom_diff) {

    AvePoolBackward_kernel<float, double>
      <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count,
        gradOutput, input_depth, depth_weight_count,
        channels, height, width, pooled_height, pooled_width,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
        bottom_diff);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in AvePoolBackward: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
  }
}
