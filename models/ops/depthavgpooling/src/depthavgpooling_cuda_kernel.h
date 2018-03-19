
void AvePoolForward(cudaStream_t stream, const int count,
    const float* const input_data, const float* const input_depth_data,const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    float* const top_data, float* const depth_weight_count);

void AvePoolBackward(cudaStream_t stream, const int count, const float* const gradOutput,const float* const input_depth,const float* const depth_weight_count,
    const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    float* const bottom_diff);

