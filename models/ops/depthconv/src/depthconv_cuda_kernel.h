template <typename DType>
void depthconv_im2col(cudaStream_t stream, const DType *data_im,
                       const DType *data_depth, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w, DType *data_col);

template <typename DType>
void depthconv_col2im(cudaStream_t stream, const DType *data_col,
                       const DType *data_depth, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w, DType *grad_im);

