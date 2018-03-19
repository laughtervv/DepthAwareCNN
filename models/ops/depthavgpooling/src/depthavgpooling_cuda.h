
int depthavgpooling_forward_cuda(THCudaTensor *input,
           THCudaTensor *input_depth,
           THCudaTensor *output,
           THCudaTensor *depthweightcount,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) ;

int depthavgpooling_backward_input_cuda(
           THCudaTensor *input,
           THCudaTensor *input_depth,
           THCudaTensor *depthweightcount,
           THCudaTensor *gradOutput,
           THCudaTensor *gradInput,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) ;
