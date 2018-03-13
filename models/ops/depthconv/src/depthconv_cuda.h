
int depthconv_forward_cuda(THCudaTensor *input,
                             THCudaTensor *input_depth,
                             THCudaTensor *weight, THCudaTensor * bias, THCudaTensor *output,
                             THCudaTensor *columns, THCudaTensor *ones, int kW,
                             int kH, int dW, int dH, int padW, int padH,
                             int dilationH, int dilationW);

int depthconv_backward_input_cuda(
    THCudaTensor *input, THCudaTensor *input_depth, THCudaTensor *gradOutput,
    THCudaTensor *gradInput, THCudaTensor *weight,
    THCudaTensor *columns, int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationH, int dilationW);

int depthconv_backward_parameters_cuda(
    THCudaTensor *input, THCudaTensor *input_depth, THCudaTensor *gradOutput,
    THCudaTensor *gradWeight, THCudaTensor *gradBias,
    THCudaTensor *columns, THCudaTensor *ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationH, int dilationW,
    float scale);
