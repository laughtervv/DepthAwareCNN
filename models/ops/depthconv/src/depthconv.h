int depthconv_forward(THFloatTensor *input, THFloatTensor *offset,
                        THFloatTensor *output);
int depthconv_backward(THFloatTensor *grad_output, THFloatTensor *grad_input,
                         THFloatTensor *grad_offset);
