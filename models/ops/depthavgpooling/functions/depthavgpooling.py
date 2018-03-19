import torch
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import cffi
from .._ext import depthavgpooling


def depth_avgpooling( input,
                      depth,
                      kernel_size=3,
                      stride=1,
                      padding=0):

    if input is not None and input.dim() != 4:
        raise ValueError(
            "Expected 4D tensor as input, got {}D tensor instead.".format(
                input.dim()))

    f = DepthavgpoolingFunction(_pair(kernel_size), _pair(stride), _pair(padding))
    return f(input, depth)



class DepthavgpoolingFunction(Function):
    def __init__(self, kernel_size, stride, padding):
        super(DepthavgpoolingFunction, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, input, depth):
        self.save_for_backward(input, depth)
        self.depth = depth
        output = input.new(*self._output_size(input))

        self.depthweightcount = input.new(*(depth.size())).zero_()

        if not input.is_cuda:
            raise NotImplementedError
        else:
            if not isinstance(input, torch.cuda.FloatTensor):
                raise NotImplementedError
            depthavgpooling.depthavgpooling_forward_cuda(
                    input, depth, output, self.depthweightcount,
                    self.kernel_size[1], self.kernel_size[0], self.stride[1], self.stride[0],
                    self.padding[1], self.padding[0])

        return output

    def backward(self, grad_output):
        input, depth, = self.saved_tensors
        grad_input = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if not isinstance(grad_output, torch.cuda.FloatTensor):
                raise NotImplementedError
            if self.needs_input_grad[0]:
                grad_input = input.new(*input.size()).zero_()
                depthavgpooling.depthavgpooling_backward_input_cuda(
                    input, depth, self.depthweightcount,grad_output, grad_input,
                    self.kernel_size[1], self.kernel_size[0], self.stride[1], self.stride[0],
                    self.padding[1], self.padding[0])
        # print 'grad_input',grad_input
        # print 'depthweightcount',self.depthweightcount
        return grad_input, None

    def _output_size(self, input):

        output_size = (input.size(0), input.size(0))
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = self.kernel_size[d]
            stride = self.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "avgpooling input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size
