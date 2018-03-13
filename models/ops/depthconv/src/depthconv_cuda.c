#include <THC/THC.h>

#include "depthconv_cuda_kernel.h"

extern THCState *state;

void shape_check(THCState *state, THCudaTensor *input, THCudaTensor *input_depth,
                 THCudaTensor *gradOutput, THCudaTensor *weight, THCudaTensor *bias, int kH, int kW,
                 int dH, int dW, int padH, int padW, int dilationH,
                 int dilationW) {

  THArgCheck(weight->nDimension == 4, 5,
             "4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, "
             "but got: %s",
             weight->nDimension);

  THArgCheck(THCudaTensor_isContiguous(state, weight), 5,
             "weight tensor has to be contiguous");

  THArgCheck(kW > 0 && kH > 0, 9,
             "kernel size should be greater than zero, but got kH: %d kW: %d",
             kH, kW);

  THArgCheck((weight->size[2] == kH && weight->size[3] == kW), 9,
             "kernel size should be consistent with weight, but got kH: %d kW: %d weight.size(2): %d, weight.size(3): %d", kH,
             kW, weight->size[2], weight->size[3]);

  THArgCheck(dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dH: %d dW: %d", dH,
             dW);

  THArgCheck(
      dilationW > 0 && dilationH > 0, 14,
      "dilation should be greater than 0, but got dilationH: %d dilationW: %d",
      dilationH, dilationW);

  //////////// check bias //////////////////

  THArgCheck(!bias || THCudaTensor_isContiguous(state, bias), 5,
             "bias tensor has to be contiguous");

  if (bias != NULL) {
//    THCUNN_check_dim_size(state, bias, 1, 0, weight->size[0]);
    THArgCheck(bias->nDimension==1, 6,
             "Need bias of dimension %d but got %d", 1, bias->nDimension);
    THArgCheck(bias->size[0]==weight->size[0], 6,
             "Need bias of size %d but got %d", weight->size[0], bias->size[0]);
  }
//////////////////////////////////////////

  int ndim = input->nDimension;
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  THArgCheck(ndim == 3 || ndim == 4, 2,
             "3D or 4D input tensor expected but got: %s", ndim);

  long nInputPlane = weight->size[1];
  long inputHeight = input->size[dimh];
  long inputWidth = input->size[dimw];
  long nOutputPlane = weight->size[0];
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

  if (outputWidth < 1 || outputHeight < 1)
    THError(
        "Given input size: (%ld x %ld x %ld). "
        "Calculated output size: (%ld x %ld x %ld). Output size is too small",
        nInputPlane, inputHeight, inputWidth, nOutputPlane, outputHeight,
        outputWidth);

  THArgCheck((inputHeight >= kH && inputWidth >= kW), 2,
             "input image is smaller than kernel");

/////////check depth map shape /////////

  int ndim_depth = input_depth->nDimension;
  int dimf_depth = 0;
  int dimh_depth = 1;
  int dimw_depth = 2;

  if (ndim_depth == 4) {
    dimf_depth++;
    dimh_depth++;
    dimw_depth++;
  }

  THArgCheck(ndim_depth == 3 || ndim_depth == 4, 3,
             "3D input depth tensor expected but got: %s", ndim);

  long inputHeight_depth = input_depth->size[dimh_depth];
  long inputWidth_depth = input_depth->size[dimw_depth];

  THArgCheck(input_depth->size[1] == 1, 3,
             "input depth should have only 1 channel",
             nInputPlane, input->size[1]);

  THArgCheck((inputHeight == inputHeight_depth && inputWidth == inputWidth_depth), 3,
             "input image and input depth should be the same size");
//////////////////////////////////////////

  if (gradOutput != NULL) {
    THArgCheck(gradOutput->size[dimf] == nOutputPlane, 4,
               "invalid number of gradOutput planes, expected: %d, but got: %d",
               nOutputPlane, gradOutput->size[dimf]);

    THArgCheck((gradOutput->size[dimh] == outputHeight &&
                gradOutput->size[dimw] == outputWidth),
               4, "invalid size of gradOutput, expected height: %d width: %d , but got height: %d width: %d", outputHeight, outputWidth,
               gradOutput->size[dimh], gradOutput->size[dimw]);
  }
}

int depthconv_forward_cuda(THCudaTensor *input, THCudaTensor *input_depth, THCudaTensor *weight, THCudaTensor *bias, THCudaTensor *output,
                             THCudaTensor *columns, THCudaTensor *ones, int kW,
                             int kH, int dW, int dH, int padW, int padH,
                             int dilationH, int dilationW) {

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 7, input, input_depth, weight, output, columns, ones, bias));

  shape_check(state, input, input_depth, NULL, weight, bias, kH, kW, dH, dW, padH, padW,
              dilationH, dilationW);

  input = THCudaTensor_newContiguous(state, input);
  input_depth = THCudaTensor_newContiguous(state, input_depth);
  weight = THCudaTensor_newContiguous(state, weight);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1],
                          input->size[2]);
    THCudaTensor_resize4d(state, input_depth, 1, input_depth->size[0], input_depth->size[1],
                          input_depth->size[2]);
  }

  long batchSize = input->size[0];
  long nInputPlane = input->size[1];
  long inputHeight = input->size[2];
  long inputWidth = input->size[3];

  long nOutputPlane = weight->size[0];

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  bias = bias ? THCudaTensor_newContiguous(state, bias) : bias;
  THCudaTensor_resize4d(state, output, batchSize, nOutputPlane, outputHeight,
                        outputWidth);

  THCudaTensor_resize2d(state, columns, nInputPlane * kW * kH,
                        outputHeight * outputWidth);

  if (ones->nDimension != 2 ||
      ones->size[0] * ones->size[1] < outputHeight * outputWidth) {
    THCudaTensor_resize2d(state, ones, outputHeight, outputWidth);
    THCudaTensor_fill(state, ones, 1);
  }

  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *depth_n = THCudaTensor_new(state);
  THCudaTensor *output_n = THCudaTensor_new(state);

  for (int elt = 0; elt < batchSize; elt++) {

    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, depth_n, input_depth, 0, elt);
    THCudaTensor_select(state, output_n, output, 0, elt);


    // Do bias first
     long m_ = nOutputPlane;
     long n_ = outputHeight * outputWidth;
     long k_ = 1;

     if (bias) {
       THCudaBlas_Sgemm(state, 't', 'n', n_, m_, k_, 1.0f,
                        THCudaTensor_data(state, ones), k_,
                        THCudaTensor_data(state, bias), k_, 0.0f,
                        THCudaTensor_data(state, output_n), n_);
     } else {
       THCudaTensor_zero(state, output_n);
     }

    depthconv_im2col(
        THCState_getCurrentStream(state), THCudaTensor_data(state, input_n), THCudaTensor_data(state, depth_n), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THCudaTensor_data(state, columns));

    long m = nOutputPlane;
    long n = columns->size[1];
    long k = nInputPlane * kH * kW;

    THCudaBlas_Sgemm(state, 'n', 'n', n, m, k, 1.0f,
                     THCudaTensor_data(state, columns), n,
                     THCudaTensor_data(state, weight), k, 1.0f,
                     THCudaTensor_data(state, output_n), n);
  }

  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, depth_n);
  THCudaTensor_free(state, output_n);

  if (batch == 0) {
    THCudaTensor_resize3d(state, output, nOutputPlane, outputHeight,
                          outputWidth);
    THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
  }

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, input_depth);
  THCudaTensor_free(state, weight);

  if (bias) THCudaTensor_free(state, bias);

  return 1;
}

int depthconv_backward_input_cuda(
    THCudaTensor *input, THCudaTensor *input_depth, THCudaTensor *gradOutput,
    THCudaTensor *gradInput, THCudaTensor *weight,
    THCudaTensor *columns, int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationH, int dilationW) {

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 6, input, input_depth, gradOutput, weight, columns, gradInput));

  shape_check(state, input, input_depth, gradOutput, weight, NULL, kH, kW, dH, dW, padH,
              padW, dilationH, dilationW);

  input = THCudaTensor_newContiguous(state, input);
  input_depth = THCudaTensor_newContiguous(state, input_depth);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  weight = THCudaTensor_newContiguous(state, weight);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1],
                          input->size[2]);
    THCudaTensor_resize4d(state, gradOutput, 1, gradOutput->size[0],
                          gradOutput->size[1], gradOutput->size[2]);
  }

  long batchSize = input->size[0];
  long nInputPlane = input->size[1];
  long inputHeight = input->size[2];
  long inputWidth = input->size[3];

  long nOutputPlane = weight->size[0];

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  THArgCheck((input_depth->size[0] == batchSize), 3, "invalid batch size of input depth");

  THCudaTensor_resize4d(state, gradInput, batchSize, nInputPlane, inputHeight,
                        inputWidth);

  THCudaTensor_resize2d(state, columns, nInputPlane * kW * kH,
                        outputHeight * outputWidth);

//  printf("columns size: %d,%d\n", columns->size[0],columns->size[1]);

  THCudaTensor *gradInput_n = THCudaTensor_new(state);
  THCudaTensor *input_depth_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  for (int elt = 0; elt < batchSize; elt++) {
    THCudaTensor_select(state, gradInput_n, gradInput, 0, elt);
    THCudaTensor_select(state, input_depth_n, input_depth, 0, elt);
    THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    long m = nInputPlane * kW * kH;
    long n = columns->size[1];
    long k = nOutputPlane;

    THCudaBlas_Sgemm(state, 'n', 't', n, m, k, 1.0f,
                     THCudaTensor_data(state, gradOutput_n), n,
                     THCudaTensor_data(state, weight), m, 0.0f,
                     THCudaTensor_data(state, columns), n);

    depthconv_col2im(
        THCState_getCurrentStream(state), THCudaTensor_data(state, columns),
        THCudaTensor_data(state, input_depth_n), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THCudaTensor_data(state, gradInput_n));
  }

  THCudaTensor_free(state, gradInput_n);
  THCudaTensor_free(state, input_depth_n);
  THCudaTensor_free(state, gradOutput_n);

  if (batch == 0) {
    THCudaTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight,
                          outputWidth);
    THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
    THCudaTensor_resize3d(state, input_depth, 1, inputHeight, inputWidth);
    THCudaTensor_resize3d(state, gradInput, nInputPlane, inputHeight,
                          inputWidth);
  }

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, input_depth);
  THCudaTensor_free(state, gradOutput);
  THCudaTensor_free(state, weight);

  return 1;
}

int depthconv_backward_parameters_cuda(
    THCudaTensor *input, THCudaTensor *input_depth, THCudaTensor *gradOutput,
    THCudaTensor *gradWeight, THCudaTensor *gradBias,
    THCudaTensor *columns, THCudaTensor *ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationH, int dilationW,
    float scale) {

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 7, input, input_depth, gradOutput,
                                         gradWeight, gradBias, columns, ones));

  shape_check(state, input, input_depth, gradOutput, gradWeight, gradBias, kH, kW, dH, dW,
              padH, padW, dilationH, dilationW);

  input = THCudaTensor_newContiguous(state, input);
  input_depth = THCudaTensor_newContiguous(state, input_depth);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1],
                          input->size[2]);
    THCudaTensor_resize4d(state, gradOutput, 1, gradOutput->size[0],
                          gradOutput->size[1], gradOutput->size[2]);
  }

  long batchSize = input->size[0];
  long nInputPlane = input->size[1];
  long inputHeight = input->size[2];
  long inputWidth = input->size[3];

  long nOutputPlane = gradWeight->size[0];

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;


  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 2 ||
      ones->size[0] * ones->size[1] < outputHeight * outputWidth) {
    THCudaTensor_resize2d(state, ones, outputHeight, outputWidth);
    THCudaTensor_fill(state, ones, 1);
  }

  THCudaTensor_resize2d(state, columns, nInputPlane * kW * kH,
                        outputHeight * outputWidth);

  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *depth_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  for (int elt = 0; elt < batchSize; elt++) {
    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, depth_n, input_depth, 0, elt);
    THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    depthconv_im2col(
        THCState_getCurrentStream(state), THCudaTensor_data(state, input_n),
        THCudaTensor_data(state, depth_n), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THCudaTensor_data(state, columns));

    long m = nOutputPlane;
    long n = nInputPlane * kW * kH;
    long k = columns->size[1];

    THCudaBlas_Sgemm(state, 't', 'n', n, m, k, scale,
                     THCudaTensor_data(state, columns), k,
                     THCudaTensor_data(state, gradOutput_n), k, 1.0f,
                     THCudaTensor_data(state, gradWeight), n);


    // Do Bias:
    // M,N,K are dims of matrix A and B
    long m_ = nOutputPlane;
    long k_ = outputHeight * outputWidth;

    // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
    if (gradBias)
        THCudaBlas_Sgemv(
          state,
          't',
          k_, m_,
          scale,
          THCudaTensor_data(state, gradOutput_n), k_,
          THCudaTensor_data(state, ones), 1, 1.0f,
          THCudaTensor_data(state, gradBias), 1);


  }

  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, depth_n);
  THCudaTensor_free(state, gradOutput_n);

  if (batch == 0) {
    THCudaTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight,
                          outputWidth);
    THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
  }

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, input_depth);
  THCudaTensor_free(state, gradOutput);
  return 1;
}
