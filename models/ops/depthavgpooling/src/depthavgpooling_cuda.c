#include <THC/THC.h>

#include "depthavgpooling_cuda_kernel.h"

extern THCState *state;


void shape_check(THCState *state,
  THCudaTensor *input, THCudaTensor *input_depth,THCudaTensor *depthweightcount, THCudaTensor *gradOutput,
  int kH, int kW, int dH, int dW, int padH, int padW) {

  THArgCheck(kW > 0 && kH > 0, 5,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 8,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);

  int ndim = input->nDimension;
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

//  THCUNN_argCheck(state, ndim == 3 || ndim == 4, 2, input,
//                  "3D or 4D input tensor expected but got: %s");

  THArgCheck(ndim == 3 || ndim == 4, 2,
             "3D or 4D input tensor expected but got: %d",
             ndim);
//  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2,
//             "pad should be smaller than half of kernel size, but got "
//             "padW = %d, padH = %d, kW = %d, kH = %d",
//             padW, padH, kW, kH);

  long nInputPlane = input->size[dimh-1];
  long nInputRows = input->size[dimh];
  long nInputCols = input->size[dimw];
  long nOutputRows, nOutputCols;
  long nOutputPlane = nInputPlane;


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

  THArgCheck((nInputRows == inputHeight_depth && nInputCols == inputWidth_depth), 3,
             "input image and input depth should be the same size, but got: weightcount(%d,%d), depth(%d,%d)",
             nInputRows, inputHeight_depth, nInputCols, inputWidth_depth);

  if (depthweightcount!=NULL){
      THArgCheck(depthweightcount->size[1] == 1, 3,
                 "input depth should have only 1 channel",
                 nInputPlane, input->size[1]);

      THArgCheck((inputHeight_depth == depthweightcount->size[2] && inputWidth_depth == depthweightcount->size[3]), 3,
                 "input depth and input depthweightcount should be the same size, but got: weightcount(%d,%d), depth(%d,%d)",
                 depthweightcount->size[dimh_depth], depthweightcount->size[dimw_depth], inputHeight_depth, inputWidth_depth);
  }
//////////////////////////////////////////

    nOutputCols = floor(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
    nOutputRows = floor(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((nOutputRows - 1)*dH >= nInputRows + padH)
      --nOutputRows;
    if ((nOutputCols  - 1)*dW >= nInputCols  + padW)
      --nOutputCols;
  }

  if (nOutputCols < 1 || nOutputRows < 1)
    THError("Given input size: (%dx%dx%d). "
            "Calculated output size: (%dx%dx%d). Output size is too small",
            nInputPlane,nInputRows,nInputCols,nInputPlane,nOutputRows,nOutputCols);

  if (gradOutput != NULL) {
//    THCUNN_check_dim_size(state, gradOutput, ndim, dimf, nOutputPlane);
//    THCUNN_check_dim_size(state, gradOutput, ndim, dimh, nOutputRows);
//    THCUNN_check_dim_size(state, gradOutput, ndim, dimw, nOutputCols);

    THArgCheck(gradOutput->size[dimf] == nOutputPlane, 4,
               "invalid number of gradOutput planes, expected: %d, but got: %d",
               nOutputPlane, gradOutput->size[dimf]);

    THArgCheck((gradOutput->size[dimh] == nOutputRows &&
                gradOutput->size[dimw] == nOutputCols),
               4, "invalid size of gradOutput, expected height: %d width: %d , but got height: %d width: %d", nOutputRows, nOutputCols,
               gradOutput->size[dimh], gradOutput->size[dimw]);
  }
  }


int depthavgpooling_forward_cuda(THCudaTensor *input,
           THCudaTensor *input_depth,
           THCudaTensor *output,
           THCudaTensor *depthweightcount,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) {
  THCAssertSameGPU(THCudaTensor_checkGPU(state, 4, input, input_depth, output, depthweightcount));
  shape_check(state, input, input_depth, NULL, NULL, kH, kW, dH, dW,
        padH, padW);

  input = THCudaTensor_newContiguous(state, input);
  input_depth = THCudaTensor_newContiguous(state, input_depth);

  int batch = 1;
  long nInputCols, nInputRows, nInputPlane, batchSize;
  long nOutputCols, nOutputRows;

  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    batchSize = 1;
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THCudaTensor_resize4d(state, input_depth, 1, input_depth->size[0], input_depth->size[1], input_depth->size[2]);
  }
  else
  {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }

  nOutputCols = floor(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
  nOutputRows = floor(float(nInputRows - kH + 2*padH) / float(dH)) + 1;

//  long batchSize = input->size[0];

  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((nOutputRows - 1)*dH >= nInputRows + padH)
      --nOutputRows;
    if ((nOutputCols  - 1)*dW >= nInputCols  + padW)
      --nOutputCols;
  }

//  input = THCudaTensor_newContiguous(state, input);
//  float* input_data = THCudaTensor_data(state, input);
//  float* input_depth_data = THCudaTensor_data(state, input_depth);

  THCudaTensor_resize4d(state, output, batchSize, nInputPlane, nOutputRows, nOutputCols);
  THCudaTensor_resize4d(state, depthweightcount, batchSize, 1, nInputRows, nInputCols);

//  float* output_data = THCudaTensor_data(state, output);
//  float* depthweightcount_data = THCudaTensor_data(state, depthweightcount);


  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *depth_n = THCudaTensor_new(state);
  THCudaTensor *depthweightcount_n = THCudaTensor_new(state);
  THCudaTensor *output_n = THCudaTensor_new(state);

  for (int elt = 0; elt < batchSize; elt++) {

    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, depth_n, input_depth, 0, elt);
    THCudaTensor_select(state, depthweightcount_n, depthweightcount, 0, elt);
    THCudaTensor_select(state, output_n, output, 0, elt);

    int count = THCudaTensor_nElement(state, output_n);

    AvePoolForward(THCState_getCurrentStream(state),
        count, THCudaTensor_data(state, input_n), THCudaTensor_data(state, depth_n),
        nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
        kH, kW, dH, dW, padH, padW, THCudaTensor_data(state, output_n), THCudaTensor_data(state, depthweightcount_n));

    THCudaCheck(cudaGetLastError());
  }

  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, depth_n);
  THCudaTensor_free(state, depthweightcount_n);
  THCudaTensor_free(state, output_n);

  if(batch == 0){
    THCudaTensor_resize3d(state, output, nInputPlane, nOutputRows, nOutputCols);
    THCudaTensor_resize3d(state, input, nInputPlane, nInputRows, nInputCols);
  }

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, input_depth);
}

int depthavgpooling_backward_input_cuda(
           THCudaTensor *input,
           THCudaTensor *input_depth,
           THCudaTensor *depthweightcount,
           THCudaTensor *gradOutput,
           THCudaTensor *gradInput,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) {

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 4, input, input_depth, gradOutput, gradInput, depthweightcount));
  shape_check(state, input, input_depth, depthweightcount, gradOutput, kH, kW, dH, dW,
        padH, padW);

  input = THCudaTensor_newContiguous(state, input);
  input_depth = THCudaTensor_newContiguous(state, input_depth);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  depthweightcount = THCudaTensor_newContiguous(state, depthweightcount);

  long nInputCols, nInputRows, nInputPlane, batchSize;
  long nOutputCols, nOutputRows;
  int dimCol = 2;
  int dimRow = 1;

  int batch = 1;
  if (input->nDimension == 3) {
    nInputPlane = input->size[0];
    batchSize = 1;
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1],input->size[2]);
    THCudaTensor_resize4d(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }
  else
  {
    dimCol = 3;
    dimRow = 2;
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }
  nInputCols = input->size[dimCol];
  nInputRows = input->size[dimRow];

  nOutputCols = floor(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
  nOutputRows = floor(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((nOutputRows - 1)*dH >= nInputRows + padH)
      --nOutputRows;
    if ((nOutputCols  - 1)*dW >= nInputCols  + padW)
      --nOutputCols;
  }

//  THCUNN_check_dim_size(state, gradOutput, input->nDimension, dimRow, nOutputRows);
//  THCUNN_check_dim_size(state, gradOutput, input->nDimension, dimCol, nOutputCols);

  THArgCheck((input_depth->size[0] == batchSize), 3, "invalid batch size of input depth");
  THCudaTensor_resizeAs(state, gradInput, input);

//  float* input_depth_data = THCudaTensor_data(state, input_depth);
//  float* depthweightcount_data = THCudaTensor_data(state, depthweightcount);

  THCudaTensor *gradInput_n = THCudaTensor_new(state);
  THCudaTensor *depth_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);
  THCudaTensor *depthweightcount_n = THCudaTensor_new(state);

  for (int elt = 0; elt < batchSize; elt++) {
    THCudaTensor_select(state, gradInput_n, gradInput, 0, elt);
    THCudaTensor_select(state, depth_n, input_depth, 0, elt);
    THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);
    THCudaTensor_select(state, depthweightcount_n, depthweightcount, 0, elt);

    int count = THCudaTensor_nElement(state, gradInput_n);

    AvePoolBackward
      (THCState_getCurrentStream(state), count,
        THCudaTensor_data(state, gradOutput_n), THCudaTensor_data(state, depth_n), THCudaTensor_data(state, depthweightcount_n),
        nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
        kH, kW, dH, dW, padH, padW,
        THCudaTensor_data(state, gradInput_n));
    THCudaCheck(cudaGetLastError());
  }

  THCudaTensor_free(state, gradInput_n);
  THCudaTensor_free(state, depth_n);
  THCudaTensor_free(state, gradOutput_n);
  THCudaTensor_free(state, depthweightcount_n);
  if (batch == 0) {
    THCudaTensor_resize3d(state, gradOutput, nInputPlane, nOutputRows, nOutputCols);
    THCudaTensor_resize3d(state, input, nInputPlane, nInputRows, nInputCols);
    THCudaTensor_resize3d(state, input_depth, 1, nInputRows, nInputCols);
    THCudaTensor_resize3d(state, gradInput, nInputPlane, nInputRows,nInputCols);
  }

  // clean
  THCudaTensor_free(state, input);
  THCudaTensor_free(state, input_depth);
  THCudaTensor_free(state, depthweightcount);
  THCudaTensor_free(state, gradOutput);
}
