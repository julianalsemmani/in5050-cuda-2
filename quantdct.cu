#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "common.h"
#include "tables.h"

#define ISQRT2 0.70710678118654f

__constant__ uint8_t zigzag_U_d[64] =
{
  0,
  1, 0,
  0, 1, 2,
  3, 2, 1, 0,
  0, 1, 2, 3, 4,
  5, 4, 3, 2, 1, 0,
  0, 1, 2, 3, 4, 5, 6,
  7, 6, 5, 4, 3, 2, 1, 0,
  1, 2, 3, 4, 5, 6, 7,
  7, 6, 5, 4, 3, 2,
  3, 4, 5, 6, 7,
  7, 6, 5, 4,
  5, 6, 7,
  7, 6,
  7,
};

__constant__ uint8_t zigzag_V_d[64] =
{
  0,
  0, 1,
  2, 1, 0,
  0, 1, 2, 3,
  4, 3, 2, 1, 0,
  0, 1, 2, 3, 4, 5,
  6, 5, 4, 3, 2, 1, 0,
  0, 1, 2, 3, 4, 5, 6, 7,
  7, 6, 5, 4, 3, 2, 1,
  2, 3, 4, 5, 6, 7,
  7, 6, 5, 4, 3,
  4, 5, 6, 7,
  7, 6, 5,
  6, 7,
  7,
};

__constant__ float dctlookup_d[8][8] =
{
  {1.0f,  0.980785f,  0.923880f,  0.831470f,  0.707107f,  0.555570f,  0.382683f,  0.195090f, },
  {1.0f,  0.831470f,  0.382683f, -0.195090f, -0.707107f, -0.980785f, -0.923880f, -0.555570f, },
  {1.0f,  0.555570f, -0.382683f, -0.980785f, -0.707107f,  0.195090f,  0.923880f,  0.831470f, },
  {1.0f,  0.195090f, -0.923880f, -0.555570f,  0.707107f,  0.831470f, -0.382683f, -0.980785f, },
  {1.0f, -0.195090f, -0.923880f,  0.555570f,  0.707107f, -0.831470f, -0.382683f,  0.980785f, },
  {1.0f, -0.555570f, -0.382683f,  0.980785f, -0.707107f, -0.195090f,  0.923880f, -0.831470f, },
  {1.0f, -0.831470f,  0.382683f,  0.195090f, -0.707107f,  0.980785f, -0.923880f,  0.555570f, },
  {1.0f, -0.980785f,  0.923880f, -0.831470f,  0.707107f, -0.555570f,  0.382683f, -0.195090f, },
};

__device__ static void transpose_block_cu(float *in_data, float *out_data, int i, int j)
{
  out_data[i*8+j] = in_data[j*8+i];
}

static void transpose_block(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    for (j = 0; j < 8; ++j)
    {
      out_data[i*8+j] = in_data[j*8+i];
    }
  }
}

__device__ static void dct_1d(float *in_data, float *out_data, int i)
{

  float dct = 0;

  for (int j = 0; j < 8; ++j)
  {
    dct += in_data[j] * dctlookup_d[j][i];
    printf("dct: %f - in_data: %f - lookup: %f\n", dct, in_data[j], dctlookup_d[j][i]);
  }

  out_data[i] = dct;
  
}

static void idct_1d(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    float idct = 0;

    for (j = 0; j < 8; ++j)
    {
      idct += in_data[j] * dctlookup[i][j];
    }

    out_data[i] = idct;
  }
}

static void scale_block(float *in_data, float *out_data)
{
  int u, v;

  for (v = 0; v < 8; ++v)
  {
    for (u = 0; u < 8; ++u)
    {
      float a1 = !u ? ISQRT2 : 1.0f;
      float a2 = !v ? ISQRT2 : 1.0f;

      /* Scale according to normalizing function */
      out_data[v*8+u] = in_data[v*8+u] * a1 * a2;
    }
  }
}


__device__ static void scale_block_cu(float *in_data, float *out_data, int u, int v)
{
  float a1 = !u ? ISQRT2 : 1.0f;
  float a2 = !v ? ISQRT2 : 1.0f;

  /* Scale according to normalizing function */
  out_data[v*8+u] = in_data[v*8+u] * a1 * a2;
}


static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[v*8+u];

    /* Zig-zag and quantize */
    out_data[zigzag] = (float) round((dct / 4.0) / quant_tbl[zigzag]);
  }
}

__device__ static void quantize_block_cu(float *in_data, float *out_data, uint8_t *quant_tbl, int i, int j)
{
  int zigzag = j*8+i;
  
  uint8_t u = zigzag_U_d[zigzag];
  uint8_t v = zigzag_V_d[zigzag];

  float dct = in_data[v*8+u];

  /* Zig-zag and quantize */
  out_data[zigzag] = (float) round((dct / 4.0) / quant_tbl[zigzag]);
}


static void dequantize_block(float *in_data, float *out_data,
    uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[zigzag];

    /* Zig-zag and de-quantize */
    out_data[v*8+u] = (float) round((dct * quant_tbl[zigzag]) / 4.0);
  }
}

__global__ static void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl)
{
  out_data += blockIdx.x * 8;
  in_data += blockIdx.x * 8 * 8;

  __shared__ float mb[8*8] __attribute((aligned(16)));
  __shared__ float mb2[8*8] __attribute((aligned(16)));

  int i = threadIdx.x;
  int j = threadIdx.y;

  // < 64
  mb2[j*8+i] = in_data[j*8+i];

  __syncthreads();

  // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
  //   for (int i = 0; i < 8; i++) {
  //     for (int j = 0; j < 8; j++) {
  //       printf("%f,", mb2[i]);
  //     }
  //     printf("\n");
  //   }
  // }

  // __syncthreads();
  /* Two 1D DCT operations with transpose */

  dct_1d(mb2+i*8, mb+i*8, j); 
  __syncthreads();

  transpose_block_cu(mb, mb2, i, j);

  __syncthreads();

  // < 8
  dct_1d(mb2+i*8, mb+i*8, j); 
  __syncthreads();

  transpose_block_cu(mb, mb2, i, j);

  __syncthreads();

  scale_block_cu(mb2, mb, i, j);

  __syncthreads();

  quantize_block_cu(mb, mb2, quant_tbl, i, j);

  __syncthreads();

  // < 64
  out_data[j*8+i] = mb2[j*8+i]; 
}

static void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  int i, v;

  for (i = 0; i < 64; ++i) { mb[i] = in_data[i]; }

  dequantize_block(mb, mb2, quant_tbl);
  scale_block(mb2, mb);

  /* Two 1D inverse DCT operations with transpose */
  for (v = 0; v < 8; ++v) { idct_1d(mb+v*8, mb2+v*8); }
  transpose_block(mb2, mb);
  for (v = 0; v < 8; ++v) { idct_1d(mb+v*8, mb2+v*8); }
  transpose_block(mb2, mb);

  for (i = 0; i < 64; ++i) { out_data[i] = mb[i]; }
}

static void dequantize_idct_row(int16_t *in_data, uint8_t *prediction, int w, int h,
    int y, uint8_t *out_data, uint8_t *quantization)
{
  int x;

  int16_t block[8*8];

  /* Perform the dequantization and iDCT */
  for(x = 0; x < w; x += 8)
  {
    int i, j;

    dequant_idct_block_8x8(in_data+(x*8), block, quantization);

    for (i = 0; i < 8; ++i)
    {
      for (j = 0; j < 8; ++j)
      {
        /* Add prediction block. Note: DCT is not precise -
           Clamp to legal values */
        int16_t tmp = block[i*8+j] + (int16_t)prediction[i*w+j+x];

        if (tmp < 0) { tmp = 0; }
        else if (tmp > 255) { tmp = 255; }

        out_data[i*w+j+x] = tmp;
      }
    }
  }
}

static void dct_quantize_row(uint8_t *in_data, uint8_t *prediction, int w, int h, int16_t *out_data, uint8_t *quantization)
{
  int x;
  int col = 0;

  int16_t block[(w/8)*(8*8)];

  int16_t *block_d;

  cudaMalloc((void **)&block_d, sizeof(int16_t)*(w/8)*(8*8));

  /* Perform the DCT and quantization */
  for(x = 0; x < w; x += 8)
  {
    int i, j;

    for (i = 0; i < 8; ++i)
    {
      for (j = 0; j < 8; ++j)
      {
        // printf("%d - %d\n", col*8*8+i*8+j, (w/8)*(8*8));
        block[col*8*8+i*8+j] = ((int16_t)in_data[i*w+j+x] - prediction[i*w+j+x]);
      }
    }
    col++;
  }

  cudaMemcpy(block_d, block, sizeof(int16_t)*(w/8)*(8*8), cudaMemcpyHostToDevice);

  dim3 blockDim(8, 8);

  /* Store MBs linear in memory, i.e. the 64 coefficients are stored
    continous. This allows us to ignore stride in DCT/iDCT and other
    functions. */
  dct_quant_block_8x8<<<w/8, blockDim>>>(block_d, out_data, quantization);

  cudaDeviceSynchronize();

  cudaFree(block_d);
}

void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, uint8_t *out_data, uint8_t *quantization)
{
  int y;

  for (y = 0; y < height; y += 8)
  {
    dequantize_idct_row(in_data+y*width, prediction+y*width, width, height, y,
        out_data+y*width, quantization);
  }
}

void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height, int imgWidth, int imgHeight, int16_t *out_data, uint8_t *quantization)
{
  int y;
  uint8_t *quantization_d;
  // , *in_data_d, *prediction_d;
  int16_t *out_data_d;

  cudaMalloc((void **)&quantization_d, sizeof(uint8_t)*imgWidth*imgHeight);
  // cudaMalloc((void **)&in_data_d, sizeof(uint8_t)*width*height);
  // cudaMalloc((void **)&prediction_d, sizeof(uint8_t)*width*height);
  cudaMalloc((void **)&out_data_d, sizeof(int16_t)*imgWidth*imgHeight);

  cudaMemcpy(quantization_d, quantization, sizeof(uint8_t)*imgWidth*imgHeight, cudaMemcpyHostToDevice);
  // cudaMemcpy(in_data_d, in_data, sizeof(uint8_t)*width*height, cudaMemcpyHostToDevice);
  // cudaMemcpy(prediction_d, prediction, sizeof(uint8_t)*width*height, cudaMemcpyHostToDevice);
  cudaMemcpy(out_data_d, out_data, sizeof(int16_t)*imgWidth*imgHeight, cudaMemcpyHostToDevice);

  for (y = 0; y < height; y += 8)
  {
    dct_quantize_row(in_data+y*width, prediction+y*width, width, height, out_data_d+y*width, quantization_d);
  }

  cudaMemcpy(out_data, out_data_d, sizeof(int16_t)*imgWidth*imgHeight, cudaMemcpyDeviceToHost);

  cudaFree(quantization_d);
  cudaFree(out_data_d);
  // cudaFree(in_data_d);
  // cudaFree(prediction_d);
}

