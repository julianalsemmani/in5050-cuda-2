#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "me.h"
#include "tables.h"

__device__ static void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
  int u, v;

  *result = 0;

  for (v = 0; v < 8; ++v)
  {
    for (u = 0; u < 8; ++u)
    {
      *result += abs(block2[v*stride+u] - block1[v*stride+u]);
    }
  }
}

/* Motion estimation for 8x8 block */
__global__ static void me_block_8x8(struct c63_common *cm, struct macroblock *mb_gpu, int mb_x, int mb_y,
    uint8_t *orig, uint8_t *ref, int color_component)
{
  struct macroblock *mb = &mb_gpu[mb_y*cm->padw[color_component]/8+mb_x];

  int range = cm->me_search_range;

  /* Quarter resolution for chroma channels. */
  if (color_component > 0) { range /= 2; }

  // int left = mb_x * 8 - range;
  // int top = mb_y * 8 - range;
  // int right = mb_x * 8 + range;
  // int bottom = mb_y * 8 + range;

  int w = cm->padw[color_component];
  int h = cm->padh[color_component];

  /* Make sure we are within bounds of reference frame. TODO: Support partial
     frame bounds. */
  // if (left < 0) { left = 0; }
  // if (top < 0) { top = 0; }
  // if (right > (w - 8)) { right = w - 8; }
  // if (bottom > (h - 8)) { bottom = h - 8; }

  int x = mb_x * 8 + threadIdx.x - range;
  int y = mb_y * 8 + threadIdx.y - range;

  if (x < 0) return;
  if (y < 0) return;
  if (x > w - 8) return;
  if (y > h - 8) return;

  int mx = mb_x * 8;
  int my = mb_y * 8;

  // Store all SADs in a flat array such that we can find the minimun SAD later
  extern __shared__ int sad_array[];

  int sad;
  sad_block_8x8(orig + my*w+mx, ref + y*w+x, w, &sad);

  // Store the SAD for this thread in the appropriate index in the array
  int flattenedThreadIdx = threadIdx.y * blockDim.x + threadIdx.x;
  sad_array[flattenedThreadIdx] = sad;

  __syncthreads();

  // Sequential addressing minimum algorithm
  for(int stride = (blockDim.x*blockDim.y)/2; stride > 1; stride /= 2)
  {
    // Each iteration the amount of threads working will be halved since we compare 2 elements each iteration
    if(flattenedThreadIdx < stride)
    {
      if(sad_array[flattenedThreadIdx] > sad_array[flattenedThreadIdx + stride])
      {
        sad_array[flattenedThreadIdx] = sad_array[flattenedThreadIdx + stride]; 
      }
    }

    __syncthreads();
  }

  if (sad == sad_array[0])
  {
    mb->mv_x = x - mx;
    mb->mv_y = y - my;
    mb->use_mv = 1;
  }

}

void c63_motion_estimate(struct c63_common *cm)
{
  /* Compare this frame with previous reconstructed frame */
  struct c63_common *cm_gpu;
  struct macroblock **mb_gpu;

  cudaMalloc((void **)&cm_gpu, sizeof(struct c63_common));
  cudaMalloc((void **)&mb_gpu, sizeof(void *)*COLOR_COMPONENTS);

  cudaMalloc((void **)&mb_gpu[Y_COMPONENT], sizeof(struct macroblock)*(cm->mb_rows)*(cm->mb_cols));
  cudaMalloc((void **)&mb_gpu[U_COMPONENT], sizeof(struct macroblock)*(cm->mb_rows/2)*(cm->mb_cols/2));
  cudaMalloc((void **)&mb_gpu[V_COMPONENT], sizeof(struct macroblock)*(cm->mb_rows/2)*(cm->mb_cols/2));

  cudaMemcpy(cm_gpu, cm, sizeof(struct c63_common), cudaMemcpyHostToDevice);

  cudaMemcpy(mb_gpu[Y_COMPONENT], cm->curframe->mbs[Y_COMPONENT], sizeof(struct macroblock)*(cm->mb_rows)*(cm->mb_cols), cudaMemcpyHostToDevice);
  cudaMemcpy(mb_gpu[U_COMPONENT], cm->curframe->mbs[U_COMPONENT], sizeof(struct macroblock)*(cm->mb_rows/2)*(cm->mb_cols/2), cudaMemcpyHostToDevice);
  cudaMemcpy(mb_gpu[V_COMPONENT], cm->curframe->mbs[V_COMPONENT], sizeof(struct macroblock)*(cm->mb_rows/2)*(cm->mb_cols/2), cudaMemcpyHostToDevice);

  // cudaMemcpy(&mb_gpu[Y_COMPONENT], cm->mb_rows * cm->mb_cols, sizeof(struct macroblock), cudaMemcpyHostToDevice);
  

  printf("cm value: %p\n", (void*)mb_gpu);

  int mb_x, mb_y;
  uint8_t *orig_Y, *ref_Y;
  cudaMalloc((void **)&orig_Y, sizeof(uint8_t)*cm->padw[Y_COMPONENT]*cm->padh[Y_COMPONENT]);
  cudaMalloc((void **)&ref_Y, sizeof(uint8_t)*cm->padw[Y_COMPONENT]*cm->padh[Y_COMPONENT]);
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));


  //cm->curframe->orig,
  cudaMemcpy(orig_Y, cm->curframe->orig->Y, sizeof(uint8_t)*cm->padw[Y_COMPONENT]*cm->padh[Y_COMPONENT], cudaMemcpyHostToDevice);
  cudaMemcpy(ref_Y, cm->curframe->recons->Y, sizeof(uint8_t)*cm->padw[Y_COMPONENT]*cm->padh[Y_COMPONENT], cudaMemcpyHostToDevice);
  
  dim3 threadsPerBlock(cm->me_search_range*2, cm->me_search_range*2);

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    { 
      me_block_8x8<<<1, threadsPerBlock, threadsPerBlock.x*threadsPerBlock.y*sizeof(int)>>>(cm_gpu, mb_gpu[Y_COMPONENT], mb_x, mb_y, orig_Y, ref_Y, Y_COMPONENT);

      printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    }
  }


  cudaDeviceSynchronize();

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      // me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->U,
      //     cm->refframe->recons->U, U_COMPONENT);
      // me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->V,
      //     cm->refframe->recons->V, V_COMPONENT);
    }
  }

  cudaFree(orig_Y);
  cudaFree(ref_Y);
  cudaFree(cm_gpu);
  cudaFree(mb_gpu);
  cudaFree(mb_gpu[Y_COMPONENT]);
  cudaFree(mb_gpu[U_COMPONENT]);
  cudaFree(mb_gpu[V_COMPONENT]);
}

/* Motion compensation for 8x8 block */
static void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *predicted, uint8_t *ref, int color_component)
{
  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  if (!mb->use_mv) { return; }

  int left = mb_x * 8;
  int top = mb_y * 8;
  int right = left + 8;
  int bottom = top + 8;

  int w = cm->padw[color_component];

  /* Copy block from ref mandated by MV */
  int x, y;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      predicted[y*w+x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
    }
  }
}

void c63_motion_compensate(struct c63_common *cm)
{
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y,
          cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U,
          cm->refframe->recons->U, U_COMPONENT);
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V,
          cm->refframe->recons->V, V_COMPONENT);
    }
  }
}
