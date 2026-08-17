#ifndef IMAGE_H
#define IMAGE_H

#include <stdint.h>

// #define IMAGE_MODE_MMIO
#define IMAGE_BIN_FILE "test_80.bin"
#define N_SAMPLES 80 

#ifdef IMAGE_MODE_MMIO
#define IMG_LOAD  ((volatile unsigned int  *) 0xFFFF0010)
#define IMG_STAGE ((volatile unsigned char *) 0x80070000)
#define IMG_LABEL ((volatile unsigned int  *) 0xFFFF0014)
#define IMG_PRED  ((volatile unsigned int  *) 0xFFFF0018)

#else //IMAGE_MODE_MMIO
extern uint8_t* image_stage_ptr;
extern uint8_t* image_label_ptr;
extern uint8_t* image_pred_ptr;
extern uint32_t image_count_var;

#define IMG_STAGE image_stage_ptr
#define IMG_LABEL image_label_ptr
#define IMG_PRED  image_pred_ptr

#endif //IMAGE_MODE_MMIO

void image_load(uint32_t index);

#endif
