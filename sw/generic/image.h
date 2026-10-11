#ifndef IMAGE_H
#define IMAGE_H

#include <stdint.h>

// inference.mk passes these from its SIZE variable
#ifndef IMAGE_BIN_FILE
#define IMAGE_BIN_FILE "test_80.bin"
#endif

#ifndef N_SAMPLES
#define N_SAMPLES 80
#endif

#ifdef IMAGE_MODE_MMIO
#define IMG_LOAD  ((volatile unsigned int  *) 0xFFFF0010)
#define IMG_STAGE ((volatile unsigned char *) 0x80070000)
#define IMG_LABEL ((volatile unsigned int  *) 0xFFFF0014)
#define IMG_PRED  ((volatile unsigned int  *) 0xFFFF0018)

#define IMG_NUM_SCALES ((volatile unsigned int  *) 0xFFFF001C)
#define IMG_SCALES     ((volatile unsigned char *) 0x80071000)

#else //IMAGE_MODE_MMIO
extern uint8_t* image_stage_ptr;
extern uint8_t* image_label_ptr;
extern uint8_t* image_pred_ptr;
extern uint8_t* image_scale_ptr;
extern uint32_t image_count_var;

// Layer 1 input scales per image, worked out from the size of the .bin
const uint32_t* image_num_scales(void);

#define IMG_STAGE image_stage_ptr
#define IMG_LABEL image_label_ptr
#define IMG_PRED  image_pred_ptr
#define IMG_SCALES     image_scale_ptr
#define IMG_NUM_SCALES image_num_scales()

#endif //IMAGE_MODE_MMIO

void image_load(uint32_t index);

#endif
