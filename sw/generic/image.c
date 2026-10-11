#include "image.h"

#ifdef IMAGE_MODE_MMIO
void image_load(uint32_t index)
{
    *IMG_LOAD = index;
}

#else //IMAGE_MODE_MMIO
// Bytes of FP4 codes per image, two codes per byte
#if defined(DS_CIFAR10) || defined(DS_CIFAR100)
  #define IMG_BYTES 1536          // 3072 codes
#else
  #define IMG_BYTES  392          // 784 codes
#endif

extern const unsigned char image_data_raw_base[];
extern const unsigned char image_data_raw_end[];

uint8_t* image_stage_ptr;
uint8_t* image_label_ptr;
uint8_t* image_pred_ptr;
uint8_t* image_scale_ptr;

// file layout: [u32 n] then n*IMG_BYTES image bytes, then n labels, then n preds,
// then the layer 1 scales, one per block per image
#define IMAGES_OFFSET  4
#define LABELS_OFFSET  (IMAGES_OFFSET + N_SAMPLES * IMG_BYTES)
#define PREDS_OFFSET   (LABELS_OFFSET + N_SAMPLES)
#define SCALES_OFFSET  (PREDS_OFFSET  + N_SAMPLES)

// the scales are whatever is left in the file after the predictions
const uint32_t* image_num_scales(void)
{
    static uint32_t num_scales;
    uint32_t scale_bytes = (uint32_t)(image_data_raw_end - &image_data_raw_base[SCALES_OFFSET]);
    num_scales = scale_bytes / N_SAMPLES;
    return &num_scales;
}

void image_load(uint32_t index)
{
    image_stage_ptr = (uint8_t*)&image_data_raw_base[0] + IMAGES_OFFSET + index * IMG_BYTES;
    image_label_ptr = (uint8_t*)&image_data_raw_base[0] + LABELS_OFFSET + index;
    image_pred_ptr  = (uint8_t*)&image_data_raw_base[0] + PREDS_OFFSET  + index;
    image_scale_ptr = (uint8_t*)&image_data_raw_base[0] + SCALES_OFFSET + index * *image_num_scales();
}

__asm__( \
    ".section .imagedata\n" \
    ".global image_data_raw_base\n" \
    ".type image_data_raw_base, @object\n" 
    ".align 4\n" \
    "image_data_raw_base:\n" \
    ".incbin \"" IMAGE_BIN_FILE "\"\n" \
    ".global image_data_raw_end\n" \
    "image_data_raw_end:\n" \
    ".section .text\n"
);

#endif