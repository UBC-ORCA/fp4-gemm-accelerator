#include "image.h"

#ifdef IMAGE_MODE_MMIO
void image_load(uint32_t index)
{
    *IMG_LOAD = index;
}

#else //IMAGE_MODE_MMIO
extern const unsigned char image_data_raw_base[];

uint8_t* image_stage_ptr;
uint8_t* image_label_ptr;
uint8_t* image_pred_ptr;

void image_load(uint32_t index)
{
    image_stage_ptr = (uint8_t*)&image_data_raw_base[0] + 4 + index*(28*28);
    image_label_ptr = (uint8_t*)&image_data_raw_base[0] + 4 + N_SAMPLES*(28*28) + index;
    image_pred_ptr =  (uint8_t*)&image_data_raw_base[0] + 4 + N_SAMPLES*(28*28) + N_SAMPLES + index;
}

__asm__( \
    ".section .imagedata\n" \
    ".global image_data_raw_base\n" \
    ".type image_data_raw_base, @object\n" 
    ".align 4\n" \
    "image_data_raw_base:\n" \
    ".incbin \"" IMAGE_BIN_FILE "\"\n" \
    ".section .text\n"
);

#endif

