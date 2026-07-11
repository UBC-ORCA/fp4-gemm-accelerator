ENTRY(_start)

MEMORY
{
  RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 512K
}

SECTIONS
{
  .text : {
    *(.text.init)
    *(.text*)
  } > RAM

  .rodata : {
    *(.rodata*)
  } > RAM

  .data : {
    *(.data*)
  } > RAM

  .mat_a 0x00060000 (NOLOAD) : ALIGN(4) {
    KEEP(*(.mat_a))
  } > RAM

  .mat_bt 0x00061000 (NOLOAD) : ALIGN(4) {
    KEEP(*(.mat_bt))
  } > RAM

  .mat_c 0x00062000 (NOLOAD) : ALIGN(4) {
    KEEP(*(.mat_c))
  } > RAM

  .tmp_prod 0x00063000 (NOLOAD) : ALIGN(4) {
    KEEP(*(.tmp_prod))
  } > RAM

  .bss (NOLOAD) : ALIGN(4) {
    *(.bss*)
    *(COMMON)
  } > RAM

  . = ALIGN(8);
  _stack_top = ORIGIN(RAM) + LENGTH(RAM);

  ASSERT(SIZEOF(.mat_a)    <= 256, ".mat_a exceeds 8x8 words")
  ASSERT(SIZEOF(.mat_bt)   <= 256, ".mat_bt exceeds 8x8 words")
  ASSERT(SIZEOF(.mat_c)    <= 256, ".mat_c exceeds 8x8 words")
  ASSERT(SIZEOF(.tmp_prod) <=  32, ".tmp_prod exceeds 8 words")
}
