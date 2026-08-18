ENTRY(_start)

MEMORY
{
  RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 1M
}

SECTIONS
{
  .text : {
    *(.text.init)
    *(.text*)
  } > RAM

  .rodata : {
    *(.rodata*)
    *(.imagedata)
  } > RAM

  .data : {
    *(.data*)
  } > RAM

  .bss (NOLOAD) : ALIGN(4) {
    *(.bss*)
    *(COMMON)
  } > RAM

  . = ALIGN(8);
  _stack_top = ORIGIN(RAM) + LENGTH(RAM);

}
