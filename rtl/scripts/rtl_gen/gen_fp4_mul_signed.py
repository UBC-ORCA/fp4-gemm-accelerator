import argparse
from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / "fp4_mul_int9_signed.templ"
PLACEHOLDER = "[[REPLACE fp4_lut]]"

# 3-bit {exp, mant} magnitude code -> real magnitude, per the E2M1 format.
e2m1_mag_map = {
    "000": .0,
    "001": .5,
    "010": 1.0,
    "011": 1.5,
    "100": 2.0,
    "101": 3.0,
    "110": 4.0,
    "111": 6.0,
}

# Full 4-bit {sign, exp, mant} code -> real signed value.
e2m1_signed_map = {
    f"{sign}{magcode}": (-1.0 if sign == "1" else 1.0) * magval
    for sign in ("0", "1")
    for magcode, magval in e2m1_mag_map.items()
}


def gen_lut_lines():
    entries = list(e2m1_signed_map.items())
    lines = []
    for i, (_, a) in enumerate(entries):
        for j, (_, b) in enumerate(entries):
            intValue = round(a * b * 4)
            is_last = (i == len(entries) - 1) and (j == len(entries) - 1)
            literal = f"-9'sd{-intValue}" if intValue < 0 else f"9'sd{intValue}"
            token = literal + ("" if is_last else ",")
            comment = f"/* {a} * {b} * 4 = {float(intValue)} */"
            lines.append(f"    {token:<10}{comment}")
    return lines


def generate(out_path):
    template = TEMPLATE_PATH.read_text()
    if PLACEHOLDER not in template:
        raise ValueError(f"template {TEMPLATE_PATH} is missing placeholder {PLACEHOLDER!r}")
    rendered = template.replace(PLACEHOLDER, "\n".join(gen_lut_lines()))
    Path(out_path).write_text(rendered)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the fp4_mul_int9_signed Verilog module from its template, "
        "filling in the signed multiplication LUT."
    )
    parser.add_argument("file", help="output path for the generated Verilog file")
    args = parser.parse_args()
    generate(args.file)


if __name__ == "__main__":
    main()
