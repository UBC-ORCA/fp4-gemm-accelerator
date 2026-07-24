import argparse
from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / "fp4_mul_int9.templ"
PLACEHOLDER = "[[REPLACE fp4_lut]]"

e2m1_input_map = {
    .0: "000",
    .5: "001",
    1.0: "010",
    1.5: "011",
    2.0: "100",
    3.0: "101",
    4.0: "110",
    6.0: "111",
}


def gen_lut_lines():
    entries = list(e2m1_input_map.items())
    lines = []
    for i, (a, _) in enumerate(entries):
        for j, (b, _) in enumerate(entries):
            intValue = int(a / 0.5) * int(b / 0.5)
            is_last = (i == len(entries) - 1) and (j == len(entries) - 1)
            token = f"8'd{intValue}" + ("" if is_last else ",")
            comment = f"/* {a} * {b} * 4 = {float(intValue)} */"
            lines.append(f"    {token:<8}{comment}")
    return lines


def generate(out_path):
    template = TEMPLATE_PATH.read_text()
    if PLACEHOLDER not in template:
        raise ValueError(f"template {TEMPLATE_PATH} is missing placeholder {PLACEHOLDER!r}")
    rendered = template.replace(PLACEHOLDER, "\n".join(gen_lut_lines()))
    Path(out_path).write_text(rendered)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the fp4_mul_int9 Verilog module from its template, "
        "filling in the multiplication LUT."
    )
    parser.add_argument("file", help="output path for the generated Verilog file")
    args = parser.parse_args()
    generate(args.file)


if __name__ == "__main__":
    main()
