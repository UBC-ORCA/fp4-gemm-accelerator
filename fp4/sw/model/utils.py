import torch
import torch.nn as nn
import numpy as np
import mptorch.quant as qpt
from mptorch import FloatingPoint
from mptorch.quant import compute_bias, scale as fp_scale, unscale


# FP4 values post tensor scaling
# bit layout [S E1 E0 M], bias_offset = 1 for exp=2
VAL_TO_BITS = {
     0.0:  0b0000,
     0.25: 0b0001,
     0.5:  0b0010,
     0.75: 0b0011,
     1.0:  0b0100,
     1.5:  0b0101,
    -0.25: 0b1001,
    -0.5:  0b1010,
    -0.75: 0b1011,
    -1.0:  0b1100,
    -1.5:  0b1101,
}

# per-layer weight scaling exponents
WEIGHT_BIASES    = [3, 2, 1]
LAYER_NAMES_INIT = ["w1_init", "w2_init", "w3_init"]
LAYER_NAMES_FP4  = ["w1_fp4",  "w2_fp4",  "w3_fp4"]


# default FP4 E2M1 quantizer
def quant_w(x):
    return qpt.float_quantize(
        x, exp=2, man=1, rounding="nearest", saturate=False, subnormals=True
    )


def encode_nibble(val):
    return VAL_TO_BITS[float(val)]


def pack_nibbles(values):
    # encode each value as a 4-bit nibble
    nibbles = []
    for v in values:
        nibbles.append(encode_nibble(v))
    # pad odd length
    if len(nibbles) % 2 != 0:
        nibbles.append(0)

    # combine pairs into bytes, high nibble first
    packed = []
    for i in range(0, len(nibbles), 2):
        packed.append((nibbles[i] << 4) | nibbles[i + 1])
    return packed


# collect Linear/QLinear submodules
def get_linear_layers(model):
    layers = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, qpt.QLinear)):
            layers.append(module)
    return layers


# FP4-quantize a layer at the given bias, bias folded as last column
def quantize_layer_scaled(layer, bias_val):
    # scale up by 2^bias, quantize, leave scaled
    bias_t  = torch.tensor([[float(bias_val)]], device=layer.weight.device)
    w_quant = quant_w(fp_scale(layer.weight.detach(), bias_t))
    b_quant = quant_w(fp_scale(layer.bias.detach().unsqueeze(0), bias_t)).squeeze(0)
    # fold bias as last column
    return torch.cat([w_quant, b_quant.unsqueeze(1)], dim=1).cpu().numpy()


# float32 export, post-quantize post-unscale
def export_weights_header(model, path="weights.h"):
    with open(path, "w") as f:
        # file header
        f.write("#pragma once\n\n")
        # one C array per layer
        for layer, name, bias_val in zip(get_linear_layers(model), LAYER_NAMES_INIT, WEIGHT_BIASES):
            # quantize then unscale to true float values
            wb = quantize_layer_scaled(layer, bias_val) * (2.0 ** -bias_val)
            wb = wb.astype(np.float32)
            out_dim, cols = wb.shape

            # array declaration
            f.write(f"static const float {name}[{out_dim}][{cols}] = {{\n")
            # one row per line
            for row in wb:
                cells = []
                for v in row:
                    cells.append(f"{v:.8f}f")
                f.write("    {" + ", ".join(cells) + "},\n")
            f.write("};\n\n")
    print(f"Weights header exported to {path}")


# FP4 nibbles packed two per uint8_t, row-major with bias folded
def export_weights_header_fp4_packed(model, path="weights_encoded.h"):
    with open(path, "w") as f:
        # file header
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")
        f.write("// FP4 E2M1 weights, codebook: {0, +/-0.25, 0.5, 0.75, 1.0, 1.5}\n")
        f.write("// effective weight = nibble_value * 2^(-LAYERx_BIAS)\n\n")
        # per-layer scale exponents
        f.write("#define LAYER1_BIAS  3\n")
        f.write("#define LAYER2_BIAS  2\n")
        f.write("#define LAYER3_BIAS  1\n\n")

        # one C array per layer
        for layer, name, bias_val in zip(get_linear_layers(model), LAYER_NAMES_FP4, WEIGHT_BIASES):
            wb = quantize_layer_scaled(layer, bias_val)
            out_dim, cols = wb.shape
            # encode + pack into bytes
            packed = pack_nibbles(wb.flatten())

            # array declaration
            f.write(f"// {name}: {out_dim} rows x {cols} cols\n")
            f.write(f"#define {name.upper()}_ROWS  {out_dim}\n")
            f.write(f"#define {name.upper()}_COLS  {cols}\n")
            f.write(f"static const uint8_t {name}[{len(packed)}] = {{\n    ")
            # 16 bytes per line
            for i, byte in enumerate(packed):
                f.write(f"0x{byte:02X}")
                if i < len(packed) - 1:
                    f.write(", ")
                if (i + 1) % 16 == 0 and i < len(packed) - 1:
                    f.write("\n    ")
            f.write("\n};\n\n")
    print(f"FP4 packed weights exported to {path}")


# FP4 nibbles, one per int16_t slot (no packing), row-major with bias folded
def export_weights_header_fp4_int16(model, path="weights_int16.h"):
    with open(path, "w") as f:
        # file header
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")
        f.write("// FP4 E2M1 weights, codebook: {0, +/-0.25, 0.5, 0.75, 1.0, 1.5}\n")
        f.write("// one nibble (0..15) per int16_t slot, no packing\n")
        f.write("// effective weight = nibble_to_int[nibble] * (64 >> LAYERx_BIAS) in Q8.8\n\n")
        # per-layer scale exponents
        f.write("#define LAYER1_BIAS  3\n")
        f.write("#define LAYER2_BIAS  2\n")
        f.write("#define LAYER3_BIAS  1\n\n")

        # one C array per layer
        for layer, name, bias_val in zip(get_linear_layers(model), LAYER_NAMES_FP4, WEIGHT_BIASES):
            wb = quantize_layer_scaled(layer, bias_val)
            out_dim, cols = wb.shape
            # encode each value as a 4-bit nibble
            nibbles = []
            for v in wb.flatten():
                nibbles.append(encode_nibble(v))

            # array declaration
            f.write(f"// {name}: {out_dim} rows x {cols} cols (one nibble per int16_t)\n")
            f.write(f"#define {name.upper()}_ROWS  {out_dim}\n")
            f.write(f"#define {name.upper()}_COLS  {cols}\n")
            f.write(f"static const int16_t {name}[{out_dim} * {cols}] = {{\n    ")
            # 32 nibbles per line
            for i, n in enumerate(nibbles):
                f.write(str(n))
                if i < len(nibbles) - 1:
                    f.write(", ")
                if (i + 1) % 32 == 0 and i < len(nibbles) - 1:
                    f.write("\n    ")
            f.write("\n};\n\n")
    print(f"FP4 int16-per-nibble weights exported to {path}")


# Q(wl-fl).fl integer export, fixed-point training path
def export_weights_header_fixed(model, path="weights.h", wl=16, fl=8):
    # fixed-point format parameters
    scale_factor = 1 << fl
    int_min, int_max = -(1 << (wl - 1)), (1 << (wl - 1)) - 1
    ctype = "int16_t" if wl <= 16 else "int32_t"

    with open(path, "w") as f:
        # file header
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define WEIGHT_WL {wl}\n")
        f.write(f"#define WEIGHT_FL {fl}\n")
        f.write(f"#define WEIGHT_SCALE {scale_factor}\n\n")

        # one C array per layer
        for layer, name in zip(get_linear_layers(model), LAYER_NAMES_INIT):
            # quantize then fold bias
            w_q = quant_w(layer.weight.detach()).cpu().numpy()
            b_q = quant_w(layer.bias.detach()).cpu().numpy()
            wb  = np.concatenate([w_q, b_q[:, None]], axis=1)
            # scale to integers, clip to range
            wb_int = np.clip(np.round(wb * scale_factor), int_min, int_max).astype(np.int32)

            out_dim, cols = wb_int.shape
            # array declaration
            f.write(f"static const {ctype} {name}[{out_dim}][{cols}] = {{\n")
            # one row per line
            for row in wb_int:
                cells = []
                for v in row:
                    cells.append(str(v))
                f.write("    {" + ", ".join(cells) + "},\n")
            f.write("};\n\n")
    print(f"Weights header exported to {path} as Q{wl - fl}.{fl} {ctype}")


def export_test_data_header(dataset, n_samples=10, path="test_data.h"):
    # pick random samples
    indices = np.random.choice(len(dataset), size=n_samples, replace=False)
    images, labels = [], []
    for i in indices:
        img, label = dataset[i]
        # tensor [1,28,28] in [0,1] -> uint8 [0,255]
        pixels = (img.numpy().flatten() * 255).clip(0, 255).astype(np.uint8)
        images.append(pixels)
        labels.append(label)

    with open(path, "w") as f:
        # file header
        f.write("#pragma once\n")
        f.write(f"#define N_SAMPLES {n_samples}\n\n")

        # one image per row
        f.write(f"static const unsigned char test_images[{n_samples}][784] = {{\n")
        for pixels in images:
            cells = []
            for p in pixels:
                cells.append(str(p))
            f.write("    {" + ", ".join(cells) + "},\n")
        f.write("};\n\n")

        # ground truth labels
        f.write(f"static const unsigned char test_labels[{n_samples}] = {{\n")
        label_cells = []
        for label in labels:
            label_cells.append(str(label))
        f.write("    " + ", ".join(label_cells) + "\n")
        f.write("};\n")
    print(f"Test data header exported to {path}")


# diagnostic weight/activation ranges
def print_ranges(model, test_loader, device):
    fmt = FloatingPoint(exp=2, man=1, subnormals=True, saturate=False)
    layer_list = get_linear_layers(model)

    print("\nMaster weight ranges:")
    for i, layer in enumerate(layer_list):
        w = layer.weight.detach()
        print(f"Layer {i+1}: min={w.min():.4f}  max={w.max():.4f}  mean_abs={w.abs().mean():.4f}")

    print("\nQuantized weight ranges in fwd pass:")
    for i, layer in enumerate(layer_list):
        w = layer.weight.detach()
        b = compute_bias(w, fmt)
        w_eff = unscale(quant_w(fp_scale(w, b)), b)
        unique_vals = w_eff.unique().cpu().numpy()
        nonzero_pct = (w_eff != 0).float().mean().item() * 100
        print(f"Layer {i+1}: unique values = {unique_vals}")
        print(f"         nonzero weights = {nonzero_pct:.1f}%")

    print("\nWeight bias per layer:")
    for i, layer in enumerate(layer_list):
        w = layer.weight.detach()
        b = compute_bias(w, fmt)
        print(f"Layer {i+1} weight bias: {b.reshape(-1)[0].item():.1f}")
