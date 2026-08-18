#!/usr/bin/env python3
"""
Usage:
    python main.py "your string here"
or (if no argument):
    python main.py
then type/paste the string (Ctrl+D or Ctrl+Z to finish).
"""

import sys
import math
import hashlib
import numpy as np
from PIL import Image

# ============================================================

# Try to import scipy for fast convolution, fallback to pure numpy
try:
    from scipy.ndimage import convolve
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# ============================================================

def string_to_bits(s: str) -> list[int]:
    """Convert string to UTF-8 bytes and then to a list of bits (MSB first)."""
    data = s.encode('utf-8')
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def compute_parameters(bits: list[int]) -> dict:
    """Compute various statistical parameters from the bit sequence."""
    N = len(bits)
    ones = sum(bits)
    zeros = N - ones
    transitions = 0
    run_lengths = []
    if N:
        current = bits[0]
        count = 1
        for b in bits[1:]:
            if b == current:
                count += 1
            else:
                run_lengths.append(count)
                transitions += 1
                current = b
                count = 1
        run_lengths.append(count)
    avg_run = sum(run_lengths) / len(run_lengths) if run_lengths else 0

    return {
        'N': N,
        'ones': ones,
        'zeros': zeros,
        'transitions': transitions,
        'avg_run': avg_run,
        'freq_ones': ones / N if N else 0,
        'freq_zeros': zeros / N if N else 0,
    }

def build_bit_matrix(bits: list[int]) -> np.ndarray:
    """Arrange bits into a near‑square matrix, padding with zeros."""
    N = len(bits)
    if N == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    width = int(math.ceil(math.sqrt(N)))
    height = int(math.ceil(N / width))
    total = width * height

    padded = bits + [0] * (total - N)
    return np.array(padded, dtype=np.uint8).reshape(height, width)

# ============================================================

def generate_kernels(bits: list[int], num: int = 3, size: int = 3) -> list[np.ndarray]:
    """
    Derive convolution kernels from the bit sequence.
    Each kernel is a `size x size` matrix with values in {-1, 1}.
    """
    kernels = []
    needed = num * size * size
    # If not enough bits, repeat the sequence
    extended = bits * (needed // len(bits) + 1)
    extended = extended[:needed]

    for k in range(num):
        start = k * size * size
        chunk = extended[start:start + size * size]
        arr = np.array(chunk, dtype=np.float32).reshape(size, size)
        # map 0 -> -1, 1 -> 1
        arr = 2 * arr - 1
        kernels.append(arr)
    return kernels

def convolve2d(matrix: np.ndarray, kernel: np.ndarray, mode='same') -> np.ndarray:
    """
    2D convolution with handling of borders (zero padding).
    Uses scipy if available, otherwise a slower pure‑numpy implementation.
    """
    if HAVE_SCIPY:
        return convolve(matrix.astype(np.float32), kernel, mode=mode)
    else:
        # Fallback: manual convolution using padding and sliding windows
        k_h, k_w = kernel.shape
        pad_h = k_h // 2
        pad_w = k_w // 2
        padded = np.pad(matrix, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        out = np.zeros_like(matrix, dtype=np.float32)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                window = padded[i:i+k_h, j:j+k_w]
                out[i, j] = np.sum(window * kernel)
        return out

# ============================================================

def normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0,1] range."""
    arr = arr.astype(np.float32)
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val < 1e-12:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

def generate_coefficients(s: str, count: int) -> list[float]:
    """Deterministically produce coefficients in [0,1] from the string hash."""
    digest = hashlib.sha256(s.encode('utf-8')).digest()
    # convert digest bytes to floats
    coeffs = []
    for i in range(count):
        # take 4 bytes as an integer, map to [0,1]
        idx = (i * 4) % len(digest)
        val = int.from_bytes(digest[idx:idx+4], 'big') / (2**32)
        coeffs.append(val)
    return coeffs

# ============================================================

def string_to_image(s: str) -> Image.Image:
    """Main pipeline: string -> bits -> matrix -> transformations -> RGB image."""
    bits = string_to_bits(s)
    if not bits:
        # empty string -> 1x1 black image
        return Image.new('RGB', (1, 1), (0, 0, 0))

    # 1. Build initial bit matrix
    M0 = build_bit_matrix(bits)
    H, W = M0.shape
    params = compute_parameters(bits)

    # 2. Generate kernels and convolve
    kernels = generate_kernels(bits, num=3)
    M1 = convolve2d(M0, kernels[0])
    M2 = convolve2d(M0, kernels[1])
    M3 = convolve2d(M0, kernels[2])

    # 3. Gradients (horizontal and vertical)
    dx = np.abs(np.roll(M0, -1, axis=1) - M0)
    dy = np.abs(np.roll(M0, -1, axis=0) - M0)
    grad = np.sqrt(dx.astype(np.float32)**2 + dy.astype(np.float32)**2)

    # 4. XOR with shifted versions
    xor_h = np.bitwise_xor(M0, np.roll(M0, -1, axis=1)).astype(np.float32)
    xor_v = np.bitwise_xor(M0, np.roll(M0, -1, axis=0)).astype(np.float32)

    # 5. Wave patterns derived from string parameters
    i_idx = np.arange(H)[:, None]
    j_idx = np.arange(W)[None, :]
    freq_i = 2 * np.pi * (params['freq_ones'] * 5 + 1)
    freq_j = 2 * np.pi * (params['freq_zeros'] * 5 + 1)
    phase = params['transitions'] / max(1, params['N']) * 2 * np.pi
    wave = np.sin(freq_i * i_idx / H + freq_j * j_idx / W + phase)

    # distance from center for radial wave
    center_i = H / 2
    center_j = W / 2
    dist = np.sqrt((i_idx - center_i)**2 + (j_idx - center_j)**2)
    radial_wave = np.sin(2 * np.pi * dist / max(H, W) * 3 + phase)

    # 6. Normalize all feature maps to [0,1]
    M1n = normalize(M1)
    M2n = normalize(M2)
    M3n = normalize(M3)
    gradn = normalize(grad)
    xor_hn = normalize(xor_h)
    xor_vn = normalize(xor_v)
    wave_norm = normalize(wave)
    radial_norm = normalize(radial_wave)

    # 7. Coefficients for combining channels (12 coefficients)
    coeffs = generate_coefficients(s, 12)

    # 8. Build RGB channels as linear combinations
    R = (coeffs[0] * M1n + coeffs[1] * wave_norm + coeffs[2] * gradn + coeffs[3] * radial_norm) * 255
    G = (coeffs[4] * M2n + coeffs[5] * wave_norm + coeffs[6] * gradn + coeffs[7] * xor_hn) * 255
    B = (coeffs[8] * M3n + coeffs[9] * wave_norm + coeffs[10] * gradn + coeffs[11] * xor_vn) * 255

    # 9. Clip and convert to uint8
    img_arr = np.stack([R, G, B], axis=2)
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)

    return Image.fromarray(img_arr, mode='RGB')

# ============================================================

def main():
    if len(sys.argv) > 1:
        s = ' '.join(sys.argv[1:])
    else:
        print("Enter string (Ctrl+D or Ctrl+Z to finish):", file=sys.stderr)
        try:
            s = sys.stdin.read()
        except KeyboardInterrupt:
            sys.exit(1)

    if not s.strip():
        print("Empty input, generating default image.")
        s = "default"

    img = string_to_image(s)
    output_filename = "output.png"
    img.save(output_filename)
    print(f"Image saved as {output_filename}")

    # Optionally show the image (if in interactive environment)
    try:
        img.show()
    except Exception:
        pass

# ============================================================

if __name__ == "__main__":
    main()
