
import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image, ImageChops
import os
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import heapq
import json

# --- Configuration ---
KERNEL_SIZE = 16
BYTES_PER_KERNEL = 2
USE_MANCHESTER = True
BRIGHTNESS = 15
STEGANO_COEFFS = [[0, 2], [0, 1], [1, 0], [1, 1], [2, 1], [2, 0], [1, 2], [3, 0], [2, 2], [0, 3], [3, 1], [4, 0], [1, 3], [4, 1], [3, 2], [0, 4], [5, 0], [1, 4], [3, 3], [5, 1], [4, 2], [2, 4], [0, 5], [6, 0], [0, 6], [3, 4], [5, 2], [4, 3], [2, 5], [1, 5], [1, 6], [6, 1]]
STEGANO_RND_NUMBERS = [221,194,162,117,216,143,126,244,95,77,129,158,95,32,153,39,190,16,79,113,99,140,2,116,65,143,225,87,165,248,166,227,215,52,245,114,197,87,100,49,64,132,138,220,222,66,220,235,227,131,11,201,97,33,23,36,129,10,132,60,11,32,39,239,68,253,251,60,83,105,51,32,91,163,199,160,81,133,243,249,26,42,36,81,191,45,243,173,6,61,90,33,73,102,60,15,6,74,83,108,187,236,70,167,100,158,93,195,238,146,220,57,105,176,231,189,27,88,91,138,159,110,106,52,231,240,147,220,201,17,137,2,65,162,81,182,144,85,160,225,135,109,29,208,146,105,62,33,18,221,19,10,191,236,158,169,228,96,46,187,25,34,85,97,138,134,186,13,88,203,41,239,46,215,207,9,139,90,84,189,31,80,17,247,74,9,115,52,41,224,213,142,113,112,166,36,79,149,58,96,51,13,51,64,28,230,3,11,63,136,139,70,106,130,119,100,126,181,150,30,85,104,44,161,78,227,226,242,68,4,6,179,229,211,239,159,119,47,0,196,26,95,47,150,121,199,169,133,51,243,224,222,103,102,248]

# --- Hamming and Manchester ---

def parity4(b):
    b ^= b >> 1
    b ^= b >> 2
    b ^= b >> 4
    return b & 1

def hamming_encode_single(b):
    p1 = parity4(b & 0b1101)
    p2 = parity4(b & 0b1011)
    p3 = parity4(b & 0b0111)
    o47 = (p1 << 6) | (p2 << 5) | ((b & 0b1000) << 1) | (p3 << 3) | (b & 0b0111)
    p4 = bin(o47).count('1') % 2
    return (o47 << 1) | p4

HAMM_ENCODE_KEY = [hamming_encode_single(i) for i in range(16)]

def hamming48_encode(data, use_manchester):
    encoded = []
    manchester_mask = 0b10101010 if use_manchester else 0
    for byte in data:
        encoded.append(HAMM_ENCODE_KEY[byte & 0b1111] ^ manchester_mask)
        encoded.append(HAMM_ENCODE_KEY[byte >> 4] ^ manchester_mask)
    return encoded

def hamming48_decode_byte(byte, use_manchester):
    if use_manchester:
        byte ^= 0b10101010
    z1 = bin(byte & 0b10101010).count('1') % 2
    z2 = bin(byte & 0b01100110).count('1') % 2
    z3 = bin(byte & 0b00011110).count('1') % 2
    err47 = z1 | (z2 << 1) | (z3 << 2)
    parity = bin(byte >> 1).count('1') % 2

    stat = 0
    if parity == (byte & 1):
        if err47 != 0:
            stat = 2
    else:
        stat = 1
        if err47 != 0:
            byte ^= 1 << (8 - err47)

    byte >>= 1
    return (byte & 0b111) | ((byte >> 1) & 0b1000), stat

def hamming48_decode(data, use_manchester):
    decoded = []
    errors = 0
    for i in range(0, len(data), 2):
        a, err1 = hamming48_decode_byte(data[i], use_manchester)
        b, err2 = hamming48_decode_byte(data[i+1], use_manchester)
        decoded.append(a | (b << 4))
        errors += err1 + err2
    return decoded, errors

# --- DCT Basis Generation ---

def idct2(block):
    return idct(idct(block.T).T)

def generate_dct_basis():
    if os.path.exists("dct0.png") and os.path.exists("dct1.png"):
        return
    print("Generating DCT basis images...")
    basis = np.zeros((KERNEL_SIZE * KERNEL_SIZE, KERNEL_SIZE * KERNEL_SIZE))
    anti_basis = np.zeros((KERNEL_SIZE * KERNEL_SIZE, KERNEL_SIZE * KERNEL_SIZE))

    for x in range(KERNEL_SIZE):
        for y in range(KERNEL_SIZE):
            coeffs = np.zeros((KERNEL_SIZE, KERNEL_SIZE))
            coeffs[x, y] = 1
            idct_res = idct2(coeffs)
            basis[y*KERNEL_SIZE:(y+1)*KERNEL_SIZE, x*KERNEL_SIZE:(x+1)*KERNEL_SIZE] = (idct_res / np.max(np.abs(idct_res))) * BRIGHTNESS
            coeffs[x, y] = -1
            idct_res = idct2(coeffs)
            anti_basis[y*KERNEL_SIZE:(y+1)*KERNEL_SIZE, x*KERNEL_SIZE:(x+1)*KERNEL_SIZE] = (idct_res / np.max(np.abs(idct_res))) * BRIGHTNESS

    Image.fromarray(np.uint8(basis + 128), "L").save("dct0.png")
    Image.fromarray(np.uint8(anti_basis + 128), "L").save("dct1.png")

# --- Encoding ---

def stegano_fill_random(data, n_bytes):
    if len(data) >= n_bytes:
        return data
    return data + STEGANO_RND_NUMBERS[:n_bytes - len(data)]

def bytes_to_dct(w, h, data, dct_basis0, dct_basis1):
    stegano_image = Image.new('L', (w * KERNEL_SIZE, h * KERNEL_SIZE))
    
    bid = 0
    for j in range(h):
        for i in range(w):
            block_img = Image.new('L', (KERNEL_SIZE, KERNEL_SIZE))
            
            byte1 = data[bid]
            byte2 = data[bid+1]
            
            bits = []
            for bit_idx in range(8):
                bits.append((byte1 >> (7-bit_idx)) & 1)
            for bit_idx in range(8):
                bits.append((byte2 >> (7-bit_idx)) & 1)

            for k, bit_val in enumerate(bits):
                st = STEGANO_COEFFS[k]
                src_x, src_y = st[0] * KERNEL_SIZE, st[1] * KERNEL_SIZE
                
                basis_img = dct_basis0 if bit_val else dct_basis1
                
                tile = basis_img.crop((src_x, src_y, src_x + KERNEL_SIZE, src_y + KERNEL_SIZE))
                
                block_img = ImageChops.add(block_img, tile)
            
            stegano_image.paste(block_img, (i * KERNEL_SIZE, j * KERNEL_SIZE))
            bid += 2
    return stegano_image

def encode(input_image_path, output_image_path, data, start_x, start_y):
    generate_dct_basis()
    dct_basis0 = Image.open("dct0.png")
    dct_basis1 = Image.open("dct1.png")
    
    w = -(-len(data) // BYTES_PER_KERNEL) # Ceiling division
    h = 1
    
    # Hamming encode
    encoded_data = hamming48_encode(data, USE_MANCHESTER)
    padded_data = stegano_fill_random(encoded_data, w * h * 2)
    
    stegano_block = bytes_to_dct(w, h, padded_data, dct_basis0, dct_basis1)
    
    original_image = Image.open(input_image_path).convert('RGB')
    
    crop_box = (start_x, start_y, start_x + w * KERNEL_SIZE, start_y + h * KERNEL_SIZE)
    target_region = original_image.crop(crop_box)
    
    target_ycbcr = target_region.convert('YCbCr')
    y, cb, cr = target_ycbcr.split()
    
    mean_luma_img = Image.new('L', y.size, int(np.mean(np.array(y))))
    
    alpha = 0.3
    blended_luma = Image.blend(mean_luma_img, stegano_block, alpha)
    
    final_region = Image.merge('YCbCr', (blended_luma, cb, cr)).convert('RGB')
    
    encoded_image = original_image.copy()
    encoded_image.paste(final_region, crop_box)
    encoded_image.save(output_image_path)
    print(f"Encoded image saved to {output_image_path}")
    return np.array(original_image), np.array(encoded_image), w, h, len(data)

# --- Decoding ---

def dct2_decode(block):
    return dct(dct(block.T).T)

def kernel_to_bytes(kernel):
    res = dct2_decode(kernel.astype(np.float32) - 128)
    bits = []
    for i in range(BYTES_PER_KERNEL * 8):
        bits.append(1 if res[STEGANO_COEFFS[i][1], STEGANO_COEFFS[i][0]] > 0 else 0)
    
    data = []
    for b in range(BYTES_PER_KERNEL):
        byte = 0
        for i in range(8):
            byte |= bits[i + b*8] << (7-i)
        data.append(byte)
    return data

def decode(encoded_image_path, w, h, start_x, start_y, original_len):
    img = np.asarray(Image.open(encoded_image_path).convert('L'))
    img = img[start_y:start_y+KERNEL_SIZE*h, start_x:start_x+KERNEL_SIZE*w]
    
    decoded_data = []
    for x in range(w):
        for y in range(h):
            kernel = img[y*KERNEL_SIZE:(y+1)*KERNEL_SIZE, x*KERNEL_SIZE:(x+1)*KERNEL_SIZE]
            decoded_data.extend(kernel_to_bytes(kernel))
            
    # Hamming decode
    hamming_decoded, errors = hamming48_decode(decoded_data, USE_MANCHESTER)
    
    return hamming_decoded[:original_len], errors

# --- Main ---

if __name__ == "__main__":
    # --- Encoding ---
    original_data = [ord(c) for c in "hello world"]
    input_image = "testOriginal.png"
    encoded_image_path = "encoded_test_python.png"
    start_x = 16*30
    start_y = 16*80
    
    print(f"Original data: {original_data}")
    
    original_img_arr, encoded_img_arr, w, h, original_len = encode(input_image, encoded_image_path, original_data, start_x, start_y)
    
    # --- Decoding ---
    decoded_data, errors = decode(encoded_image_path, w, h, start_x, start_y, original_len)
    
    print(f"Decoded data: {decoded_data}")
    print(f"Errors corrected: {errors}")
    
    # --- Image Quality Metrics ---
    psnr_value = psnr(original_img_arr, encoded_img_arr)
    ssim_value = ssim(original_img_arr, encoded_img_arr, multichannel=True, channel_axis=2, data_range=255)
    
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")

