---
title: "Caching ASTC Parameters to Disk"
date: 2025-11-18
---

## Intro

Here's a negative result around compressing and caching ASTC parameters to reduce transfer overhead.

Taking an Adreno 650 as our reference:

1. _GPU_ - 1.2 TFlops (1200 GFlops)
2. _CPU SIMD int8_ - 18 GFlops
3. _CPU single core_ - up to 2.2 GFlops
4. _DRAM_ bandwidth - 44 GB/s
5. _DRAM -> GPU_ load bandwidth - 44 GB/s (Adreno/mobile GPUs are all integrated with direct DRAM access)
6. _Disk -> DRAM_ load bandwidth - 2 GB/s

This creates a set of "frontiers" for what aspect (compute, memory, disk) of a purely optimal compute kernel is the bottleneck:

1. A kernel is (GPU) compute bound _if_ it reaches more than **~28 FLOPs per byte of memory read**, _and_ it reaches more than **600 FLOPs per byte of disk read or written**
2. A kernel is memory bound _if_ it cannot achieve ~28 FLOPs per byte of memory read, _and_ it reaches more than **22 bytes of memory load for every byte of disk read**
3. A kernel is disk bound _if_ it cannot achieve 600 FLOPs per byte of disk read _and_ it cannot achieve 22 bytes of memory load for every byte of disk read

Notice the scaling here:

1. **GPU** O(1200 GFLOPs) 
2.  (~30x)>>> **DRAM** O(20 GB/s) ~ **SIMD int8** O(20 GFLOPs)  
3.  (~20x)>>> **Disk** O(2 GB/s) ~ **CPU single core** O(2 GFLOPS). 

This inherent unbalanced hierarchy is what makes disk caching vs just a raw compute shader such a hard tradeoff, because in order for it to be worth it, you need to either:

1. Heavily amplify the size of the cache so that you're effectively achieving 1:22 compression rates to avoid the disk->memory transfer being the bottleneck, OR
2. Have a shader with such high arithmetic intensity that it needs to perform 600 FLOPs for each byte read.

Before tackling #1, I want to disprove #2. For a simple astc transcoder, we work on 16 bits per pixel (per thread) from non-BC1 textures. This gives us a budget of 1.6 KFlops per thread. Our transcoder is very simple, and often finishes in less than $\frac{1}{10}$ of that budget.

The transcoder thread will read (on average) a single pixel (16 bits encoded in BC), and write a single pixel in astc (16 bits encoded in astc). This gives us a budget of ~120 flops before we're compute bound. As it stands, we're only slightly compute bound today (thanks largely to the complexity of astc block packing).

Next, I will show you how it's impossible to get 1:22 compression (or even close to 1:2) to make disk caching worth it.

NOTE: this math is different for the CPU SIMD units, which is only about $\frac{1}{64}$ the parallelization of the iGPU (with the same Disk and DRAM speed). As a result, if your **SIMD-based vector transcoder takes more than 25 flops per pixel, you are already CPU-bound**, so a cache is definitely worth it. On a serialized design, this is even worse (assuming ~2.2 GHz CPU), you are CPU-bound if your transcoder takes more than 1 flop per pixel, and you better find ways to trade compute for disk (precalculations) as much as you can.

---

## Cache Design

I want to create a file format for caching the parameters of predetermined astc images (where every block is on a uniform block mode, described below). This format should ideally reduce the amount of file IO and memory transfer, hence we want to compress the object while keeping the code reasonably memory-bound (not too high of arithmetic intensity for your typical ARM/Adreno iGPU)

1. Only support 4x4 rgb_ldr and rgba_ldr (single plane) for full color ep range, but weights up to 4 bits
2. Is not a direct astc format, but contains all of the necessary parameters to pack into the equivalent .astc object

Parameters:

$$
\begin{matrix}
r_0 & g_0 & b_0 & a_0 & \in \mathbb{B}^8 \\
r_1 & g_1 & b_1 & a_1 & \in \mathbb{B}^8 \\
w_0 & w_1 & w_2 & w_3 & \in \mathbb{B}^4 \\
w_4 & w_5 & w_6 & w_7 & \in \mathbb{B}^4 \\
w_8 & w_9 & w_{10} & w_{11} & \in \mathbb{B}^4 \\
w_{12} & w_{13} & w_{14} & w_{15} & \in \mathbb{B}^4 \\
\end{matrix}
$$

Natively, this block can be packed into 128 bits (16 bytes)

BC1 can be transcoded into a simpler format:

$$
\begin{matrix}
r_0 & g_0 & b_0 & ~ & \in \mathbb{B}^{(5 \times 6 \times 5)} \\
r_1 & g_1 & b_1 & ~ & \in \mathbb{B}^{(5 \times 6 \times 5)} \\
w_0 & w_1 & w_2 & w_3 & \in \mathbb{B}^2 \\
w_4 & w_5 & w_6 & w_7 & \in \mathbb{B}^2 \\
w_8 & w_9 & w_{10} & w_{11} & \in \mathbb{B}^2 \\
w_{12} & w_{13} & w_{14} & w_{15} & \in \mathbb{B}^2 \\
\end{matrix}
$$

Natively, this block can be packed into 64 bits (8 bytes), same as, well, BC1.

In general, can we do better than this?

### Option 1: ASTC reduced format

We can pack this as the final astc block, but just removed the first 17 bits of redundant information. This will take us down to 111 bits per block, which is a marginal improvement, at the cost of a significantly increased packing complexity.

We're not going to consider this option at the moment.

### Option 2: Variable sized format

Let's break the components up into $(r, g, b, a, w)$, where within each block, you have 2 components each of rgba, and 16 of w. They _should_ exhibit some locality both within (in the case of the weights) and across (in the case of the colors) blocks.

We can think of a set of blocks that are encoded within a single shared subgroup (say 64 blocks, or 256 pixels) as a walk on each of the components.

Let's start with a sequence of the red color endpoint components:

$$
\begin{pmatrix}
r_{0,1}, r_{0,2}, \cdots, r_{0,64} \\
r_{1,1}, r_{1,2}, \cdots, r_{1,64} \\
\end{pmatrix}
$$

now, we expect some locality here such that $r_{0,x} \sim r_{0,y}$ if $x \sim y$ are close together. This means that with high probability:

$$
|r_{0,x+1} - r_{0,x}| < \delta
$$

let's say that $\delta \in \{-15..15, \infty\} \equiv \mathbb{B}^5$ (31 integer values and an $\infty$), then we can delta-code the differences between the parts of a "walk" of $r_0$ within the radius of $\delta$ in the following fashion

$$
\mathbf{enc}(x_{n+1}, x_n, s = \{x_0, \delta_1, \cdots, \delta_n\}) = s \cup \begin{cases}
\{\delta_{n+1}\} & \text{if} |x_{n+1} - x_n| = \delta_{n+1} < \infty \\
\{\infty, x_n\}
\end{cases}
$$

given this, a typical walk over the component $x_n$ will typically look something like:

$$
\{\infty, x_0, \delta_1, \delta_2, \cdots, \infty, x_n, \delta_{n+1}, \cdots\}
$$

as long as the frequency of the $\infty$ symbol (which means that we need to explicitly specify the next component, a full byte) is low, then we can either huffman or range code this sequence with hopefully good performance.

Note that for RGBA (64 uint8s), a $\delta_\infty = 16$ should be good. For weights (256 uint4s), an absolute coding of the values should suffice.

For simplicity, we can choose a **huffman code** with a per-subgroup (64 blocks) shared dictionary/tree for:

1. The RGB components, since they share very similar structures
2. A distinct alpha channel, since it's likely that they will just have a single value for most subgroups
3. A distinct weight channel, since this is the most variable aspect of this whole scheme

#### Group start offsets

Since a group/chunk of blocks are no longer deterministically mapped to a fixed point, we need to also include a **central directory** (cd) lookup.

We'll go with the naive approach of just specifying $\frac{w \times h}{256}$ additional uint32s, to be optimized later.

#### Parallelization

The aim is to do this process in subgroups of 64 threads on an iGPU (ARM/Adreno). Each subgroup will compute its subgroup id, and then find its group start offset in the file (already mapped into memory). Then it will perform either the encoding (from astc/bc1 weights to astc) or decoding (from the .fastc group object into astc)

#### Empirical Results

```
Using device: cuda
Image processed into 129600 blocks.
Processing 100 groups of 64 blocks...

--- Compression Efficacy Report ---
Original Image Blocks: 6400
Original ASTC Size: 100.00 KB
-----------------------------------
Compressed Size Breakdown:
  - Huffman Data: 67.72 KB
  - Huffman Trees: 7.35 KB
  - Huffman RGB Data: 18.67 KB
  - Huffman RGB Trees: 4.65 KB
  - Huffman W Data: 49.06 KB
  - Huffman W Trees: 2.54 KB
  - Absolute Values: 3.54 KB
  - Central Directory: 0.39 KB
-----------------------------------
Total Compressed Size: 79.00 KB
Compression Ratio: 1.27:1
SAVING: 21.00%
RGB Codebook: [(0, '00'), (1, '011'), (-2, '1010'), (-1, '1111'), (2, '1011'), (3, '0100'), (4, '1001'), (16, '1101'), (-4, '11000'), (-3, '10001'), (6, '01011'), (-8, '110010'), (-6, '010100'), (5, '111011'), (7, '010101'), (10, '110011'), (-15, '1000000'), (-10, '1000001'), (-7, '1110011'), (-5, '1110100'), (8, '1000010'), (9, '1000011'), (11, '1110101'), (13, '1110000'), (-14, '11100010'), (-9, '11100100'), (14, '11100101'), (-11, '111000110'), (15, '111000111')]
RGB Freq: Counter({0: 70, 1: 44, -1: 37, 16: 30, 2: 26, -2: 25, 4: 24, 3: 19, -4: 13, -3: 12, 6: 11, 5: 8, 10: 7, -8: 7, 7: 5, -6: 5, 11: 4, -5: 4, -7: 4, 8: 3, -10: 3, -15: 3, 13: 3, 9: 3, -9: 2, -14: 2, 14: 2, 15: 1, -11: 1})
Alpha Codebook: {0: ''}
Alpha Freq: Counter({0: 126})
Weight Codebook: [(0, '0001'), (1, '0110'), (2, '0011'), (3, '0100'), (4, '0111'), (5, '1000'), (6, '0101'), (7, '1011'), (8, '1110'), (9, '1111'), (10, '1100'), (11, '1010'), (12, '1001'), (13, '1101'), (14, '0000'), (15, '0010')]
Weight Freq: Counter({9: 83, 8: 82, 13: 80, 10: 78, 7: 68, 11: 66, 12: 65, 5: 63, 4: 61, 1: 59, 6: 57, 3: 57, 2: 56, 15: 55, 0: 49, 14: 45})
Weight deltas: [2, -1, -1, -1, -3, -1, 0, -4, 0, 2, -2, -1, 0, 0, 0, 14, -2, -1, 0, -4, 2, 2, 0, -8, 2, 2, 0, -5, 0, -1, 0, 11, 0, -3, 2, -7, 3, 0, 3, -5, 0, 0, -2, -1, 1, 0, -2, 11, 2, -3, 0, -2, 0, 0, 0, -4, 0, 0, 0, -3, 0, 0, -3, 15, 0, 0, 0, 0, 0, 0, -5, 0, 0, -5, 0, 0, 0, 0, -5, 14, 0, 0, 0, -4, 0, 0, 0, -5, 0, 0, 0, -4, 0, 0, 0, 11, 0, 0, -2, 3, -3, 0, 0, -1, 0, 0, -4, -5, 2, 0, 0, 13, 0, 0, 0, -6, 0, 0, -6, -1, 0, 0, 0, 0, 3, 0, 0, 6, -1, -3, 0, 0, 0, -3, 2, -2, 0, 1, 4, -3, 0, 0, 0, 3, -3, 2, -3, 4, 0, 2, 0, -1, 0, 0, 0, -3, 0, 0, 0, 0, 0, -4, 0, 1, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, -4, -4, 0, 0, 6, -5, 0, 6, 0, -2, 0, 0, 0, 0, 0, 0, 0, -7, 0, 3, 7, -6, 0, 4, 3, -1, 0, 1, 0, -4, 0, -3, 1, 4, 0, 0, 3, -3, 3, 0, 0, -6, 6, 0, 0, -13, 1, 2, 0, 9, 0, -1, 1, -2, 0, 2, 1, -7, -2, 3, 2, -10, 1, 1, 3, 3, -1, -3, -3, 14, 0, -3, -4, 3, 4, 0, -5, -6, 4, 0, 0, -7, 2, 2, 2, -5, 1, 4, 3, -8, 0, 4, 4, -2, 0, 3, 4, -15, 3, 5, 0, -1, 2, 2, 3, -7, 0, 2, 4, 0, -4, 2, 3, -3, -1, 0, 1, -4, 1, 0, 0, -8, 2, 2, 2, 1, 2, 4, 0, -4, -4, 4, 4, -9, 0, 3, 7, -13, 3, 4, 3, -3, 0, 3, 0, -6, -2, 0, 3, -5, 5, 1, -5, 4, -2, -1, 6, -2, -3, -1, 12, 0, -2, -6, 1, -2, -4, 0, 6, -2, -6, 1, 3, 3, -7, 1, 3, 11, -11, -1, 9, 2, -13, -1, 9, -5, -2, 2, 4, -3, 2, 2, 0, 5, -1, 1, 1, -3, 0, -2, 0, -2, -1, -6, 0, 7, -3, -3, 6, 1, 4, -6, 2, 1, 4, -3, 2, 1, -4, 2, -1, 4, -14, 6, -7, 11, -4, 1, 1, 6, -2, 0, 0, -3, 3, -1, -1, -11, 12, 1, 1, -14, 1, 0, 0, 9, -1, -1, -3, 6, 0, 0, -1, 4, 0, 1, -1, -11, -2, -1, 0, 6, -2, -1, 0, 8, 0, 1, 0, 3, 0, 0, 0, -14, 0, 1, 1, 0, 0, 1, 1, 8, -1, -1, -1, 5, 0, -1, -1, -9, -1, -1, -2, 6, 0, 0, 0, 5, 1, 2, 0, 1, -1, -2, 0, -11, -1, 1, -1, 3, 0, 0, -1, 7, -1, 0, -1, 8, -1, -1, -2, -11, 1, 1, 0, 1, -1, 1, -1, 6, 0, -2, 0, 9, -3, -2, -2, -7, 0, 5, 4, -7, 0, 4, 4, -5, 0, 0, 5, -8, 0, 6, 2, -9, 0, 5, 0, -2, 0, 4, 0, -7, 5, 2, 4, -10, 0, 7, 0, -10, 3, 3, 0, -3, 0, 6, 3, -9, 6, 3, 0, -6, 6, 3, 0, -15, 0, 5, 4, -9, 0, 5, 5, -10, 5, 5, 5, -10, 5, 0, 5, -14, 9, 3, -4, -2, 0, -3, 0, 9, -7, -6, 3, 2, 2, 4, 1, -1, 0, 2, 0, 0, 2, -2, -5, 4, 1, -5, -6, 6, -5, -3, 2, 7, -1, -4, 2, 1, 0, -4, 0, -2, 5, -1, -1, -3, 11, 3, -4, -3, -1, -3, -4, 10, 5, -4, -8, 5, 5, -1, -5, 1, -1, 2, 0, 1, -1, -7, 1, 6, 4, -6, -7, 4, 11, -2, -10, 5, 0, -2, -2, 4, 0, 0, 3, -10, 6, 1, -1, -7, 9, 3, -2, -9, 8, 4, 1, 0, -1, 0, -1, 1, 0, 0, 0, -6, 1, 1, 4, -13, 1, 4, 6, -7, 0, 8, 0, -3, -4, 3, 4, -3, -4, -1, 4, 1, -4, -4, 3, 1, 0, 0, 0, 10, 0, 0, -5, 0, 0, 0, 0, -10, 0, 0, 0, 13, 0, 0, 0, -1, 0, -4, 0, 4, -4, 0, 0, -8, 0, 0, 0, 14, -2, -3, -2, 2, 0, -4, 0, 2, 5, -2, -7, 1, 6, -3, -6, 11, 0, 1, 1, -2, 1, 0, 0, 0, 2, -3, -5, 6, 2, -6, -9, 12, -3, 2, 2, -5, 5, -1, -11, 10, -9, -2, 6, 4, -3, 0, 2, 3, -6, -2, 5, -3, 6, -5, -7, 14, -2, -4, 1, 4, -3, -3, 2, 2, -7, -3, 4, -4, 0, 6, 7, -5, 1, 3, -3, 3, -1, 2, -1, -12, 0, 4, 4, 0, -4, 2, 0, -4, 5, 7, -2, -8, 0, 2, -4, 6, -1, 1, 2, -8, -2, 1, 5, -5, -1, 1, 2, 3, 4, 1, -4, 4, -4, 1, 5, -2, -1, -2, 1, -2, 3, -2, -3, -3, 2, 4, 0, 0, 2, 2, 2, -10, 2, 1, 2, -7, 1, 2, 0, -4, 0, 1, -2, 14, -3, -2, -2, 2, -2, -4, -2, 6, -2, -4, 0, 1, 1, 0, -2, 11, 0, -3, -3, 7, 0, 0, 0, -13, 3, 4, 0, -4, 0, 0, 2, -2, -3, 0, 0, 12, 1, -3, 0, -6, 0, 0, 0, -3, 4, 3, 0, 6, -3, -3, 0, -7, 3, 3, 3, -9, 9, 3, 0, -12, 5, 3, -9, 9, 3, 2, -2, -2, 2, 0, -3, 0, 1, -2, -4, -5, 3, 0, -2, 4, 0, 1, 4, -2, 0, 2, 5, -8, 1, 1, 1, -10, 1, 3, -1, 7, 1, 0, -1, 4, 1, 0, -2, -4, -1, -1, -1, -4, -2, 0, 1, 12, 0, 0, 0, -4, 0, 2, 0, -5, -1, 2, -1, -5, 0, -1, 0, 13, -2, 0, -1, 1, -1, -2, 0, -3, -1, 0, 0, -4, 0, 1, 1]
```

1. The good - the color endpoints (256 values) are very compressible, usually going for ~50% size reduction with the combination of delta encoding and huffman coding
    * In fact, if you forego huffman coding, just doing a 4-bit integer encoding for the top 15 highest frequency will account for ~82% of your components. So the math works out to be $4 \times 0.82 + 12 \times 0.18 = 5.5$ bits per component for a 32% reduction in size.
2. The bad - the weights (16 values) are generally not very compressible. As raw weights, they're generally uniformly distributed when plotted as a histogram. As delta-encodings, beyond a spike at 0, they are generally still uniformly distributed. As a result, the best you can generally do is to encode them directly as 4-bit fields packed together.

So, with an optimal huffman encoder for the color endpoints, you can expect about a 22% improvement in size (3% of extra overhead outside of the huffman coding). With a suboptimal frequency code for the color endpoints, you can expect about a 15% improvement in size (which is identical to option 1, but with a lot more arithmetic intensity).

In the limit, this does not really represent a big improvement for transfer sizes if we are going to offload the intermediate parameters of the astc to a disk cache.

Because the actual decode + encode (BC 6/7) / transcode (BC 1) pipeline is already so fast, a 25% reduction to the astc format will not eliminate the IO stall that will likely underperform (significantly) the disk-less GPU-accelerated real-time transcoding.
