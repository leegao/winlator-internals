---
title: "Optimal Quantization Mode Search within ASTC"
date: 2025-11-18
---


# "Optimal" Quantization Mode Search within ASTC

Let's say we just spent a generous number of flops doing an incredible job solving the crap out of a series of linear algebra or descent problems in order to find the perfect set of parameters for an ASTC block. What do you do now?

Well, one final step before you write out your block - select your quantization mode.

ASTC supports per-block dynamic quantization modes for both the color endpoints, as well as your matrix of weights. This is a blessing and a curse - you can do a lot of things to really fine-tune the quantization mode that best suits your block, but doing so can be complicated since you have a potentially large search space.

In this post, I'll present (a so far seemingly novel?) heuristic that approximately minimizes the $L^2$ error of the optimal quantization mode search, without doing the discrete combinatorial search.

---

## ASTC Parameters

Let's start with a simplified setup. To encode a block of 16 RGB direct pixels (in 4x4 block mode) within ASTC, using a single pair of color endpoints ($2 \times (r, g, b)$, as well as 16 interpolation/lerp weights, you need, well, those 22 specific things as the parameters of an ASTC block:

$$
\theta = \begin{pmatrix}
r_0 \\
g_0 &= ep_0\\
b_0 \\
r_1 \\
g_1 & = ep_1\\
b_1 \\
w_0 \\
\vdots \\
w_{15}
\end{pmatrix}
$$

Now, given the block parameters $\theta$, we need to select a valid quantization mode-pair $(Q_{color}, Q_{weight})$ that can quantize the colors and weights "optimally".

What does it mean to quantize something optimally? Ideally, we'd like to minimize the error/loss of the reconstructed pixel values using the quantization scheme relative to the ground-truth image $\mathsf{pixel}_i$. 

That is, we want to minimize

$$
\mathsf{decode}\begin{pmatrix}
Q_{color}(ep_0)  \\
Q_{color}(ep_1) \\
Q_{weight}(w_i)
\end{pmatrix} - {\mathsf{pixel}}_i
$$

or we want to minimize the pixel reconstruction error of the image $\hat{\mathsf{pixel}}_i$ decoded from $Q_{c,w}(\theta)$ relative to the ground truth ${\mathsf{pixel}}_i$.

## ASTC Requirements

Before we continue, let's actually describe the set of valid ASTC quantization modes for our single-partition rgb block.

An ASTC block consists of 128 bits:

1. 11 bits of header
2. 6 bits of additional metadata (18 bits for 2-partition mode)
3. You specify (within those 6 - 18 bits of metadata) the $Q_w$ (weight quantization mode), and $Q_c$ will be implied to be the best quantization mode that fits the total 128 bits budget
4. (roughly) 6 (because 2 x (R,G,B)) times $Q_c$ bits of color endpoint data
5. (roughly) 16 times $Q_w$ bits of weight data

Note that unlike BCn, ASTC actually allows you to specify fractional bit integer encodings (for e.g. packing 5 "trits" (0,1,2) into 8 bits, or 3 "quints" (0,1,2,3,4) into 8 bits). As a result, $Q_c, Q_w$ may be fractional, but the packed endpoints and weight data must finish at full bit boundaries.

For our problem, we have 17 bits of header and metadata, leaving us a budget of 111 bits to pack $6 \times Q_c + 16 \times Q_w$ bits of data.

This gives us an approximate "budget" function to get $Q_w$ for a given $Q_c$:

$$
Q^*_w(Q_c) \approx \frac{111 - 6 \times Q_c}{16}
$$

## Mathematical Modeling

Alright, with that out of the way, let's get down to the dirty gritty details.

### What is Quantization?

At its simplest, quantization is the process of turning a continuous variable (e.g. a float x $\in [0 .. 1]$ into a discrete set of value ranges.

For example, let's say we're going to quantize 0.33 within the continuous range [0-1] into a discrete set of 8 values. Well, it's as simple as:

$$
Quant_8(0.33) = \frac{\mathsf{round}(0.33 \times 7)}{7}
$$

Here, the `round(0.33 * 7) = round(2.31) = 2` selects an "index" 2 within your discrete set of 8 values:

```python
QUANT_8 = [
  0 / 7 = 0.000 # index 0
  1 / 7 = 0.143 # index 1
  2 / 7 = 0.286 # index 2
  3 / 7 = 0.429 # index 3
  4 / 7 = 0.571 # index 4
  5 / 7 = 0.714 # index 5
  6 / 7 = 0.857 # index 6
  7 / 7 = 1.000 # index 7
]
```

so this effectively quantizes your 0.33 down to 0.286 (with an absolute error of $|0.33 - 0.286|$)

As you can probably intuit, a larger quantization range results in lower average absolute errors, because your discrete value points are closer together.

In particular, the average error of uniformly distributed data when quantizing to range $R = 2^b$ (where $b$ can be fractional) is given by the following formula:

$$
\epsilon_{b} \approx 2^{-b - 2}
$$

(technically, it is $2^{-\log2(R-1) - 2}$, but we won't fret too much about this, there are errors in means too)

### Characterizing the per-weight error

Okay, so let's say I have a pair of quantization modes $(Q_c, Q_w)$ in units of bits. Based on the error bounds above, we can actually give a bounded error form to our quantization functions:

$$
\begin{align*}
Q_c(x) &= \mathsf{round}(x \times (2^{Q_c})) / 2^{Q_c} &= x + \epsilon_{Q_c} \\
Q_w(w) &= \mathsf{round}(w \times (2^{Q_w})) / 2^{Q_w} &= w + \epsilon_{Q_w} \\
\end{align*}
$$

this then lends itself to a "perturbation" form of our full problem:

$$
\begin{align*}
\hat{\mathsf{pixel}}_i &= Q_c(ep_0) \times (1 - Q_w(w_i)) + Q_c(ep_1) \times Q_w(w_i) \\
&=(ep_0 + \epsilon_c)(1 - w_i - \epsilon_w) + (ep_1 + \epsilon_{c'}) \times (w_i + \epsilon_w) \\
&= ep_0(1-w_i) - ep_0\epsilon_w + \epsilon_c (1-w_i) + ep_1w_i + ep_1\epsilon_w +\epsilon_{c'}w_i + O(\epsilon^2) \\
&= \mathsf{pixel}_i + (ep_1 - ep_0)\epsilon_w + (1-w_i)\epsilon_c + w_i\epsilon_{c'} + O(\epsilon^2)
\end{align*}
$$

note that to maximize the error term ($\epsilon$), we can set $\epsilon_{c'} = -\epsilon_{c}$, which gives us a mean pixel reconstruction error of:

$$E
(\epsilon_w, \epsilon_c, \delta, w_i) = \hat{\mathsf{pixel}}_i - \mathsf{pixel}_i = \delta\epsilon_w + (1-2w_i)\epsilon_c + O(\epsilon^2)
$$

where we parameterize $\delta = ep_1 - ep_0$ to be the "color spread" within this block.

Recall from above that

$$
Q^*_w(Q_c) \approx \frac{111 - 6 \times Q_c}{16} = A - BQ_c
$$

then we can characterize $\epsilon_w$ in terms of $\epsilon_c$:

$$
\epsilon_w = 2^{-Q_w(Q_c) - 2} = 2^{-(A-BQ_c) - 2}
$$

so that

$$
E(Q_c, \delta, w_i) = \delta 2^{-(A-BQ_c) - 2} + (1-2w_i)2^{-Q_c - 2}
$$

where we can omit the $O(\epsilon^2)$ term since it's at most $2^{-11}$.

### Single Element Optimization

We can actually analytically minimize this equation relative to $Q_c$:

$$
\begin{align*}
E(x, \delta, w_i) &= \delta 2^{-(A-Bx) - 2} + (1-2w_i)2^{-x - 2} \\
& \text{isolate constants} \\
&= 2^{-2} (\delta2^{Bx - A} + (1 - 2w_i) 2^{-x}) \\
& \text{taking the x derivative mod constants} \\
dE/dx &= (1 + 2w_i)2^{-x} - B\delta2^{Bx-A} \\
& \text{set to 0 to optimize} \\
B\delta2^{Bx-A} &= (1 + 2w_i)2^{-x} \\
Q^*_c(\delta, w_i) &= \frac{A + \log_2{\lparen\frac{1 + 2w_i}{B\delta}\rparen}}{1 + B}
\end{align*}
$$

### Perturbation Analysis

We can take the partial derivative of $Q^*_c$ wrt $d\delta$ and $dw$ to understand the behavior of the solution space:

$$
\begin{align*}
\dfrac{dQ_c}{d\delta} &= - \frac{\delta^{-1}}{(1 + B)\ln 2} \\
\dfrac{dQ_c}{dw} &= + \frac{2(1 + 2w)^{-1}}{(1 + B)\ln 2}
\end{align*}
$$


