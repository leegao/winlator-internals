---
title: "Optimal Quantization Mode Search within ASTC"
date: 2025-11-18
---


# "Optimal" Quantization Mode Search within ASTC

Let's say we just spent a generous number of flops doing an incredible job solving the crap out of a series of linear algebra or descent problems in order to find the perfect set of parameters for an ASTC block. What do you do now?

Well, one final step before you write out your block - select your quantization mode.

ASTC supports per-block dynamic quantization modes for both the color endpoints, as well as your matrix of weights. This is a blessing and a curse - you can do a lot of things to really fine-tune the quantization mode that best suits your block, but doing so can be complicated since you have a potentially large search space.

In this post, I'll present a heuristic that approximately minimizes the $L^2$ error of the optimal quantization mode search, without doing any discrete combinatorial search.

---

**TL;DR:**

The optimal number of bits per color endpoint component per pixel is given by

$$
\hat{Q^*}_c(\delta_j, w_i) = \frac{A + \log_2{\frac{1 + 2w_i}{B\delta_j}}}{1 + B}
$$

where

1. $A = \frac{\text{number of bits not in header+metadata}}{16}$, so $\frac{111}{16}$ for single partition (17 bits of header + metadata), and $\frac{99}{16}$ for double partition (29 bits of header + metadata + partition seed)
2. $B = \frac{\text{number of color endpoint components}}{16}$, so 6 for 1 x RGB, 8 for 1 x RGBA, 12 for 2 x RGB, and 16 for 2 x RGBA
3. $\delta_j = ep_1[j] - ep_0[j]$ is the "spread" or the difference between the first and second color endpoint component for this pixel
4. $w_i$ is the $i^{th}$ weight of the pixel

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

or we want to minimize the pixel reconstruction error of the image $\hat{\mathsf{pixel}}$ decoded from $Q_{c,w}(\theta)$ relative to the ground truth ${\mathsf{pixel}}$.

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

Here, the $\mathsf{round}(0.33 \times 7) = \mathsf{round}(2.31) = 2$ selects an "index" 2 within your discrete set of 8 values:

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

so this effectively quantizes your 0.33 down to 0.286 (with an absolute error of $0.33 - 0.286$)

As you can probably intuit, a larger quantization range results in lower average absolute errors, because your discrete value points are closer together.

In particular, the average error of uniformly distributed data when quantizing to range $R = 2^b$ (where $b$ can be fractional) is given by the following formula:

$$
E[\mathsf{abs}(\epsilon_{b})] = 2^{-b - 2}
$$

(technically, it is $2^{-\log2(R-1) - 2}$, but we won't fret too much about this, there are errors in means too)

The cross error defined by

$$
E[\mathsf{abs}(\epsilon_{b} - \epsilon_{b}')] = \frac{2^{-b}}{3}
$$

which is a little bit bigger than just the average absolute quantization error.

### Characterizing the per-pixel reconstruction error

Okay, so let's say I have a pair of quantization modes $(Q_c, Q_w)$ in units of bits. Based on the error bounds above, we can actually give a bounded error form to our quantization functions:

$$
\begin{align*}
Q_c(x) &= \mathsf{round}(x \times (2^{Q_c})) / 2^{Q_c} &= x + \epsilon_{c} \\
Q_w(w) &= \mathsf{round}(w \times (2^{Q_w})) / 2^{Q_w} &= w + \epsilon_{w} \\
\end{align*}
$$

where $\mathbb{E}[\mathsf{abs}(\epsilon_c)] = 2^{-Q_c-2}$ and $\mathbb{E}[\mathsf{abs}(\epsilon_w)] = 2^{-Q_w-2}$

This then lends itself to a natural "perturbation" form of our full problem in terms of these probabilistic epsilons. Taking the red ($r$) component for the $i^{th}$ pixel as an example:

$$
\begin{align*}
\hat{\mathsf{pixel}}_{r,i} &= Q_c(r_0) \times (1 - Q_w(w_i)) + Q_c(r_1) \times Q_w(w_i) \\
&=(r_0 + \epsilon_c)(1 - w_i - \epsilon_w) + (r_1 + \epsilon_{c'}) \times (w_i + \epsilon_w) \\
&= r_0(1-w_i) - r_0\epsilon_w + \epsilon_c (1-w_i) + r_1w_i + r_1\epsilon_w +\epsilon_{c'}w_i + O(\epsilon^2) \\
&= \mathsf{pixel}_{r,i} + \underbrace{(r_1 - r_0)}_{\delta_r}\epsilon_w + (1-w_i)\epsilon_c + w_i\epsilon_{c}' + O(\epsilon^2)
\end{align*}
$$

Now, we don't know exactly what these $\epsilon$s are, as they are probabilistic variables sampled from the $\epsilon_b$ distribution described above. However, we can make the (reasonable) assumption that in the limit of large images, the quantization errors of the weights and endpoint colors tend to be drawn uniformly. 

As a result, we can use the heuristic where we substitute the $\epsilon$s with their expected absolute value, in order to compute an _expected_ pixel reconstruction error.

This then gives us a construction of the _expected_ pixel reconstruction error of:

$$
\begin{align*}
E(\epsilon, \delta_r, w_i) &= \hat{\mathsf{pixel}}_i - \mathsf{pixel}_i = \delta\epsilon_w + \epsilon_c + w_i(\epsilon_c' - \epsilon_c) + O(\epsilon^2) \\
\mathbb{E}_\epsilon[\mathsf{abs}(E)] &= \delta \underbrace{2^{-Q_w - 2}}_{\mathbb{E}[|\epsilon_w|]} + \underbrace{2^{-Q_c - 2}}_{\mathbb{E}[|\epsilon_c|]} + w\underbrace{\frac{2^{-Q_c}}3}_{\mathbb{E}[|\epsilon_c - \epsilon_c'|]} \\
&= \delta 2^{-Q_w-2} + k2^{-Q_c - 2}
\end{align*}
$$

where we parameterize $\delta_r =\mathsf{abs}(r_1 - r_0)$ to be the "color spread" for the red component/color channel within this block, and $k_i = (1 + \frac{4w_i}{3})$ to be a weight-sensitivity factor for convenience.

Recall from above that

$$
Q^*_w(Q_c) \approx \frac{111 - 6 \times Q_c}{16} = A - BQ_c
$$

then we can characterize $\epsilon_w$ in terms of $\epsilon_c$:

$$
\epsilon_w = 2^{-Q_w(Q_c) - 2} = 2^{-(A-BQ_c) - 2}
$$

so that the average reconstruction error (assuming that your weights and color endpoints are uniformly distributed) comes out to be

$$
E(Q_c, \delta_j, k_i) = \delta_j 2^{-(A-BQ_c) - 2} + k_i2^{-Q_c - 2}
$$

where we can omit the $O(\epsilon^2)$ term since it's at most $2^{-11}$.

### Single Element Optimization

We can actually analytically minimize this equation relative to $Q_c$:

$$
\begin{align*}
E(x, \delta_j, k_i) &= \delta_j 2^{-(A-Bx) - 2} + k_i2^{-x - 2} \\
& \text{isolate constants} \\
&= 2^{-2} (\delta_j2^{Bx - A} + k_i 2^{-x}) \\
& \text{taking the x derivative mod constants} \\
C \dfrac{dE}{dx} &= k_i2^{-x} - B\delta_j2^{Bx-A} \\
& \text{set to 0 to optimize} \\
B\delta_j2^{Bx-A} &= k_i2^{-x} \\
\hat{Q}_c(\delta_j, w_i) &= \frac{A + \log_2{\frac{k_i}{B\delta_j}}}{1 + B}
\end{align*}
$$

Woot! We now have an analytic description of the optimal value as a "tug-of-war" between $k_i$ (AKA $w_i$) and $\delta$:

$$
\log_2{k_i} - \log_2{B\delta}
$$

(note: you can, and should, make use of libraries like sympy to do these symbolic differentiations)

```python
from sympy import symbols, diff, solve, log, simplify, Eq, cancel
Q = symbols('Q', real=True, positive=True)
A, B = symbols('A B', real=True, positive=True)
delta = symbols('delta', real=True, positive=True)
k = symbols('k', real=True, positive=True)

E = delta * 2**(-(A - B*Q) - 2) + k * 2**(-Q - 2)
dE_dQ = diff(E, Q)
print("Derivative:", dE_dQ)
print("Solution:", solve(dE_dQ, Q))

# Derivative: -2**(-Q - 2)*k*log(2) + 2**(-A + B*Q - 2)*B*delta*log(2)
# Solution: log((2**A*k/(B*delta))**(1/((B + 1)*log(2))))
```

#### Perturbation Analysis

We can take the partial derivative of $\hat{Q}_c$ wrt $d\delta$ and $dw$ to understand the behavior of the solution space:

$$
\begin{align*}
\dfrac{dQ_c}{d\delta} &= - \frac{\delta^{-1}}{(1 + B)\ln 2} \\
\dfrac{dQ_c}{dw} &= + \frac{\frac{4}{3}k^{-1}}{(1 + B)\ln 2}
\end{align*}
$$

Observations:

1. $Q_c$ decreases with increasing $\delta$ - that is, as your color endpoints become more spread apart, you want to spend more bits on weights. Conversely, as your colors become more packed together, you'll need more precision for your colors.
2. In particular, as $\delta$ gets smaller, $Q_c$ becomes a lot more sensitive, becoming almost singular the closer it is to 0. In other words, if your color spread is close to 0, then small changes in your color spread will cause a large change in the $Q_c$. At this range, you're better off just capping to the max-$Q_c$ mode.
3. $Q_c$ increases with increasing $w$ - that is, the larger your weights are, the (ironically) fewer bits you will want to spend on $Q_w$.
4. As $w$ becomes smaller, $Q_c$ becomes slightly more sensitive to it. But in general, $Q_c$ is not very sensitive to perturbations in the weights.
5. The sensitivity frontier happens at ~ $\delta = \frac{3}{4}k = w + \frac{3}{4}$. Below this line ($\delta - w < 0.75$), the effect of $\delta$ dominates; above it ($w < \delta - 0.75$), $w$ dominates. In general, $\delta$ will be the dominating factor given the geometry of this solution space (since $\delta$ must be big _and_ $w$ must be small for this to not be the case).

The intuition to draw from this is that if your pair of endpoint colors are very close to each other, you'll want to use more color precision to represent their differences. If your weights are large, you'll want to spend more bits to represent them. In the vast majority of the cases, $\delta - w < 0.75$ and the effect of your $\delta$ dominates the decision on where to spend your bits.

### M x N Optimization

So far, we've only been considering the subproblem where your weights and pairs of $ep_0, ep_1$ are scalars (e.g. just the red, green, or blue components). Let's now extend our error function to the multi-pixels (N) x multi-channel (M) problem.

#### Direct Solution

One way to approach this problem is to calculate the sum of the errors. Let $N$ be the # of pixels (16), and $M$ be the number of channels (3 for RGB, 4 for RGBA), then

$$
\begin{align*}
E_\sigma(Q_c, \delta, k) &= \sum_{i,j}^{N,M} E(Q_c, \delta_j, k_i) \\
&=\sum_{i,j} \delta_j 2^{-Q_w - 2} + \sum_{i,j} k_i2^{-Q_c-2} \\
&= N 2^{-Q_w - 2} \sum_j \delta_j + M 2^{-Q_c-2} \sum_i k_i \\
& \text{define the averages } \tilde{\Delta} = \Sigma_j\delta_j/M, \tilde{K} = \Sigma_ik_i/N \\
&= NM \times (\tilde\Delta 2^{-Q_w - 2} + \tilde K 2^{-Q_c - 2}) \\
&= NM \times E(Q_c, \tilde \Delta, \tilde K)
\end{align*}
$$

In other words, the sum (or the $L^1$ norm)  of the absolute expected reconstruction error is identical to $N \times M$ times the reconstruction error of the mean/average color spread and weight sensitivity: $\tilde\Delta, \tilde K$.

Note that $E_\sigma$ is just $NM \times E_{L^1}$, the $L^1$ error over the $(N \times M)$ pixel tensor.

The optimal $\tilde Q_c$ is now given by the same 

$$
{Q}_c(\tilde \Delta, \tilde K) = \frac{A + \log_2\left(\frac{ \tilde K}{B \cdot \tilde \Delta}\right)}{1 + B}
$$

with the same analytical properties as before.

A crucial observation: this solution is completely parameterized by the mean/average of $\delta_j, k_i$s, meaning that **it is invariant to variance in $\delta, k$**. This may or may not be a desired property.

From an optimization perspective, this is the optimal solution to minimize the $L^1$ (sum of absolute errors) norm of the errors.

#### Alternative Formulation 1: $L^2$ Reconstruction Error

We can also extend this to the natural $L^2$ form as well.

$$
\begin{align*}
||E(Q_c, \delta, k)||_2^2 &= \sum_{i,j}^{N,M}\frac{E(Q_c, \delta_j, k_i)^2}{NM} \\
&= \sum_{i,j} \frac{\delta_j^2 \epsilon_w^2 + k_i^2 \epsilon_c^2 + \overbrace{2\delta_j k_i \epsilon_c \epsilon_w}^{=O(2^{-10})}}{NM}
\end{align*}
$$

TBD

TODO: note that this is a classic statistical term that is dependent on the sample variance.

#### Alternative Formulation 2: Mean of vector of Reconstruction Errors

If you're disturbed by the lack of 

TODO: reframe this as a reparameterization of the L1 optimization with variance adjusted based on information density of the variance of both k and d.

### Packing

At the end of this process, we have a $Q_c$ between 1 and 8 bits that denotes the optimal color quantization mode (in some fractional bits). However, not all (most) $Q_c$s are valid ASTC quantization mode. We can do a final step to iterate through all valid ASTC modes, and snap $Q_c$ to the closest valid one. However, if the quantization levels available in ASTC is coarse for a particular $Q_c$, it is probably better to perform a "two-tap" procedure:

1. Identify the two valid quantization methods closest to $Q^*_c$
2. Calculate the reconstruction error loss for each (either by actually reconstructing the pixels, or using the approximation $E$ above)
3. Return the best quantization method between the two.

### Final Algorithm

```python
# Assume we're doing single partition
M = block.channels # 3 for RGB, 4 for RGBA

A = 111 / 16
B = 2 * M / 16

Delta = abs(block.ep1 - block.ep0).flatten().mean() # + ep2,ep3 too
K = (1 + 4 * block.weights / 3).mean()
Qc = clamp((A + math.log2(K / (B * Delta))) / (1 + B), 1, 8)

# Look for the pair of modes closes to Qc
for i, mode in enumerate(ASTC_MODES):
	next_mode = ASTC_MODES[i+1]
	if mode.color_bits <= Qc <= next_mode.color_bits:
		return # either mode or next_mode based on lower error
```
