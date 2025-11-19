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
C \dfrac{dE}{dx} &= (1 + 2w_i)2^{-x} - B\delta2^{Bx-A} \\
& \text{set to 0 to optimize} \\
B\delta2^{Bx-A} &= (1 + 2w_i)2^{-x} \\
\hat{Q}_c(\delta, w_i) &= \frac{A + \log_2{\frac{1 + 2w_i}{B\delta}}}{1 + B}
\end{align*}
$$

Woot! We now have an analytic description of the optimal value as a "tug-of-war" between $w_i$ and $\delta$:

$$
\log_2{(1 + 2w_i)} - \log_2{B\delta}
$$

(note: you can, and should, make use of libraries like sympy to do these symbolic differentiations)

```python
from sympy import symbols, diff, solve, log, simplify, Eq, cancel
Q = symbols('Q', real=True, positive=True)
A, B = symbols('A B', real=True, positive=True)
delta = symbols('delta', real=True, positive=True)
w = symbols('w', real=True, positive=True)

E = delta * 2**(-(A - B*Q) - 2) + (1 + 2*w) * 2**(-Q - 2)
dE_dQ = diff(E, Q)
print("Derivative:", dE_dQ)
print("Solution:", solve(dE_dQ, Q))

# Derivative: -2**(-Q - 2)*(2*w + 1)*log(2) + 2**(-A + B*Q - 2)*B*delta*log(2)
# Solution: log((2**A*(2*w + 1)/(B*delta))**(1/((B + 1)*log(2))))
```

#### Perturbation Analysis

We can take the partial derivative of $\hat{Q}_c$ wrt $d\delta$ and $dw$ to understand the behavior of the solution space:

$$
\begin{align*}
\dfrac{dQ_c}{d\delta} &= - \frac{\delta^{-1}}{(1 + B)\ln 2} \\
\dfrac{dQ_c}{dw} &= + \frac{2(1 + 2w)^{-1}}{(1 + B)\ln 2}
\end{align*}
$$

Observations:

1. $Q_c$ decreases with increasing $\delta$ - that is, as your color endpoints become more spread apart, you want to spend more bits on weights. Conversely, as your colors become more packed together, you'll need more precision for your colors.
2. In particular, as $\delta$ gets smaller, $Q_c$ becomes a lot more sensitive, becoming almost singular the closer it is to 0. In other words, if your color spread is close to 0, then small changes in your color spread will cause a large change in the $Q_c$. At this range, you're better off just capping to the max-$Q_c$ mode.
3. $Q_c$ increases with increasing $w$ - that is, the larger your weights are, the (ironically) fewer bits you will want to spend on $Q_w$.
4. As $w$ becomes smaller, $Q_c$ becomes slightly more sensitive to it. But in general, $Q_c$ is not very sensitive to perturbations in the weights.
5. The sensitivity frontier happens at ~ $\delta - w = 0.5$. Below this line, the effect of $\delta$ dominates, above, $w$ dominates. In general, $\delta$ will be the dominating factor given the geometry of this solution space.

The intuition to draw from this is that if your pair of endpoint colors are very close to each other, you'll want to use more color precision to represent their differences. If your weights are large, you'll want to spend more bits to represent them. In most of the cases, $\delta - w < 0.5$ and the effect of your $\delta$ dominates the decision on where to spend your bits.

### Minimizing the full vec-16

Unfortunately, the full $L^2$ loss function on the pixel reconstruction error is much more unwieldy to use or analyze. Instead, I propose a weight-average heuristic that approximates the optimal argmin of the $L^2$ loss based on the single-element optimization problem:

$$
Q_c^* \approx \frac{\sum_i (1 + 2w_i) \hat{Q}(\delta, w_i)}{\sum_i(1 + 2w_i)}
$$

note that $\delta$ is invariant wrt the weight indices.

#### Perturbation Analysis

Intuitively, this approximation should be good if the spread of $w_i$ itself is low. Let us also look at how sensitive $Q_c$ is to this spread.

Let's transform the problem slightly. First, let $K_i = 1 + 2w_i$ denote the "sensitivity" factor of $Q_c$ (since increasing $K$ increases $Q_c$ almost linearly when $K$ is close to 1). We can also reduce the weighted means $Q_c$ to just the variables containing $K$:

$$
\begin{align*}
F(K) &= \frac{\sum_i K_i Q_i}{\sum_i K_i} \\
& \text{since all but the } \log_2(1+2w_i) \text{ term are independent of K} \\
&\sim C\frac{\sum_i K_i \log(K_i)}{\sum_i K_i}
\end{align*}
$$

where $C = (1+B)^{-1}$ is the scale factor from $Q_c$

We can further reduce this problem to just a 2-pixels problem of the spread of $K$:

1. let $K_{min} = \mu_K - d_K = \mu - \frac{\Delta_K}{2}$
2. let $K_{max} = \mu_K + d_K = \mu + \frac{\Delta_K}{2}$

where $\mu_K$ is the mean sensitivity $K_i$, and $d_K$ is its variance (defined as half the spread $\Delta_K = K_{max} - K_{min}$.

From this, we can reparameterize $F(K)$ into one that looks at the max and min pixels only:

$$
F(d_K) = C\frac{ \overbrace{(\mu_K - d_K)\log(\mu_K - d_K)}^{K_{min}} + \overbrace{(\mu_K + d_K)\log(\mu_K + d_K)}^{K_{max}}}{(\mu_K - d_K) + (\mu_K + d_K)}
$$

taking the directional derivative with the variance $\dfrac{\partial F}{\partial d_K}$ yields:

$$
\begin{align*}
\frac{\partial F}{\partial d_K} &= \frac{C}{2\mu_K} (\log(\mu_k + d_k) - \log(\mu_k - d_k) ) \\
&= \frac{C}{2\mu_K} \log\left(\frac{K_{max}}{K_{min}} = 1 + \frac{2d_k}{K_{min}}\right)
\end{align*}
$$

when $d_K$ is small, this term approaches $0$ and behaves almost linearly (since $\log(1+\epsilon) \approx 1 + \epsilon$; when $d_K$ is large, this term approaches $\log_2(3)$ and behaves logarithmically (less sensitive).

This suggests that:

1. As the spread of your weights $w_i$ goes up, you're going to want to spend more bits on $Q_c$ (color precision)
2. The decision will be dominated by pixels with large $w_i$ values, that is, the votes of pixels with large $w_i$ will overwhelmingly dominate that of smaller $w_i$s.

This means that if our spread is high, we'll just pull the $Q_c$ up, which is a desirable property.

### Multichannel Optimization

So far, we've only been considering the subproblem where your pair of $ep_0, ep_1$ are scalars (e.g. just the red, green, or blue components). Let's now extend our error function to the multichannel problem (3 for RGB, or 4 for RGBA).

#### Weighted Mean

A straightforward extension here is to compute the same weighted sum as before, just now summing over the color channels:

$$
Q_c = \frac{\sum_{i,j} (1 + 2w_i) \hat{Q}(\delta_j, w_i)}{\sum_{i,j}(1 + 2w_i)} = \mathsf{mean}(Q^R_c, Q^G_c, Q^B_c)
$$

Note that in this formulation, $Q_c$ gives equal vote to each color channel, which may not be great if you have a block where one channel is flat while the others are high.

Instead, we can formulate another variant of the problem that allows all components to participate in voting for their favorite $Q_c$ allocation.

#### Alternative Solution - Global Optimization

To start, let's see how our cost function $E$ changes. 

We can parameterize it by $\Delta_\Sigma = \delta_R + \delta_G + \cdots$ which is effectively the sum of the spread of the color endpoints, and $K_\Sigma = \sum_i 1 + 2w_i$. Let $M$ be the number of color components (3 or 4), then doing the same error propagation on the reconstruction error will yield:

$$
4 E_{\Sigma}(Q_c, \Delta_\Sigma, K_\Sigma) \approx \underbrace{\Delta_{\Sigma} \cdot 2^{-Q_w}}_{\text{Weighted Gradient Noise}} + \underbrace{(M \cdot K_{\Sigma}) \cdot 2^{-Q_c}}_{\text{Endpoint Quantization Noise}} 
$$

where $Q_w = A - BQ_c$.

Note that we've moved the sum over the weights within the endpoint quantization noise term within the error directly (meaning we no longer need to do an explicit weighted means).

The optimal $Q_c$ is now given by

$$
\hat{Q}_c(\Delta_\Sigma, K_\Sigma) = \frac{A + \log_2\left(\frac{M \cdot K_{\Sigma}}{B \cdot \Delta_{\Sigma}}\right)}{1 + B}
$$

You'll notice that this solution has the same exact sensitivity to $\delta, K$, however, unlike the weighted-sum solution, this one is completely invariant to the spread of $\delta, K$. This is also a lot more computationally efficient than the earlier solution.

### Packing

At the end of this process, we have a $Q_c$ between 1 and 8 bits that denotes the optimal color quantization mode (in some fractional bits). However, not all (most) $Q_c$s are valid ASTC quantization mode. We can do a final step to iterate through all valid ASTC modes, and snap $Q_c$ to the closest valid one. However, if the quantization levels available in ASTC is coarse for a particular $Q_c$, it is probably better to perform a "two-tap" procedure:

1. Identify the two valid quantization methods closest to $Q^*_c$
2. Calculate the reconstruction error loss for each (either by actually reconstructing the pixels, or using the approximation $E$ above)
3. Return the best quantization method between the two.

### Final Algorithm

```python
# Assume we're doing single partition
M = block.channels # 3 for RGB, 4 for RGBA
Delta = abs(block.ep1 - block.ep0).sum() # + ep2,ep3 too
K = (1 + 2 * block.weights).sum()
A = 111 / 16
B = 2 * M / 16
Qc = clamp((A + math.log2(M * K / (B * Delta))) / (1 + B), 1, 8)

# Look for the pair of modes closes to Qc
for i, mode in enumerate(ASTC_MODES):
	next_mode = ASTC_MODES[i+1]
	if mode.color_bits <= Qc <= next_mode.color_bits:
		return # either mode or next_mode based on lower error
```
