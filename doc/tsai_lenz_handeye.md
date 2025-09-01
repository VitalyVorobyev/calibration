# Tsai-Lenz hand-eye pose estimate

*[by ChatGPT5 Thinking]*

Here’s the math the function `estimate_handeye_dlt` is implementing, step-by-step, with all the pieces spelled out.

# 1) Kinematic equation and motion pairs

We seek the hand–eye transform $X \in \mathrm{SE}(3)$ (camera in gripper):

$$
X=\begin{bmatrix}R_X & t_X\\ 0 & 1\end{bmatrix},\quad R_X\in\mathrm{SO}(3),\; t_X\in\mathbb{R}^3.
$$

For two time indices $i<j$, define **relative motions** in the base→gripper and camera→target chains:

$$
A_{ij} \;=\; {}^{b}\!T_{g,i}^{-1}\,{}^{b}\!T_{g,j}
= \begin{bmatrix}R_{A,ij}&t_{A,ij}\\0&1\end{bmatrix},\qquad
B_{ij} \;=\; {}^{c}\!T_{t,i}\,{}^{c}\!T_{t,j}^{-1}
= \begin{bmatrix}R_{B,ij}&t_{B,ij}\\0&1\end{bmatrix}.
$$

Eye-in-hand kinematics gives the **hand–eye constraint** for every pair:

$$
A_{ij}\,X \;=\; X\,B_{ij} \quad\Longleftrightarrow\quad
\begin{cases}
R_{A,ij}\,R_X \;=\; R_X\,R_{B,ij} & \text{(rotation)}\\
(R_{A,ij}-I)\,t_X \;=\; R_X\,t_{B,ij} - t_{A,ij} & \text{(translation)}.
\end{cases}
\tag{1}
$$

Using **all pairs** $0\le i<j<n$ just means we stack many independent equations of the form (1). If the data are consistent, every pair satisfies the same $X$, and statistically you gain a lot from the $\binom{n}{2}$ equations.

---

# 2) Rotation: Tsai–Lenz linearization over $\mathrm{SO}(3)$

Write relative rotations via the Lie exponential:

$$
R_{A,ij}=\exp(\widehat{\alpha}_{ij}),\quad
R_{B,ij}=\exp(\widehat{\beta}_{ij}),\quad
R_X=\exp(\widehat{r}),
$$

where $\widehat{(\cdot)}$ maps a vector in $\mathbb{R}^3$ to a skew matrix, and vectors $\alpha,\beta,r\in\mathbb{R}^3$ are angle-axis (“rotation vectors”).

From the rotation block in (1):

$$
R_{A,ij}\,R_X \;=\; R_X\,R_{B,ij}
\quad\Longleftrightarrow\quad
R_{A,ij}\,R_X\,R_{B,ij}^{\!\top}\,R_X^{\!\top} = I.
\tag{2}
$$

Let $\Delta R_{ij} := R_{A,ij}\,R_X\,R_{B,ij}^{\!\top}\,R_X^{\!\top}$.
In noise-free data, $\Delta R_{ij}=I$. For small relative motions (which is typical for frame-to-frame or modest baselines), apply a first-order BCH linearization on $\mathrm{SO}(3)$. The classic Tsai–Lenz / Park–Martin derivation yields the **linear equation in $r$**:

$$
\underbrace{[\alpha_{ij}+\beta_{ij}]_\times}_{3\times 3} \; r
\;=\;
\beta_{ij} - \alpha_{ij},
\tag{3}
$$

where $[\cdot]_\times$ is the skew-symmetric matrix such that $[x]_\times y = x\times y$.

Intuition (one-liner): expand $\exp(\widehat{\alpha})\exp(\widehat{r})$ and $\exp(\widehat{r})\exp(\widehat{\beta})$ to first order, match terms, and rearrange; the non-commutativity shows up as a cross-product with $r$, giving (3).

**Stacking all pairs.** For $m$ selected motion pairs (we take all $i<j$), stack rows

$$
M r = d,\qquad
M = \begin{bmatrix}
[\alpha_1+\beta_1]_\times\\ \vdots\\ [\alpha_m+\beta_m]_\times
\end{bmatrix},\quad
d=\begin{bmatrix}\beta_1-\alpha_1\\ \vdots\\ \beta_m-\alpha_m\end{bmatrix}.
\tag{4}
$$

**Weighting.** Small-angle motions are noisy/low-information; we weight each pair $k$ by $w_k\propto\min(\|\alpha_k\|,\|\beta_k\|)$. In practice we scale each 3-row block and the matching entry in $d$ by $\sqrt{w_k}$ (this is equivalent to diagonal pre-whitening):

$$
M_w r = d_w,\quad M_w = W M,\; d_w = W d,\; W=\operatorname{diag}(\sqrt{w_1}I_3,\ldots,\sqrt{w_m}I_3).
\tag{5}
$$

**Solve.** Use a numerically stable (possibly ridge-regularized) least squares:

$$
\hat r
= \arg\min_r \|M_w r - d_w\|^2
= (M_w^\top M_w + \lambda I)^{-1} M_w^\top d_w.
\tag{6}
$$

Then recover the rotation

$$
\hat R_X = \exp(\widehat{\hat r}).
\tag{7}
$$

**Implementation notes mirrored by the function**

* Before taking $\log$, each input rotation is **projected to $\mathrm{SO}(3)$** (SVD) to remove tiny shear/scale from numeric drift.
* **Degenerate pairs** are dropped: $\min(\|\alpha\|,\|\beta\|)$ below a threshold, or nearly parallel rotation axes (keeps $M$ well-conditioned).
* Ridge $\lambda$ is tiny (e.g. $10^{-12}$)—only there for near singular $M^\top M$.

---

# 3) Translation: linear least squares given $\hat R_X$

From the translation block in (1) for each pair:

$$
(R_{A,ij}-I)\,t_X \;=\; R_X\,t_{B,ij} - t_{A,ij}.
\tag{8}
$$

This is **exact** (no linearization) once $R_X$ is fixed. Stack all weighted equations:

$$
C\, t_X = w,\qquad
C = \begin{bmatrix}\sqrt{w_1}(R_{A,1}-I)\\ \vdots\\ \sqrt{w_m}(R_{A,m}-I)\end{bmatrix},\quad
w = \begin{bmatrix}\sqrt{w_1}(\hat R_X t_{B,1}-t_{A,1})\\ \vdots\end{bmatrix}.
\tag{9}
$$

Solve (again with optional ridge):

$$
\hat t_X = (C^\top C + \lambda I)^{-1} C^\top w.
\tag{10}
$$

---

# 4) Why “all pairs” helps

For $n$ timestamps you have $m=\binom{n}{2}$ independent constraints. Under a simple Gaussian noise model on pose increments, stacking more pairs:

* increases the rank margin of $M$ and $C$,
* averages down noise, especially when rotations cover **at least two non-parallel axes**,
* amortizes the weakness of any single small/degenerate step.

You still need **exciting motions**: if all rotation axes are (nearly) the same, $M$ tends to rank-deficient and $\hat r$ becomes ill-conditioned; similarly, if all $R_{A,ij}\approx I$, then $C$ is weak for $t_X$.

---

# 5) Error model and weighting rationale

* The linear rotation model (3) is a **first-order** approximation in $\alpha,\beta$. It is excellent up to $\sim\!20^\circ$ increments; beyond that, approximation error grows. Empirically, very **small** increments are dominated by sensor/estimation noise; very **large** increments add model error. A simple and effective compromise is to **down-weight very small** pairs (as done here); you can also cap weights for very large angles, or use $w_k \propto \sin\big(\tfrac{\|\alpha_k\|}{2}\big)+\sin\big(\tfrac{\|\beta_k\|}{2}\big)$.
* The translation equations (8) are exact given $R_X$; we reuse the same weights so pairs with informative rotations influence translation more.

---

# 6) Summary (algorithm the function runs)

1. Build **all** motion pairs $A_{ij},B_{ij}$ from input pose sequences.
2. For each pair:

   * Orthonormalize rotations, compute $\alpha_{ij}=\log R_{A,ij}$, $\beta_{ij}=\log R_{B,ij}$.
   * Compute weight $w_{ij}=\min(\|\alpha_{ij}\|,\|\beta_{ij}\|)$; **discard** if too small or axes nearly parallel.
3. **Rotation**: form $M$ and $d$ from (3), scale by $\sqrt{w}$, solve (6), set $\hat R_X = \exp(\widehat{\hat r})$.
4. **Translation**: form $C,w$ from (9) and solve (10) for $\hat t_X$.
5. Return $\hat X = (\hat R_X,\hat t_X)$.

That’s the complete math behind `estimate_handeye_dlt`: Tsai–Lenz’s linear rotation solve extended to **all pairs with weighting and filtering**, followed by the **exact** linear translation solve conditioned on $\hat R_X$.
