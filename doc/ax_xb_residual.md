Here’s the math, end-to-end, for the residual we called `AX_XBResidual`, plus one subtle but important sign detail.

---

# Setup and notation

* Hand–eye unknown: $X \in \mathrm{SE}(3)$, the **camera in gripper** transform $^{g}\!T_c$.

  $$
  X=\begin{bmatrix}R_X & t_X\\ 0 & 1\end{bmatrix},\quad R_X\in\mathrm{SO}(3),\ t_X\in\mathbb{R}^3.
  $$
* From two timestamps $i<j$, build **relative motions**:

  $$
  A = {}^{b}\!T_{g,i}^{-1}\,{}^{b}\!T_{g,j}=\begin{bmatrix}R_A&t_A\\0&1\end{bmatrix},\qquad
  B = {}^{c}\!T_{t,i}\,{}^{c}\!T_{t,j}^{-1}=\begin{bmatrix}R_B&t_B\\0&1\end{bmatrix}.
  $$
* Eye-in-hand kinematics implies the classic **hand–eye constraint** for each motion pair:

  $$
  A\,X \;=\; X\,B.
  $$

Write it in block form:

$$
\underbrace{\begin{bmatrix}R_A & t_A\\ 0&1\end{bmatrix}}_{A}
\underbrace{\begin{bmatrix}R_X & t_X\\ 0&1\end{bmatrix}}_{X}
=
\underbrace{\begin{bmatrix}R_X & t_X\\ 0&1\end{bmatrix}}_{X}
\underbrace{\begin{bmatrix}R_B & t_B\\ 0&1\end{bmatrix}}_{B}.
$$

This gives two coupled equations:

* **Rotation equation**

  $$
  R_A\,R_X \;=\; R_X\,R_B.
  \tag{1}
  $$
* **Translation equation**

  $$
  R_A\,t_X + t_A \;=\; R_X\,t_B + t_X
  \;\;\Longleftrightarrow\;\;
  (R_A-I)\,t_X = R_X\,t_B - t_A.
  \tag{2}
  $$

(2) is exactly the linear Tsai–Lenz translation relation we’re solving in least squares.

---

# Rotation residual construction

From (1),

$$
R_A\,R_X\,R_B^{\!\top}\,R_X^{\!\top}=I.
$$

Define the **rotation misfit**

$$
\Delta R \;\stackrel{\text{def}}{=}\; R_A\,R_X\,R_B^{\!\top}\,R_X^{\!\top}\in\mathrm{SO}(3).
\tag{3}
$$

In the noise-free case, $\Delta R=I$. A natural, right-invariant error on $\mathrm{SO}(3)$ is the **Lie log**:

$$
r_{\text{rot}} \;=\; \operatorname{Log}(\Delta R)\in\mathbb{R}^3,
\tag{4}
$$

where $\operatorname{Log}:\mathrm{SO}(3)\to\mathfrak{so}(3)\cong\mathbb{R}^3$ maps a rotation to its angle-axis vector ($\theta\mathbf{a}$).

Concretely (what Ceres does when you feed it a 3×3 rotation):

* Convert $\Delta R$ to an angle-axis vector $r_{\text{rot}}$.
* In small angle, $r_{\text{rot}}\approx \mathrm{vee}(\Delta R - \Delta R^\top)/2$.

> ⚠️ **Sign detail to verify in code:**
> The residual must use $R_B^{\!\top}$ (i.e., $R_B^{-1}$) as in (3).
> If you compute `RS = RA * RX * RB * RX.transpose()`, that equals $R_A R_X R_B R_X^\top$, which would only be identity if (1) were $R_A R_X = R_X R_B^{\!\top}$—not our definition of $B$.
> **Correct is:**
>
> ```cpp
> const Eigen::Matrix<T,3,3> RS = RA * RX * RB.transpose() * RX.transpose();
> ```
>
> Then `RotationMatrixToAngleAxis(RS, aa)` yields the proper $r_{\text{rot}}$.

---

# Translation residual construction

Use (2) directly as a 3-vector residual:

$$
r_{\text{tr}} \;=\; (R_A - I)\,t_X \;-\; (R_X\,t_B - t_A)\;\in\mathbb{R}^3.
\tag{5}
$$

When $(R_X,t_X)$ satisfy (1)–(2), $r_{\text{tr}}=0$.

---

# Combined 6-D residual (what `AX_XBResidual` emits)

Per motion pair $(A,B)$ we stack

$$
r \;=\; 
\begin{bmatrix}
r_{\text{rot}} \\ r_{\text{tr}}
\end{bmatrix}
\in\mathbb{R}^6,
\qquad
\text{optionally pre-scaled by }\sqrt{w}\text{ (motion weight)}.
\tag{6}
$$

* Units: $r_{\text{rot}}$ in **radians**, $r_{\text{tr}}$ in **length units** (m or mm).
  If you want balanced influence, you can scale translation by $1/\sigma_t$ and rotation by $1/\sigma_\theta$, or use a robust loss on the whole 6-vector.
* Weighting: a simple, information-proportional choice is $w=\min(\|\log R_A\|,\|\log R_B\|)$, since tiny rotations carry little orientation information.

Ceres implementation (conceptually):

```cpp
// Build RS = R_A R_X R_B^T R_X^T  →  aa (3)
T RS_cols[9] = { ... };
T aa[3];
ceres::RotationMatrixToAngleAxis(RS_cols, aa);

// Build et = (R_A - I) t_X - (R_X t_B - t_A)  (3)
Eigen::Matrix<T,3,1> et = (RA - I)*tX - (RX*tB - tA);

// Stack, with optional sqrt weight s
residuals[0..2] = s * aa[0..2];
residuals[3..5] = s * et[0..2];
```

(Use `QuaternionParameterization` for $R_X$ so optimization is 3-dof on the manifold while storing 4 numbers.)

---

# Relation to the linear Tsai–Lenz rotation solve

Linearizing (1) with the Lie‐log (let $\alpha=\log R_A,\ \beta=\log R_B,\ r=\log R_X$) yields the classic equation

$$
[\alpha+\beta]_\times\,r \;\approx\; \beta - \alpha,
\tag{7}
$$

stacked over motion pairs and solved in least squares for $r$, then $R_X=\exp(r)$. That’s your initializer. The non-linear residual above simply removes the small-angle approximation and enforces the full group equation.

---

# Why this residual is well-behaved

* **Rotation term** uses a **bi-invariant** distance on $\mathrm{SO}(3)$: right/left multiplying both sides by any fixed rotation doesn’t change $r_{\text{rot}}$. It exactly penalizes the deviation from (1).
* **Translation term** comes directly from (2). It ties $t_X$ to both $R_X$ and $(t_A,t_B)$.
  If motions are near-pure rotations about a single axis, the problem is weak in translation; weighting and a good motion set help.

---

# Degeneracy and weighting

* If $\| \log R_A\|$ or $\| \log R_B\|$ is tiny, the rotation constraint carries little information—weight those pairs down or drop them.
* If rotation axes are nearly parallel across all pairs, the rotation block is ill-conditioned; use diverse motions (at least two non-parallel axes).

---

# Summary and the one fix to apply

The `AX_XBResidual` is just the stacked Lie-log of the **rotation closure** (3) and the **linear translation** (5), optionally weighted. It’s the exact, non-linear version of Tsai–Lenz, solved jointly in Ceres.

**Fix in your implementation:** make sure the rotation misfit uses **$R_B^{\!\top}$**:

```cpp
const Eigen::Matrix<T,3,3> RS = RA * RX * RB.transpose() * RX.transpose();
```

Everything else in your derivation (including the translation residual) aligns with the math above.
