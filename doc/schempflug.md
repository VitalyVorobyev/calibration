# Scheimpflug camera calibration

*[by ChatGPT5 Thinking]*

“Scheimpflug” means your sensor plane is **not perpendicular** to the camera’s principal ray (because the lens was tilted to rotate the plane of sharp focus). Geometrically, that’s a **central** camera whose image plane is **oblique**. You can—and should—model this directly in the projection step instead of hoping skew and principal point soak it up.

Below is a practical, math-first recipe you can drop into your calibrator (Ceres, g2o, etc.), plus key identifiability notes.

---

# 1) Coordinate frames and parameters

* World point (homogeneous): $X_w \in \mathbb{R}^3$.

* Camera (pinhole) center at the origin of the **camera frame**; canonical optical axis is $\mathbf{e}_z=(0,0,1)^\top$.

* Extrinsics for view $i$: $X_c = R_i X_w + t_i$.

* **Tilted sensor plane** $\Pi_s$ is defined by:

  * Unit normal $\mathbf{n}_s$ (2 DOF). It is obtained from two “Scheimpflug” angles, e.g. pitch/roll of the sensor plane:

    $$
    R_s(\tau_x,\tau_y) = R_y(\tau_y)\,R_x(\tau_x),\quad
    \mathbf{n}_s = R_s \,\mathbf{e}_z.
    $$
  * Two in-plane unit axes $\mathbf{a}_s, \mathbf{b}_s$ spanning $\Pi_s$:

    $$
    \mathbf{a}_s = R_s\,\mathbf{e}_x,\quad \mathbf{b}_s = R_s\,\mathbf{e}_y,
    $$

    so that $\{\mathbf{a}_s,\mathbf{b}_s,\mathbf{n}_s\}$ is an orthonormal triad.
  * Distance from the pinhole to the plane along its normal: $d>0$ (meters). **Scale ambiguity:** only the product $d$ with focal scales matters; set $d=1$ (or absorb into $f_x,f_y$).

* Pixel mapping (intrinsics):

  $$
  K =
  \begin{bmatrix}
  f_x & s & u_0\\
  0 & f_y & v_0\\
  0 & 0 & 1
  \end{bmatrix},
  $$

  with optional skew $s$. (With tilt present, **do not** force $s=0$ or $f_x=f_y$ during initialization.)

* Distortion (Brown–Conrady or your preferred polynomial), applied in the **metric coordinates on $\Pi_s$** and centered at the principal point on that plane.

---

# 2) Ray–plane intersection (the core Scheimpflug geometry)

Given $X_c$ in the camera frame, the projection ray is $\lambda X_c$, $\lambda>0$.

The tilted sensor plane $\Pi_s$ is

$$
\Pi_s = \{ x\in\mathbb{R}^3 \;|\; \mathbf{n}_s^\top x = d \}.
$$

The intersection scale is

$$
\lambda = \frac{d}{\mathbf{n}_s^\top X_c}.
$$

The 3D intersection point on the plane is $P_s = \lambda X_c$.
Express it in **metric plane coordinates** (meters) using the in-plane basis:

$$
\begin{bmatrix} m_x \\ m_y \end{bmatrix}
= 
\begin{bmatrix}
\mathbf{a}_s^\top \\ \mathbf{b}_s^\top
\end{bmatrix}
P_s
=
\frac{d}{\mathbf{n}_s^\top X_c}
\begin{bmatrix}
\mathbf{a}_s^\top X_c \\
\mathbf{b}_s^\top X_c
\end{bmatrix}.
$$

This is the Scheimpflug-accurate “normalized” coordinate where straight lines stay straight (before lens distortion).

> Compactly: define
>
> $$
> A_s =
> \begin{bmatrix}
> \mathbf{a}_s^\top\\
> \mathbf{b}_s^\top\\
> \frac{1}{d}\mathbf{n}_s^\top
> \end{bmatrix}
> \quad\Rightarrow\quad
> \tilde{u} \sim A_s X_c,\;\;
> m_x=\tilde{u}_1/\tilde{u}_3,\;\; m_y=\tilde{u}_2/\tilde{u}_3.
> $$

---

# 3) Distortion (done in the tilted plane)

Let the **principal ray intersection** with the plane be the distortion center. That is the image of $X_c=\mathbf{e}_z$:

$$
m_{0} =
\frac{d}{\mathbf{n}_s^\top \mathbf{e}_z}
\begin{bmatrix}
\mathbf{a}_s^\top \mathbf{e}_z \\
\mathbf{b}_s^\top \mathbf{e}_z
\end{bmatrix}.
$$

Work with **centered** plane coordinates $\bar{m} = (m_x,m_y)^\top - m_0$.
Apply your distortion model to $\bar{m}$ (e.g., Brown–Conrady):

$$
\bar{m}' = D(\bar{m};\,\kappa),
$$

then shift back: $m' = \bar{m}' + m_0$.

(If you prefer, you can carry the principal point only in pixel space; the above is the physically correct order—**projection to the tilted plane → distortion about the principal ray**—and avoids mixing a 2D homography with radial distortion.)

---

# 4) Pixel mapping

Finally,

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
\sim
K
\begin{bmatrix} m'_x \\ m'_y \\ 1 \end{bmatrix}.
$$

Putting (2)+(3)+(4) together gives a forward model

$$
u(X_w;\,\theta) = \Pi\!\big(K,\kappa,\tau_x,\tau_y, R_i,t_i \big),
$$

with global parameters $\theta = \{f_x,f_y,s,u_0,v_0,\kappa,\tau_x,\tau_y\}$ and per-view $\{R_i,t_i\}$.

---

# 5) How to calibrate (step-by-step)

**Data.** Prefer a **non-planar** target (points on multiple known $Z$ levels, a 3D grid, or a board you can “step” along depth). With only a planar checkerboard, Scheimpflug tilt is **not identifiable** on its own (see §6).

**Algorithm.**

1. **Initialization (standard):**

   * Run a conventional calibration **without tilt** (Zhang or multi-view PnP) to get $K$ (allow skew!), $\kappa$, and $\{R_i,t_i\}$.
   * Initialize $\tau_x=\tau_y=0$. Set $d=1$ and keep it fixed (absorbed into $f_x,f_y$).

2. **Define residuals.** For each observed target point $X_w^{(j)}$ in view $i$, residual

   $$
   r_{ij} =
   \begin{bmatrix}
   u_{ij}^{\text{meas}} - u(X_w^{(j)};\,\theta,R_i,t_i)\\
   v_{ij}^{\text{meas}} - v(X_w^{(j)};\,\theta,R_i,t_i)
   \end{bmatrix}.
   $$

3. **Nonlinear bundle adjustment.**

   * Optimize over $\theta$ and all $\{R_i,t_i\}$.
   * Use a robust loss (Huber/Cauchy).
   * Keep $d=1$ fixed to avoid the focal/plane-distance scale ambiguity.

4. **Refinements / constraints (optional):**

   * If you know the pixel axes are orthogonal, you can regularize $s\to 0$ but don’t hard-fix it initially.
   * If you have a mechanical datum for the sensor plane (e.g., tilt about a known axis), add a soft prior on $\tau_x,\tau_y$.

5. **Validation.**

   * Check straight lines in 3D project to straight lines in the image after undistortion.
   * Check reprojection on points with **varying depths**—that’s where tilt matters most.
   * If you calibrated with a planar board only, test on a small 3D stack; if errors grow with depth, tilt is not properly constrained.

---

# 6) Identifiability and the “planar trap”

With **only planar targets**, the mapping from the plane to the image is a homography:

$$
H_i = K\,[\mathbf{a}_s\;\; \mathbf{b}_s\;\; \mathbf{n}_s/d]\,[r_{1i}\;\;r_{2i}\;\;t_i],
$$

so a **global** constant rotation $R_s$ (your sensor tilt) can be absorbed into the **per-view** rotations $R_i' = R_s R_i$. Therefore, planar data **cannot separate** $(R_s,\{R_i\})$. In practice you will still get a good $K$ (with nonzero skew) and good reprojection, but $\tau_x,\tau_y$ are not uniquely determined.

**Conclusion:** if you care about the *physical* Scheimpflug tilt (e.g., relating optical axis to a robot/tool frame), include depth variation or additional priors.

---

# 7) Jacobians (for fast, stable optimization)

You can differentiate the metric mapping (before distortion) analytically. Let

$$
s = \mathbf{n}_s^\top X_c,\quad
\alpha = \mathbf{a}_s^\top X_c,\quad
\beta = \mathbf{b}_s^\top X_c,\quad
m =
\frac{d}{s}
\begin{bmatrix}\alpha\\ \beta\end{bmatrix}.
$$

* **w\.r.t. the 3D point** (useful if you optimize target pose):

$$
\frac{\partial m}{\partial X_c}
= \frac{d}{s}
\begin{bmatrix}
\mathbf{a}_s^\top - \frac{\alpha}{s}\mathbf{n}_s^\top\\
\mathbf{b}_s^\top - \frac{\beta}{s}\mathbf{n}_s^\top
\end{bmatrix}.
$$

* **w\.r.t. tilt angles $\tau\in\{\tau_x,\tau_y\}$** (use quotient rule):

$$
\frac{\partial m}{\partial \tau}
= \frac{d}{s}
\begin{bmatrix}
(\partial_\tau \mathbf{a}_s)^\top X_c\\
(\partial_\tau \mathbf{b}_s)^\top X_c
\end{bmatrix}
- \frac{d}{s^2}
\begin{bmatrix}\alpha\\ \beta\end{bmatrix}
(\partial_\tau \mathbf{n}_s)^\top X_c.
$$

Here $\partial_\tau \mathbf{a}_s = (\partial_\tau R_s)\mathbf{e}_x$, etc. For implementation, either code these from the SO(3) generators or let Ceres do local-parameterization Jacobians; both work.

Propagate through distortion $D(\cdot)$ and the pixel mapping $K$ in the usual way.

---

# 8) Small-angle view (intuition)

For small tilts $|\tau_x|,|\tau_y|\ll 1$, the oblique plane mainly introduces:

* a **principal point shift** (the principal ray no longer hits the chip center),
* a **nonzero skew**,
* slight **$f_x$ vs $f_y$** imbalance that varies with field height.

That’s why simple toolboxes that forbid skew often “fake” Scheimpflug with odd distortion coefficients. Modeling tilt explicitly stabilizes the fit and keeps distortion truly radial.

---

# 9) Practical tips

* If you can’t get a 3D target, you can still **use the model** and set $\tau_x=\tau_y=0$ (or leave them free but regularized) and let skew take the strain. Re-check on a few 3D points if possible.
* For **macro / large tilts**, stay with the central model if your lens behaves well; extreme optics can violate single-viewpoint—then you need a more general (non-central) model. Most tilt-shift work stays central enough to be captured by the above.
* For **line-scan** with Scheimpflug, the same ray–plane intersection applies per line; time adds the second image dimension (pushbroom). The per-line principal point drifts with motion; handle in the motion model, not in intrinsics.

---

# 10) Drop-in projection pseudocode

```cpp
// Inputs: Xw (3), Ri (3x3), ti (3), intrinsics fx, fy, s, u0, v0, distortion kappa,
// tilt angles tau_x, tau_y
// Output: (u,v)

Xc = Ri * Xw + ti;

// Tilt basis
Rs = Ry(tau_y) * Rx(tau_x);
as = Rs.col(0);  // a_s
bs = Rs.col(1);  // b_s
ns = Rs.col(2);  // n_s

// Ray-plane intersection (set d=1)
double sden = ns.dot(Xc);                 // n_s^T Xc
double mx   = as.dot(Xc) / sden;
double my   = bs.dot(Xc) / sden;

// Principal-point on the tilted plane (metric)
double s0 = ns(2);                        // n_s^T e_z
double mx0 = as(2) / s0;
double my0 = bs(2) / s0;

// Distortion in plane coords around (mx0,my0)
double dx = mx - mx0, dy = my - my0;
applyBrownConrady(dx, dy, kappa, &dx, &dy);
mx = dx + mx0;  my = dy + my0;

// Pixels
double u = fx*mx + s*my + u0;
double v =       fy*my + v0;
```

Bundle-adjust $\{fx,fy,s,u_0,v_0,\kappa,\tau_x,\tau_y, R_i,t_i\}$ on all observations.

---

If you like, I can sketch Ceres residual functors for this model (including minimal SO(3) parameterization and analytic Jacobians), or show how to retrofit it into OpenCV’s `calibrateCamera` workflow by replacing its projection step.
