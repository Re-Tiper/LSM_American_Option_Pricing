# Least Squares Monte Carlo (LSM) Algorithm for Pricing American Options
---

We present the **LSM (Least Squares Monte Carlo)** algorithm \[LSM 1, LSM 2\] for pricing American options using **Monte Carlo simulations** and provide its implementation in Python.

## Value of American Options

The value of an American option can be expressed as:

$$
V(t) = \sup_{t \leq \tau \leq T} \mathbb{E}_{\widehat{\mathbb{P}}}\left[e^{-r(\tau-t)} G(\tau) \mid S(t)\right]
$$

where $\tau$ is the exercise time. The calculation corresponds to finding the optimal exercise time $\tau$, i.e., the moment when the option should be exercised. Once $\tau$ is determined, we compute the expected value:

$$
V(t) = \mathbb{E}_{\widehat{\mathbb{P}}}\left[e^{-r(\tau-t)} G(\tau) \mid S(t)\right]
$$

The present value of the option is then:

$$
V(0) = \sup_{0 \leq \tau \leq T} \mathbb{E}_{\widehat{\mathbb{P}}}\left[e^{-r\tau} G(\tau) \mid S(0)\right]
$$

## Steps of the Longstaff-Schwartz Method (LSM)

The LSM algorithm for pricing American options consists of the following steps:

1. **Path Simulation**:  
   Simulate the price of the underlying asset \( M \) times over \( n \) equally spaced time steps, according to the geometric Brownian motion (or another stochastic model), using the Monte Carlo method. This generates paths reflecting potential asset prices.

2. **Dynamic Programming**:  
   After simulating the paths, the algorithm works backward from the option's expiration date \( T \) to time \( 0 \). At each time step:
   - Compute the payoff if the option is exercised at that time.
   - Estimate the continuation value using least squares regression, which approximates the expected payoff if the holder continues to hold the option.

3. **Regression**:  
   The core of the method is the use of least squares regression to estimate the continuation value at each time step:

   $$
   V(t_i) = \mathbb{E}_{\widehat{\mathbb{P}}}\left[e^{-r(\tau' - t_i)} G(\tau') \mid S(t_i)\right]
   $$

   Here, \( \tau' \) is the optimal exercise time in \( \{t_{i+1}, t_{i+2}, \dots, t_n = T\} \subseteq [0, T] \). The regression uses a set of basis functions, often polynomials, to approximate these expected values, enabling the decision of whether to exercise the option or continue holding it.

4. **Exercise Decision**:  
   At each time step, for every simulated path, compare the immediate exercise value with the estimated continuation value. If the immediate exercise value is higher, exercise the option, and the optimal stopping time becomes that time step. Otherwise, do not exercise. Repeat this process backward to determine the optimal stopping time.

5. **Option Value Calculation**:  
   After determining the optimal stopping time for all paths, discount the payoffs back to present value using the risk-free rate. The average of these discounted payoffs is the estimated option value.

## Basis Functions

We will use **Laguerre polynomials** as the basis functions for estimating the expected value \( V(t_i) \). Laguerre polynomials form an orthogonal basis of the space \( L^2([0,\infty]) \) with respect to the inner product:

$$
\langle f, g \rangle = \int_{0}^{\infty} f(x) g(x) e^{-x} \, dx
$$

The general form of Laguerre polynomials is:

$$
L_n(x) = \sum_{k=0}^{n} \binom{n}{k} \frac{(-x)^k}{k!}
$$

We will use the first 5 such polynomials:

$$
\begin{aligned}
L_0(x) &= 1 \\
L_1(x) &= 1 - x \\
L_2(x) &= \frac{1}{2}(2 - 4x + x^2) \\
L_3(x) &= \frac{1}{6}(6 - 18x + 9x^2 - x^3) \\
L_4(x) &= \frac{1}{24}(24 - 96x + 72x^2 - 16x^3 + x^4)
\end{aligned}
$$

The expected value \( V(t_i) \) can be expressed as a linear combination of these basis functions:

$$
\widehat{V}(S_l(t_i)) = \sum_{j=0}^{k} \beta_j L_j(S_l(t_i))
$$

Here, \( \beta_j \) are the regression coefficients obtained using least squares.### Least Squares Monte Carlo (LSM) Algorithm for Pricing American Options

We will present the **LSM (Least Squares Monte Carlo)** algorithm \cite{LSM 1}, \cite{LSM 2} for pricing American options using **Monte Carlo simulations** and provide its implementation in Python.

#### Value of American Options
The value of an American option can be expressed as:

\[
V(t) = sup {t ≤ τ ≤ T} E_[P̂] [exp(-r(τ-t)) × G(τ) | S(t)]
\]

where \(\tau\) is the exercise time. The calculation corresponds to finding the optimal exercise time \(\tau\), i.e., the moment when the option should be exercised. Once \(\tau\) is determined, we compute the expected value:

\[
V(t) = \mathbb{E}_{\widehat{\mathbb{P}}}[e^{-r(\tau-t)}G(\tau)|S(t)]
\]

The present value of the option is then:

\[
V(0) = \sup_{0 \leq \tau \leq T} \mathbb{E}_{\widehat{\mathbb{P}}}[e^{-r\tau}G(\tau)|S(0)]
\]

#### Steps of the Longstaff-Schwartz Method (LSM)
The LSM algorithm for pricing American options consists of the following steps:

1. **Path Simulation**:
   - Simulate the price of the underlying asset \(M\) times over \(n\) equally spaced time steps, according to the geometric Brownian motion (or another stochastic model), using the Monte Carlo method. This generates paths reflecting potential asset prices.

2. **Dynamic Programming**:
   - After simulating the paths, the algorithm works backward from the option's expiration date \(T\) to time \(0\). At each time step:
     1. Compute the payoff if the option is exercised at that time.
     2. Estimate the continuation value using least squares regression, which approximates the expected payoff if the holder continues to hold the option.

3. **Regression**:
   - The core of the method is the use of least squares regression to estimate the continuation value at each time step:

\[
V(t_i) = \mathbb{E}_{\widehat{\mathbb{P}}}[e^{-r(\tau'-t_i)}G(\tau')|S(t_i)] \tag{3.5.1}\label{ExpValLSM}
\]

   Here, \(\tau'\) is the optimal exercise time in \(\{t_{i+1}, t_{i+2}, \dots, t_n=T\} \subset [0,T]\). The regression uses a set of basis functions, often polynomials, to approximate these expected values, enabling the decision of whether to exercise the option or continue holding it.

4. **Exercise Decision**:
   - At each time step, for every simulated path, compare the immediate exercise value with the estimated continuation value. If the immediate exercise value is higher, exercise the option, and the optimal stopping time becomes that time step. Otherwise, do not exercise. Repeat this process backward to determine the optimal stopping time.

5. **Option Value Calculation**:
   - After determining the optimal stopping time for all paths, discount the payoffs back to present value using the risk-free rate. The average of these discounted payoffs is the estimated option value.

#### Basis Functions
We will use **Laguerre polynomials** as the basis functions for estimating the expected value \(V(t_i)\) (Equation \ref{ExpValLSM}). Laguerre polynomials form an orthogonal basis of the space \(L^2([0,\infty])\) with respect to the inner product:

\[
\langle f, g \rangle \coloneqq \int_{0}^{\infty} f(x)g(x)e^{-x} \, dx
\]

The general form of Laguerre polynomials is:

\[
L_n(x) = \sum_{k=0}^{n}\binom{n}{k}\frac{(-x)^k}{k!}
\]

We will use the first 5 such polynomials:

\[
\begin{aligned}
L_0(x) &= 1 \\
L_1(x) &= 1-x \\
L_2(x) &= \frac{1}{2}(2 - 4x + x^2) \\
L_3(x) &= \frac{1}{6}(6 - 18x + 9x^2 - x^3) \\
L_4(x) &= \frac{1}{24}(24 - 96x + 72x^2 - 16x^3 + x^4)
\end{aligned}
\]

Since the expected value \(V(t_i)\) belongs to \(L^2([0,\infty])\), it can be expressed as a linear combination of basis functions. Let \(S(t_i) = (S_1(t_i), \dots, S_M(t_i))\) represent the paths that are **in-the-money** at time \(t_i\), i.e., \(G(t_i) > 0\). If a path is **out-of-the-money**, its payoff is 0, and exercising the option is not beneficial. The expected value for the \(l\)-th path is approximated by:

\[
\widehat{V}(S_l(t_i)) = \sum_{j=0}^{k}\hat{\beta}_j L_j(S_l(t_i))
\]

where \(\hat{\beta}_j\), \(j = 0, \dots, k\), are the regression coefficients obtained using least squares. Specifically:

\[
Y = A \cdot \hat{\beta}
\]

where:

- \(Y = (y_1, y_2, \dots, y_M)^T\) is the vector of future discounted payoffs (continuation values) with \(y_l = e^{-r(\tau'-t_i)}G_l(\tau')\) for \(l = 1, \dots, M\), where \(\tau'\) is the optimal exercise time in \(\{t_{i+1}, t_{i+2}, \dots, t_n=T\}\) and \(G_l\) is the payoff of the \(l\)-th path.
- \(\hat{\beta} = (\hat{\beta}_0, \hat{\beta}_1, \dots, \hat{\beta}_k)^T\) are the regression coefficients.
- \(A_{M \times k}\) is the matrix of basis functions, where \(A_{lj} = L_j(S_l(t_i))\), \(l = 1, \dots, M\), \(j = 0, 1, \dots, k\) (e.g., for a 2nd-degree polynomial, \(A = [1, 1-X, \frac{1}{2}(2 - 4X + X^2)]\)).

The least squares solution for \(\hat{\beta}\) is:

\[
\hat{\beta} = (A^{T}A)^{-1}A^{T}Y
\]

It can be shown that:

\[
\lim_{k \to \infty}\widehat{V}(S_l(t_i)) = V(S_l(t_i)) \quad \forall l \in \{1, \dots, M\}
\]

but without knowing the rate of convergence.
