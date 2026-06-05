"""Reproduce every numerical claim in the note. Run:  python experiments/reproduce.py"""
import numpy as np
from eds_stealing import (
    OneLayerModel, TwoLayerModel, recover_dimension, fit_quadric,
    polar_vs_tangent, recover_W, noise_sweep, rank_deficient_dimension,
    intrinsic_dimension,
)


def hr(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    m1 = OneLayerModel(seed=0)

    hr("Step 1  --  hidden dimension from the spectral gap (ideal, degree 1)")
    Z, _ = m1.query(3 * m1.h)
    h_hat, sv = recover_dimension(Z)
    print(f"true h = {m1.h}   recovered h_hat = {h_hat}")
    print(f"sigma[h-1] = {sv[m1.h-1]:.4e}   sigma[h] = {sv[m1.h]:.4e}")

    hr("Step 2  --  the quadric / ellipsoid (ideal, degree 2)")
    Z, _ = m1.query(3000)
    U, A, diag = fit_quadric(Z, m1.h)
    print(f"quadric residual max = {diag['quad_residual']:.2e}")
    print(f"A positive definite  = {diag['pd']}   "
          f"(eig in [{diag['eig_min']:.3e}, {diag['eig_max']:.3e}])")
    print(f"design rank          = {diag['design_rank']}/{diag['n_unknowns']}  (well posed)")

    hr("Step 3a --  polar space == tangent space of the output manifold")
    print(f"|cos(angle)| data-tangent vs recovered polar normal = "
          f"{polar_vs_tangent(m1, U, A):.2e}")

    hr("Step 3b --  recover W up to the O(h) gauge")
    for k, v in recover_W(Z, m1.W, m1.h).items():
        print(f"  {k:14s} = {v:.4e}")

    hr("Step 4a --  noise sweep: R1 robust (gap ~ 1/sigma), R2/R3 fragile (RMS ~ sigma)")
    print(f"{'sigma':>8} | {'h_hat':>5} | {'gap(h/h+1)':>11} | {'orth_RMS':>10}")
    for r in noise_sweep(m1, [0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]):
        print(f"{r['sigma']:>8.0e} | {r['h_hat']:>5d} | {r['gap']:>11.3e} | {r['orth_rms']:>10.3e}")

    hr("Step 4b --  rank deficiency recovers the EFFECTIVE rank (GPT-2-Small effect)")
    for r in rank_deficient_dimension(m1, [64, 50, 32]):
        print(f"  effective rank {r['effective_rank']:>3d}  ->  recovered h_hat = {r['h_hat']}")

    hr("Step 5  --  intrinsic dimension exposes a nonlinear sublayer the attack misses")
    m2 = TwoLayerModel(seed=1)
    Z2, _ = m2.query(4000)
    sp, _ = recover_dimension(Z2)
    idim = intrinsic_dimension(Z2, sp, rng=np.random.default_rng(1))
    Z1, _ = m1.query(4000)
    sp1, _ = recover_dimension(Z1)
    idim1 = intrinsic_dimension(Z1, sp1, rng=np.random.default_rng(1))
    print(f"one-layer : linear span = {sp1:2d}   intrinsic dim = {idim1:2d}   (sphere, span-1)")
    print(f"two-layer : linear span = {sp:2d}   intrinsic dim = {idim:2d}   "
          f"(content k = {m2.k}: a low-rank nonlinear bottleneck)")


if __name__ == "__main__":
    main()
