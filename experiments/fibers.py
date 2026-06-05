"""Demonstrate the explicit non-identifiability fibers below the last layer.
Run:  python experiments/fibers.py

Different sublayers -- and even different architecture widths -- produce
bit-identical outputs. The max logit differences below are all at machine
precision (~1e-14), i.e. the weights live in a free fiber of the observation map.
"""
from eds_stealing import TwoLayerModel, fiber_offsupport, fiber_width


def main():
    m = TwoLayerModel(h=64, k=8, m=32, seed=1)

    a = fiber_offsupport(m)
    print("(a) W1 acting off the input subspace col(B):")
    print(f"    perturbation norm ||Delta||_F   = {a['delta_norm']:.2f}")
    print(f"    invisible parameters            = {a['invisible_params']} "
          f"of {a['total_params']} in W1 "
          f"({100*a['invisible_params']/a['total_params']:.0f}%)")
    print(f"    max |logit difference|          = {a['max_logit_diff']:.2e}\n")

    b = fiber_width(m)
    print("(b) MLP width is not identifiable (cancelling neuron pair):")
    print(f"    width m = {b['m']} -> {b['m_new']} (distinct architecture)")
    print(f"    max |logit difference|          = {b['max_logit_diff']:.2e}\n")

    print("(c) B -> B@R for R in GL(k) leaves the attainable input set, and hence")
    print("    the entire output manifold, unchanged: B is identifiable only up to GL(k).")


if __name__ == "__main__":
    main()
