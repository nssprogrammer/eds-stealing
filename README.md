# The Geometry of Last-Layer Model Stealing

A small, fully reproducible companion to the note of the same name. It gives a
geometric / exterior-differential-systems (EDS) reading of the last-layer
model-stealing attack of Carlini et al. (2024, *Stealing Part of a Production
Language Model*, arXiv:2403.06634), and an explicit identifiability analysis of
the layer *beneath* the unembedding.

Everything runs on a small, controlled toy "production" model so the ground
truth is known and each claim can be checked line by line in pure NumPy.

## Scope

This is an **expository synthesis plus a concrete identifiability boundary**, not
a new attack. The last-layer recovery (including recovery up to an orthogonal
matrix) is reproduced from Carlini et al.; the EDS framing *organizes* it but is
not the engine; and the identifiability results rest on classical
neural-network identifiability (Sussmann 1992) and on the known low intrinsic
dimension of learned representations (Ansuini et al. 2019). What is new but
modest: the intrinsic-dimension *observable* in the extraction setting, and the
explicit non-identifiable fibers below the last layer.

## What it shows

| step | claim | headline number (toy, `h=64`) |
|------|-------|-------------------------------|
| 1 | hidden dim from the spectral gap (degree-1 part of the ideal) | `h_hat = 64`, 14-order cliff |
| 2 | the normalization quadric (degree-2 part), well posed | residual `2e-14`, design rank `2080/2080` |
| 3a | polar space = tangent space of the output manifold | `cos = 5.8e-7` |
| 3b | recover `W` up to the `O(h)` gauge | orthogonal RMS `6e-16` = affine RMS; `‖Ω−I‖≈√(2h)` |
| 4a | R1 robust (`gap~1/σ`), R2/R3 fragile (`RMS~0.036σ`) | see noise table |
| 4b | rank deficiency → effective rank (GPT-2-Small effect) | `50→50`, `32→32` |
| 5 | intrinsic dim exposes a nonlinear bottleneck | two-layer: span `40`, intrinsic `8` |
| 6 | explicit non-identifiable fibers below the last layer | 87.5% of `W1` invisible; width `32→34` identical |

## Layout

```
eds-stealing/
├── eds_stealing/          # the library
│   ├── __init__.py
│   └── core.py            # models, attack, polar check, regularity, manifold, fibers
├── experiments/
│   ├── reproduce.py       # runs steps 1–5 and prints the tables
│   ├── make_figures.py    # regenerates the paper figures
│   └── fibers.py          # demonstrates the identifiability fibers (step 6)
├── paper/
│   ├── note.tex           # the arXiv source (self-contained)
│   ├── note.pdf           # compiled
│   └── figures/           # spectrum.pdf, main.pdf
├── requirements.txt
├── LICENSE
└── README.md
```

## Install & run

```bash
git clone https://github.com/nssprogrammer/eds-stealing && cd eds-stealing
pip install -r requirements.txt          # numpy, matplotlib

# reproduce every number in the note
PYTHONPATH=. python experiments/reproduce.py

# the identifiability fibers (machine-precision identical outputs)
PYTHONPATH=. python experiments/fibers.py

# regenerate the figures
PYTHONPATH=. python experiments/make_figures.py

# build the paper
cd paper && pdflatex note.tex && pdflatex note.tex
```

The library is tiny and importable:

```python
from eds_stealing import OneLayerModel, recover_dimension, recover_W

m = OneLayerModel(h=64, seed=0)
Z, _ = m.query(3000)
print(recover_dimension(Z)[0])          # 64
print(recover_W(Z, m.W, m.h))           # orthogonal RMS ~1e-16
```

## Credits

- Carlini et al., *Stealing Part of a Production Language Model*, ICML 2024 (arXiv:2403.06634) — the attack this builds on.
- Hohloch, Mestdag, Yasaka, *The Cartan–Kähler theorem for EDS on transitive Lie algebroids*, arXiv:2605.29083 — the EDS vocabulary.
- Bryant, Chern, Gardner, Goldschmidt, Griffiths, *Exterior Differential Systems*, Springer 1991.
- Sussmann, *Uniqueness of the weights for minimal feedforward nets…*, Neural Networks 1992.
- Ansuini, Laio, Macke, Zoccolan, *Intrinsic dimension of data representations in deep neural networks*, NeurIPS 2019 (arXiv:1905.12784).

## License

MIT — see `LICENSE`.
