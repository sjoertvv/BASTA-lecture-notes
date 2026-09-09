# Bayesian Statistics for Astrophysics (BASTA) — lecture notes

Source of the BASTA lecture notes (Leiden Observatory). Each chapter is a Jupyter notebook
(`0N_*.ipynb`) with its outputs stored; `intro.md` is the landing page.

## Editing

The notebooks are the source: open them in Jupyter, edit, run all cells, save.
(`src/*.md` contains the same chapters in MyST-markdown/jupytext form, which is convenient
for editing text in a text editor; `./build_nb.sh 03` regenerates and executes chapter 3 from
its `.md` file. If you edit the `.ipynb` directly, `jupytext --to myst 03_*.ipynb` updates the `.md`.)

## Building the website

    pip install -r requirements.txt
    jupyter-book build .

The site is written to `_build/html/` (a copy of the last build is in `website/`; open `website/index.html`). To publish on GitHub Pages:

    pip install ghp-import
    ghp-import -n -p -f _build/html

Read the Docs also works: `.readthedocs.yaml` is included (it runs `jupyter-book config sphinx .`
to generate the `conf.py` that RTD needs).

## Running the notebooks

    pip install numpy scipy matplotlib emcee corner jupyter

Random seeds are fixed, so all figures and numbers reproduce. Chapter 2 (bootstrap and the
300-dataset AIC/BIC simulation) and Chapter 6 (bootstrap of the ML fit) take about 20 s each;
the rest run in a few seconds.
