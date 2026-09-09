# What changed with respect to the 2025 notes

Chapter order is unchanged (it follows the lectures). Every chapter was rewritten for
consistent notation (see the table in `intro.md`), explicit cross-references between
chapters, shorter examples, and a "Test yourself" section with exam-level questions and
hidden answers. The example notebooks/scripts from the BASTA folder are woven in as
indicated. Numbers below refer to the new chapters.

## Introduction (new)
Purpose of the notes, chapter map with the lecture each chapter belongs to, notation
table, the three running datasets (simulated "multiverse", Hogg et al. Table 1, photon
counts), software, reading list, relation to assignment and exam.

## 1. Hypothesis testing and maximum likelihood
* Hypothesis testing rewritten around the three ingredients (H0, TS, p-value), with the
  lecture's three quiz questions, the sigma/p-value table and the "don't overdo it" advice.
* The astropy Doppler example is replaced by the **week-6 photon-counts example**
  (`BASTA_week6_example.py`): MC p-value from the Poisson likelihood vs. chi2 p-value.
  The same dataset (fixed seed) returns in Chapters 4 and 6.
* Likelihood / generative model section follows Hogg §2; the students' grid-search over
  (a, b) was moved out (it is Chapter 4 material) to remove overlap.
* Least squares = maximum likelihood, chi2/dof and goodness of fit; worked fit to **Hogg
  Table 1, points 5-20** (reproduces Hogg Fig. 1: 2.24 +/- 0.11).
* Confidence-interval section replaced by **multiverse Part 1** (`absolute_sigma`, N=3 vs
  N=10; the old Example 1.6 was not a parameter confidence interval and is gone).
* Proof box "ML = least squares" (from the board, lecture 1).
* Test yourself: exam 2025 Q2.1-2.3 (Taylor), exam-2023 dark-matter counts, practice-exam
  "three ways to get an uncertainty". (The inverse-variance weighted mean is exercise 4 of
  the tutorial and is therefore *not* worked out in the notes.)

## 2. Model selection and the bootstrap
* Nested models, residual checks, the lecture-3 "uncertainties too large" puzzle, the
  two chi2 quiz questions.
* AIC/BIC: definitions, relative likelihood, AIC-vs-BIC (Wasserman vs Gelman quotes,
  Kass & Raftery thresholds). Worked example is **multiverse Part 3** (polynomial degree,
  chi2, chi2/dof, AIC/BIC, 300-dataset histogram). The students' version had bugs
  (`test` undefined, `np.min(L)` used as the *maximum* likelihood) and is replaced.
* Bootstrap: percentile intervals (the old version used mean +/- z*std), **multiverse
  Part 2** (linear model, non-linear exponential model, "who is right" Monte Carlo),
  plus a bootstrap of the Hogg slope.
* Summary table of the three (four) ways to get an uncertainty.
* Test yourself: exam-2023 Q1 (weather company), bootstrap-or-not.

## 3. Mathematical foundations
* Venn diagrams are now generated in code (the five PNGs are no longer needed).
* Addition rule, product rule, independence -> likelihood (the lecture's "connection to
  likelihood" derivation), law of total probability -> marginalization and nuisance
  parameters (distance-modulus example kept, shortened).
* Bayes' theorem derived from the product rule; rare-disease example moved here from
  Chapter 4 (replacing "Bob the astronomy student"); posterior/likelihood/prior/evidence
  named; notation box comparing BASTA / Gelman / Wasserman / Hogg and explaining where x, sigma_y, I go.
* Random variables; frequentist vs Bayesian with the Laplace/Saturn history (moved here
  from the students' Chapter 5 to remove overlap). Galaxy-with-life integral removed.
* Kolmogorov axioms, proof of the addition rule (dropdown), derivation box for Bayes'
  theorem (from the board), remark on conditioning (preparation for Monty Hall).
* Test yourself: exam 2025 Q1.1 and Q1.3, plus a P(A|B)-vs-P(B|A) question. (The
  two-event marginalization proof is exercise 3 of the tutorial and is not in the notes.)

## 4. Bayesian inference
* Three steps with the x / sigma_y notation of lecture 3.
* One-parameter Poisson-rate example (exam-2023 Q3): analytic posterior + grid; credible
  intervals (equal-tailed and highest-density) with reusable functions. NB: the exam-2023
  solution sheet writes the normalized posterior as 10 theta^2 e^{-10 theta}; the correct
  normalization is 500 theta^2 e^{-10 theta} (evidence = 0.1).
* Photon counts: posterior of the mean with a Poisson likelihood.
* **Intrinsic scatter** section (lecture 4 part 1 + lecture 6): derivation of the
  log-likelihood with sigma^2 + eps^2, why curve_fit cannot do it, connection to chi2
  rescaling and to the p-value; ML fit, AIC/BIC change, likelihood-ratio remark; **2-D
  grid posterior of (mu, eps)** with marginals and 68/95% contours (replaces the slow
  2000x2000 Sersic example).
* Posterior updating with new data (kept, shortened, with code).
* Reporting results; credible vs confidence interval table (lecture 6).
* Derivation box for the intrinsic-scatter log-likelihood (from the board, lecture 4).
* Test yourself: exam 2025 Q2.4-2.6, exam-2023 Q3.1.

## 5. Priors
* Taxonomy condensed; flat-in-log for scale parameters and improper priors explained.
* Beta-Binomial coin example kept (prior fixed to Beta(10,12)); exoplanet log-normal
  example removed (the "likelihood" there was arbitrary).
* **Conjugate priors done properly**: Gaussian-Gaussian derivation (precision-weighted
  mean; "the prior acts like extra data", lecture 4 slide 13) with a figure for N = 1,
  10, 100; Poisson-Gamma stated (derivation left to exercise 5). The students'
  "Poisson x Poisson" example (which multiplied two PMFs in the data variable) is removed.
* **Kidney-cancer shrinkage** illustration (lecture 3, BDA 2.7) with simulated counties;
  the numerical part of exercise 5 is deliberately left to the werkcollege.
* Priors in practice: physical bounds, other experiments' posteriors, prior sensitivity.
* Derivation box for the Gaussian-Gaussian posterior (completing the square, from the
  board, lecture 4); warning box on rate vs scale parametrization of the Gamma distribution.
* Test yourself: exam-2023 Q3.4, practice exam Q2.4, "how much is a prior worth". (The
  Poisson-Gamma derivation is part of exercise 5; only the general result is stated.)

## 6. MCMC
* Why sampling; Markov chains; Metropolis-Hastings algorithm exactly as in lecture 5.
* **Speedrun** (y=[1,2,3]) in ~15 lines; proposal-width tuning; autocorrelation and
  integrated autocorrelation time.
* **"Find the bug" box** with the exam-2025 code (missing factor 1/2 in the log-likelihood).
* Affine-invariant stretch move and emcee usage; **worked emcee example** on the (mu, eps)
  model with trace plots, autocorrelation time, corner plot, credible intervals, error
  propagation; **bootstrap vs MCMC** comparison (lecture 6 slides 13/19, from
  `BASTA_week6_example.py`). MCMC checklist.
* The students' affine-invariance section (with the two PNG figures) is condensed into
  the emcee section.
* Test yourself: practice exam Q3, exam-2023 Q4.

## 7. Advanced examples
* **Bayesian billiards** with the Beta-function solution and the vectorized Monte Carlo
  of `BayesBiljard_solutions.py` (incl. the remark that p must be redrawn per game);
  table for all scores 5-0 ... 5-4.
* Sigma clipping cut to a 10-line function and a paragraph of critique.
* **Hogg mixture model** on the full Table 1: the likelihood bug of the 2025 version
  (random assignment of points inside the likelihood) is fixed; flat-in-ln V_b prior;
  rogue-walker removal; posterior outlier probability per point (Eq. 17-18 of Hogg);
  posterior-draw plot. Points 2-4 are flagged, point 1 sits in the grey zone.
* Notation matched to the rest of the notes: the billiards parameter is theta (densities p,
  events P), the line is a + b x with theta = (a, b), and Hogg's P_b, Y_b, V_b are written
  P_bad, mu_bad, tau_bad (mean and width of the outlier population; the mapping to
  Hogg's symbols is stated once) and are introduced as hyperparameters, linking to a new
  definition box in Chapter 5.
* Test yourself: practice exam Q1 (radioactive mixture), plug-in trap.

## Appendix A (new)
Full derivation of the mixture likelihood (summing out the 2^N outlier flags) and of the
per-point outlier probability; referenced from Chapter 7.

## References and names
* All references to the STAN syllabus removed; Wasserman (2004) and Gelman et al. (2013)
  cited instead. URLs (arXiv / DOI / free PDF) added to every reference list.
* The photon-counts dataset is now the star "Blinky".

## Build
* Long two-part equations split over several lines; `_static/custom.css` lets any remaining
  wide equation scroll horizontally instead of overflowing.
* The built website is in `website/` (open `website/index.html`); rebuild with `jupyter-book build .`.
Jupyter Book 1.x (`_config.yml`, `_toc.yml`), no Colab/Binder buttons, download button on.
Notebooks are stored with outputs and are not re-executed at build time.
`src/*.md` are the jupytext sources; `build_nb.sh NN` regenerates and executes chapter NN.
