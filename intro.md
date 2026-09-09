# Bayesian Statistics for Astrophysics

*Lecture notes for BASTA (Leiden Observatory)*

These are the lecture notes of the course **Bayesian Statistics for Astrophysics** (BASTA), a 3 EC elective in the Leiden astronomy bachelor. They were first written by the class of 2024, one chapter per group, and have since been revised and extended by the teaching team. They follow the six lectures of the course, and they are meant to be read alongside them: the lectures introduce the ideas and the notes work them out, with derivations, runnable code and questions at the level of the exam.

## What this course is about

Almost every astronomer fits models to data. Some do so by minimizing $\chi^2$, perhaps even with a black-box routine, reading off the uncertainties it returns, and hoping for the best. This course is about replacing hope with understanding. The central idea, borrowed from the article by Hogg, Bovy & Lang (2010) that we follow closely, is the **generative model**: a quantitative description of how your data could have been produced, including the noise. Once you have that, everything else follows without arbitrary choices. The likelihood tells you how well the parameters explain the data; Bayes' theorem turns it into a probability distribution for the parameters; marginalization gets rid of the parameters you do not care about; and Markov chain Monte Carlo lets a computer do the integrals. Along the way you will see where the classical tools (hypothesis tests, least squares, bootstrap, information criteria) come from, when they work, and when they fail.


## How the notes are organized

| chapter | topic | lecture |
|---|---|---|
| {ref}`1 <ch:testing>` | Hypothesis testing, likelihood, least squares, confidence intervals | 1 (and 6) |
| {ref}`2 <ch:model-selection>` | Adding parameters: AIC and BIC; the bootstrap | 2, 3 |
| {ref}`3 <ch:foundations>` | Probability calculus, marginalization, Bayes' theorem | 2 |
| {ref}`4 <ch:inference>` | The three steps of Bayesian inference; grid posteriors; intrinsic scatter | 3, 4, 6 |
| {ref}`5 <ch:priors>` | Priors: strong, weak, flat, conjugate; shrinkage | 4 |
| {ref}`6 <ch:mcmc>` | Markov chain Monte Carlo: Metropolis–Hastings and `emcee` | 5, 6 |
| {ref}`7 <ch:advanced>` | Bayesian billiards; outliers and mixture models | 6 |

Chapters 1 and 2 are the "frequentist warm-up" and include tools you have already used in your studies. Chapter 3 is the mathematical foundation, kept relatively short. Chapters 4 to 6 are the core of the course. Chapter 7 contains a bit more advanced material.

Each chapter starts with a list of **goals**, contains **runnable code** that you are encouraged to modify, and ends with **Test yourself** questions: pen-and-paper problems at the level of the written exam. Three datasets run through the notes:

* **The Multiverse**: simulated straight-line data, so that we know the truth and can check every method against it;
* **Hogg's Table 1**: the 20 data points of Hogg et al. (2010), with real outliers, used for the worked line fits;
* **Blinky**: 100 exposures, in photon counts, of a star that may or may not be variable; we test it for variability in Chapter 1, model its intrinsic scatter in Chapter 4, and sample the posterior in Chapter 6.

## Relation to the rest of the course

The course has three components: the lectures and werkcolleges, a **hand-in assignment** (a full Bayesian analysis of the light curve of a real tidal disruption event, 50% of the grade), and a **written exam** (50%). These notes support both. The methods in the assignment (least squares with and without $\chi^2$ rescaling, bootstrap, MCMC with an extra variance parameter, model comparison, mixture models) are all introduced here, on different data, so that you can adapt the code rather than copy it. The exam is pen-and-paper: writing down likelihoods and posteriors, deriving simple results, sketching distributions, and explaining what a piece of MCMC code does or does not do. The "Test yourself" questions and the werkcollege exercises are the best preparation; the full exams of previous years are also available (on Brightspace).

## Notation

We try to use one set of symbols throughout. Different books use different conventions, and the box on notation in Chapter 3 lists the most common alternatives.

| symbol | meaning |
|---|---|
| $y = \{y_1, \dots, y_N\}$ | the data; $N$ data points |
| $x_i$ | the independent variable of data point $i$ (time, position); assumed to have no uncertainty |
| $\sigma_{y,i}$ | the (Gaussian) uncertainty of data point $y_i$; often shortened to $\sigma_i$ |
| $\theta = \{\theta_1,\dots,\theta_k\}$ | the model parameters; $k$ of them |
| $m(x, \theta)$ | the model prediction at $x$ |
| $\theta^*$, $\hat\theta$ | the true parameter values; an estimate of them (e.g. the maximum-likelihood estimate) |
| $\mathcal{N}(\mu, \sigma^2)$ | the normal distribution with mean $\mu$ and variance $\sigma^2$; $y\sim\mathcal{N}(\mu,\sigma^2)$ reads "$y$ is drawn from" |
| $P(A)$ | the probability of an event $A$ (Chapter 3) |
| $p(y \mid \theta)$ | a probability density of $y$ given $\theta$; as a function of $\theta$ this is the *likelihood* $\mathcal{L}(\theta)$ |
| $p(\theta)$, $p(\theta \mid y)$ | the prior and the posterior probability density of the parameters |
| $I$ | background information (the $x_i$, the $\sigma_{y,i}$, the choice of model); usually left implicit |
| $\chi^2$, dof | the weighted sum of squared residuals; the number of degrees of freedom, $N-k$ |
| $\epsilon$ | the intrinsic scatter (an extra standard deviation added in quadrature to $\sigma_{y,i}$) |
| $\mu_0, \tau_0$; $\alpha, \beta$ | hyperparameters: the parameters of a prior (Chapter 5) |

## Software

The code in these notes uses Python 3 with `numpy`, `scipy` and `matplotlib`, plus [`emcee`](https://emcee.readthedocs.io) for MCMC and [`corner`](https://corner.readthedocs.io) for corner plots (Chapters 6 and 7). Every chapter is a Jupyter notebook that can be downloaded from the page (the download button at the top) and run from start to finish; the random seeds are fixed, so you should reproduce the figures exactly. Do change the seeds and the numbers: watch what happens when $N=3$ becomes $N=30$, or when the prior is made narrower.

## Literature

* **Hogg, Bovy & Lang (2010)**, *Data analysis recipes: Fitting a model to data*, [arXiv:1008.4686](https://arxiv.org/abs/1008.4686). Mandatory reading. If you understand this article you are in good shape for the exam.
* **Gelman, Carlin, Stern, Dunson, Vehtari & Rubin (2013)**, *Bayesian Data Analysis*, 3rd edition ([free PDF](https://sites.stat.columbia.edu/gelman/book/BDA3.pdf)), Chapters 1–3. The standard reference; the kidney-cancer example of Chapter 5 is in Section 2.7.
* **Hogg & Foreman-Mackey (2018)**, *Data analysis recipes: Using Markov Chain Monte Carlo*, [arXiv:1710.06068](https://arxiv.org/abs/1710.06068).
* **Sivia & Skilling (2006)**, *Data Analysis: A Bayesian Tutorial*, Oxford University Press. Short and readable.
* **Wasserman (2004)**, *All of Statistics*, Springer ([publisher page](https://link.springer.com/book/10.1007/978-0-387-21736-9)). A compact reference for the frequentist side: axioms, estimators, tests, bootstrap.
* **VanderPlas (2014)**, *Frequentism and Bayesianism*, a [series of blog posts](http://jakevdp.github.io/blog/2014/03/11/frequentism-and-bayesianism-a-practical-intro/) from which several examples in these notes descend.
* **Uttley**, [*Statistical methods for astrophysics*](https://philuttley.github.io/stats-methods-24/), lecture notes of the sister course in Amsterdam, for a second look at many of the same topics.

## Feedback

These notes are a living document. Typos, unclear passages, better examples: open an issue or a pull request on the GitHub repository, or send an email. 

*Sjoert van Velzen, with the BASTA teaching assistants and the class of 2024.*
