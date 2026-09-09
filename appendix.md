(app:mixture)=
# Appendix A: Marginalizing the outlier flags

This appendix works out the step in Chapter {ref}`7 <ch:advanced>` where the $N$ binary outlier flags $q_i$ of the mixture model are summed out of the likelihood. Hogg et al. (2010) describe the result in words ("marginalization is in fact analytic"); here are the lines in between.

## Setup

For each data point $i$ we have a flag $q_i \in \{0, 1\}$ ($1$ = good, $0$ = bad), a foreground density $p_{\rm fg}(y_i)$ (the straight line with Gaussian noise, Eq. {eq}`eq:fg` of Chapter 7) and a background density $p_{\rm bg}(y_i)$ (the broad Gaussian for outliers, Eq. {eq}`eq:bg`). As in Chapter 7, $\theta = \{a, b, P_{\rm bad}, \mu_{\rm bad}, \tau_{\rm bad}\}$ denotes the five continuous parameters (the two line parameters and the three hyperparameters of the outlier population); we suppress them where they are not needed. The two ingredients are:

* the likelihood given the flags, which picks the foreground density for good points and the background density for bad ones,

  $$
  p(y\,|\,\{q_i\}, \theta) = \prod_{i=1}^{N}\big[p_{\rm fg}(y_i)\big]^{q_i}\big[p_{\rm bg}(y_i)\big]^{1-q_i} ;
  $$ (eq:app-like)

* the prior on the flags, which says that every point is bad with probability $P_{\rm bad}$, independently of the others,

  $$
  p(\{q_i\}\,|\,P_{\rm bad}) = \prod_{i=1}^{N}(1-P_{\rm bad})^{q_i}P_{\rm bad}^{1-q_i} .
  $$ (eq:app-prior)

(Both expressions use the trick that $u^{q}v^{1-q}$ equals $u$ for $q=1$ and $v$ for $q=0$.)

## Step 1: what we want

We want the likelihood of the data given only the continuous parameters, $p(y\,|\,\theta)$, with the flags gone. By the law of total probability (Chapter 3, Eq. {eq}`eq:total-probability`), summing over all possible values of the flags, weighted by their prior probability:

$$
p(y\,|\,\theta) = \sum_{\{q_i\}} p(y\,|\,\{q_i\}, \theta)\,p(\{q_i\}\,|\,P_{\rm bad}) .
$$ (eq:app-total)

The sum runs over every combination of $N$ zeros and ones: $2^N$ terms. For $N = 20$ that is a million terms; for $N = 100$, more than the number of atoms in the observable Universe. This is the "exponential" model of Hogg et al., and if we had to evaluate it term by term we would be stuck.

## Step 2: the product of sums

Substitute Eqs. {eq}`eq:app-like` and {eq}`eq:app-prior` into Eq. {eq}`eq:app-total`. Both are products over $i$, so their product is too:

$$
p(y\,|\,\theta) = \sum_{q_1=0}^{1}\sum_{q_2=0}^{1}\cdots\sum_{q_N=0}^{1}\;\prod_{i=1}^{N} f_i(q_i),
\qquad\text{with}\qquad
f_i(q_i) \equiv \big[(1-P_{\rm bad})\,p_{\rm fg}(y_i)\big]^{q_i}\big[P_{\rm bad}\,p_{\rm bg}(y_i)\big]^{1-q_i} .
$$

Each factor $f_i$ depends on *its own* flag only. Now use the distributive law: a sum over all combinations of a product of independent factors is the product of the individual sums. For two points this is just

$$
\sum_{q_1}\sum_{q_2} f_1(q_1)\,f_2(q_2) = \sum_{q_1} f_1(q_1)\,\Big[\sum_{q_2} f_2(q_2)\Big] = \Big[\sum_{q_1} f_1(q_1)\Big]\Big[\sum_{q_2} f_2(q_2)\Big] ,
$$

because $\sum_{q_2} f_2(q_2)$ is a number that does not depend on $q_1$ and can be pulled out of the first sum; for $N$ points, repeat $N$ times. Hence

$$
p(y\,|\,\theta) = \prod_{i=1}^{N}\;\sum_{q_i=0}^{1} f_i(q_i) .
$$ (eq:app-factor)

The $2^N$-term sum has become $N$ sums of two terms each.

## Step 3: the two-term sum

For a single point the sum is trivial: $q_i = 1$ gives the first factor, $q_i = 0$ the second,

$$
\sum_{q_i=0}^{1} f_i(q_i) = f_i(1) + f_i(0) = (1-P_{\rm bad})\,p_{\rm fg}(y_i) + P_{\rm bad}\,p_{\rm bg}(y_i) .
$$

Putting this into Eq. {eq}`eq:app-factor`:

$$
p(y\,|\,\theta) = \prod_{i=1}^{N}\Big[(1-P_{\rm bad})\,p_{\rm fg}(y_i\,|\,a,b) + P_{\rm bad}\,p_{\rm bg}(y_i\,|\,\mu_{\rm bad},\tau_{\rm bad})\Big] ,
$$

which is Eq. {eq}`eq:mixture` of Chapter 7: the *mixture* likelihood. Each data point is drawn from the foreground distribution with probability $1-P_{\rm bad}$ and from the background distribution with probability $P_{\rm bad}$, and we never need to decide which.

Strictly speaking, $p(y\,|\,\theta)$ is the likelihood marginalized over the flags *with their prior*; the remaining prior on $\theta$ (flat in $a$, $b$, $P_{\rm bad}$, $\mu_{\rm bad}$, flat in $\ln \tau_{\rm bad}$) is multiplied in afterwards, as usual.

## Step 4: the posterior probability that a point is bad

The same two terms answer the question "is point $i$ an outlier?". Bayes' theorem applied to the single point $i$, with the two "hypotheses" $q_i = 0$ and $q_i = 1$, gives

$$
P(q_i = 0\,|\,y_i, \theta) = \frac{p(y_i\,|\,q_i=0,\theta)\,P(q_i=0\,|\,\theta)}{p(y_i\,|\,\theta)}
= \frac{P_{\rm bad}\,p_{\rm bg}(y_i)}{(1-P_{\rm bad})\,p_{\rm fg}(y_i) + P_{\rm bad}\,p_{\rm bg}(y_i)} ,
$$ (eq:app-outlier)

where the denominator is the law of total probability over the two values of $q_i$, i.e. exactly the two-term sum of Step 3. This is Eq. {eq}`eq:outlier-prob` of Chapter 7. It depends on the continuous parameters $\theta$; averaging it over the posterior samples of $\theta$ marginalizes them out, in the same way that Chapter 6 propagates any function of the parameters through the samples.

## Why this matters beyond outliers

The pattern "sum over discrete labels, factorize, get a mixture" appears whenever data come from several populations: two isotopes in a barrel (the practice-exam question in Chapter 7), stars and galaxies in a catalogue, signal and background events in a detector, or several Gaussian components in a velocity distribution. The mixture likelihood is always $\prod_i \sum_k w_k\,p_k(y_i)$ with weights $w_k$ that sum to one, and the posterior membership probabilities are always the terms of that sum divided by the sum.

## Source

Hogg, Bovy & Lang (2010), *Data analysis recipes: Fitting a model to data*, [arXiv:1008.4686](https://arxiv.org/abs/1008.4686), Section 3, Eqs. 13–17.
