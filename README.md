# Benchmark score prediction

We are given a set of language models `r_1, r_2, …`
that are scored on various benchmarks `b_1, b_2, …` with scores `s_{r_i, b_j}`.
But some of the `s_{r_i, b_j}` are unknown, and some are known.
The benchmarks have a high correlation for a given entity.
We want to predict one benchmark score `b_k` from the others,
with an easy-to-compute formula `\hat{s}_{r_i, b_k} = f(s_{r{\cdot}, b_{\cdot}})`.
We thus need to find the parameters of `f` that minimize
the mean squared error of this prediction,
`L = \Sigma_i \Sigma_{jk} (\hat{s}_{r_i, b_k} - s_{r_i, b_k})^2`.

The known model benchmark information is located in data/models.json.

We evaluate different approaches (described in the next section) below,
using the data snapshot 8a6870be:

| Predictor               | Iterations | Standard error | Latency   |
|-------------------------|------------|----------------|-----------|
| Weighed Bivariate reg.  |          1 | 0.6841         |    1.925s |
| Multiv. Gradient Desc.  |          1 | 0.6841         |    8.962s |
| Multiv. Gradient Desc.  |         10 | 0.6841         |   10.593s |
| Multiv. Gradient Desc.  |        100 | 0.6841         |   15.652s |
| Multiv. Reg. w/ Bivar.  |          1 | 0.6841         |    7.296s |
| Multiv. Reg. w/ Bivar.  |         10 | 0.6841         |   64.459s |
| Multiv. Reg. w/ Bivar.  |        100 | 0.6841         |  622.205s |
| Means                   |          1 | 0.9632         |    0.003s |

Here is another, slightly worse benchmark
using less model data (9a6257b1 snapshot), only 50 tests,
and benchmarks with a poorer random sampling, but still representative:

| Predictor               | Iterations | MSE   | Standard deviations | Latency |
|-------------------------|------------|-------|---------------------|---------|
| Cons. Bivariate reg.    |          1 | 3.031 | 1.741               |  0.222s |
| Cons. Biv. Grad. Desc.  |       1000 | 3.030 | 1.741               | 19.165s |
| Cons. Biv. Grad. Desc.  |      10000 | 3.024 | 1.739               |187.975s |
| Weighed Bivariate reg.  |          1 | 0.932 | 0.965               |  0.232s |
| Weighed Bivariate reg.  |         10 | 1.027 | 1.015               |  0.562s |
| Weighed Bivariate reg.  |        100 | 1.047 | 1.023               |  4.353s |
| Weighed Bivariate reg.  |       1000 | 1.047 | 1.023               | 37.695s |
| Weighed Bivariate reg.  |      10000 | 1.047 | 1.023               |346.601s |
| Means                   |          1 | 1.122 | 1.059               |  0.195s |
| Multivariate Regression |          1 | 1.046 | 1.023               |  0.272s |
| Multivariate Regression |         10 | 1.026 | 1.013               |  0.546s |
| Multivariate Regression |        100 | 0.899 | 0.948               |  2.602s |
| Multivariate Regression |       1000 | 0.886 | 0.941               | 21.932s |
| Multivariate Regression |      10000 | 0.874 | 0.935               |221.142s |
| Multiv. Reg. w/ Bivar.  |          1 | 0.939 | 0.969               |  0.233s |
| Multiv. Reg. w/ Bivar.  |         10 | 1.007 | 1.004               |  0.366s |
| Multiv. Reg. w/ Bivar.  |        100 | 0.921 | 0.960               |  1.756s |
| Multiv. Reg. w/ Bivar.  |       1000 | 0.801 | 0.895               | 17.181s |
| Multiv. Reg. w/ Bivar.  |      10000 | 0.810 | 0.900               |145.750s |
| Multiv. Gradient Desc.  |          1 | 1.377 | 1.173               |  0.240s |
| Multiv. Gradient Desc.  |         10 | 1.039 | 1.019               |  0.572s |
| Multiv. Gradient Desc.  |        100 | 1.038 | 1.019               |  0.641s |
| Multiv. Gradient Desc.  |       1000 | 1.035 | 1.018               |  0.864s |

## Approaches

### Normalizing

Each score is converted to (score - mean score) / std dev,
where the mean and standard deviation are over the benchmark.

### Consolidated Bivariate Regression

https://chat.deepseek.com/share/2wddmoutvj9v4z821y

### Weighed Bivariate Regression

We compute linear regression between each set of benchmarks:
a single benchmark predicts one other benchmark,
as sk = αk sj + βk.

The formula for a and b are:
α = cov(X, Y) / var(X)
β = mean(Y) - α * mean(X)

A more efficient formula is:
α[βj,βk] = Σ((s[βj] - mean[βj]) * (s[βk] - mean[βk])) / Σ((s[βj] - mean[βj])²)
β[βj,βk] = mean[βk] - a[βj,βk] * mean[βj]

From there, we can derive the full prediction as the average:
sk = (Σj αk sj + βk) ÷ n.

### Multivariate regression

Imputing the benchmark mean on the missing scores.
Then using multivariate regression to predict scores from that of all other
benchmarks with sk = αk + Σj βkj sj, where Bk = (αk, βkj) are benchmark-dependent.
For each benchmark, Bk = (X^T X)^-1 X^T Sk,
where X = [[1, s[r1;b1], s[r1;b2], …]; [1, s[r2;b1], s[r2;b2], …]].

The initial estimation of scores has an impact on the quality of the approach.
Instead of means, we can use the output of the weighed bivariate regression.

### Multivariate Gradient Descent

Initializes the predicted scores with the multivariate regression.
Then uses the multivariate regression formula as a loss:
L = Σi (si - (αk + Σj βkj sj))² where the sum is over known scores only.
The αk and βkj parameters are optimized using gradient descent.

### Latent Factor Model

A latent factor model would assign a vector embedding to each benchmark,
and another to each model. The score would be estimated as the dot product
of the model and benchmark embeddings, using the mean squared error
over the known scores as a loss function.

However, this seems equivalent to the multivariate gradient descent approach:
the [αk, ...βkj] form a latent vector on each benchmark,
and the [1, s1, s2, ...] form a latent vector on each model.

## Adding benchmarking data

Rules to add a benchmark:
- Only add benchmarks to `data/models.json`. Other `/data/` files are generated from it.
- All benchmarks must be sourced with links to the authoritative information,
  typically the official model announcement, or the official model benchmark.
- For a given source page, add all the benchmark scores listed on the page,
  except if they clearly are reusing numbers from a different source.
- Do not merge benchmark results. When a benchmark score appears in multiple sources with a different result,
  add both results as separate benchmarks, each referencing its source.
  They will be averaged in the estimation process.
- When a specific benchmark changes over time, such as LiveCodeBench or LMArena,
  keep it as if it was a continuous benchmark.
  Indeed, there is some intrinsic variance in all benchmark scores:
  running any twice will yield slightly different values,
  [even when they are run with a temperature of 0][determinism].
  We presume that the benchmark will maintain similar scores for a given set of weights over time.
- If a benchmark that can be run without tools, was ran with tools,
  label it as a separate benchmark, adding "(with tools)" to the benchmark name.
- The default benchmark for for Aider Polyglot should be diff (or diff-fenced for Gemini).
- If the model does not support vision, add a *MMMU* benchmark with a score of 25, with a source saying "No vision capabilities".
- if the model does not support tool calls, add a *τ²-Bench Telecom* benchmark with a score of 0, with a source saying "No tool capabilities".

Rules to add a model:
- Only add models to `data/models.json`. Other `/data/` files are generated from it.
- Keep an empty line between companies.
- Keep the models of a given company ordered from most recent to oldest.
- Each defined model should correspond to a unique set of weights and a reasoning setting.
  If a company publishes a new set of weights under the same name,
  add the date (typically as YYYY-MM) to the model name to differentiate it.
- The following fields are mandatory: `name`, `company`, `url` (of announcement), `release_date`.
- The following benchmarks are mandatory, placed at the top, and in this order: `Input cost`, `Output cost`.
  The website requires them to be present, and will break if they are not.
  Use the price of the official provider's API if it exists, or openrouter otherwise.
- If open-sourced, the following benchmarks are mandatory, at the top, and in this order: `Input cost`, `Output cost`.

[determinism]; https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
