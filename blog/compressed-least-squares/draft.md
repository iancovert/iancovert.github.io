# Averaged compressed least squares: column subsampling for linear models

Flexible ML models tend to overfit, and one way to mitigate that is *column subsampling*, or forcing the model to use only a subset of features. This approach is common in tree ensemble models, like [XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html) and [random forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) for example.$^1$

Why does column subsampling work? Intuitively, it's because a model trained with fewer features is less likely to overfit. On the other hand, it operates with less information, so it may be unable to express the correct model; this should increase the error, and that's why random forests average predictions across many models trained with different feature subsets.

I wondered if it was possible to justify column subsampling more precisely by analyzing a simpler model class, and it turns out it is. There's a technique called *averaged compressed least squares* (ACLS) [1] that's basically a linear analogue to random forests, and I'll share a couple basic results about it here.

Here's how the post is organized:

1. Ordinary least squares (OLS)
2. Compressed least squares (CLS)
3. Error analysis: bias-variance decomposition
4. Averaged compressed least squares (ACLS)

Let's jump in.

<!-- Why does column subsampling work? To think about why doing so could improve a model's out-of-sample perforance, let's consider the standard bias-variance decomposition. 

[Need to explain this, including the notation.] -->

<!-- $$\mathbb{E}_{\mathcal{D}, \epsilon}[(f(x; \mathcal{D}) - y)^2] = \mathbb{E}_\mathcal{D}[f(x; \mathcal{D} - \mathbb{E}_\mathcal{D}[f(x; \mathcal{D})])^2] + (\mathbb{E}_\mathcal{D}[f(x; \mathcal{D})] - y)^2 + \sigma^2$$ -->

<!-- $$\mathbb{E}_{\mathcal{D}, \epsilon}[(\hat{f}(x; \mathcal{D}) - y)^2] = (\text{Bias}_\mathcal{D}[\hat{f}(x; \mathcal{D})] )^2 + \text{Var}_\mathcal{D}[\hat{f}(x; \mathcal{D})] + \sigma^2$$

$$\text{Bias}_\mathcal{D}[\hat{f}(x; \mathcal{D})] = \mathbb{E}_\mathcal{D}[\hat{f}(x; \mathcal{D})] - f(x)$$

$$\text{Var}(\hat{f}(x; \mathcal{D})) = \mathbb{E}_\mathcal{D}[(\hat{f}(x; \mathcal{D}) - \mathbb{E}_\mathcal{D}[\hat{f}(x; \mathcal{D})])^2]$$

When we train using a smaller number of features, two things should happen. First, the relative lack of flexibility means that our model may be unable to expression the true function $f(x)$, so the bias should increase; this should increase the error. Second,  -->

## Ordinary least squares (OLS)

Assume a data matrix $X \in \mathbb{R}^{n \times d}$ and label vector $Y \in \mathbb{R}^n$ generated from the following model,

$$Y = X\beta^* + \epsilon,$$

where we have $\mathbb{E}[\epsilon] = 0$ and $\text{Var}(\epsilon) = \sigma^2 I_n$. Given a sampled response vector $Y$, we can fit an OLS model as follows:

$$\hat{\beta} = \argmin_\beta ||Y - X\beta||^2 = (X^\top X)^{-1} X^\top Y$$

For simplicity, we'll assume that $X$ is full-rank so that $X^\top X$ is invertible.

To understand if this is a good estimator, let's think about its bias and variance. We'll do so assuming a fixed design matrix $X$, which makes this analysis considerably easier. The bias and variance for our OLS estimator $\hat{\beta}$ are the following:

$$\text{Bias}(\hat{\hat{\beta}}) = \mathbb{E}[\hat{\beta}] - \beta^* = (X^\top X)^{-1} X^\top \mathbb{E}[Y] - \beta^* = 0$$

<!-- $$\text{Var}(\hat{\beta}) = (X^\top X)^{-1} X^\top \text{Var}(Y) X (X^\top X)^{-1}$$ -->

<!-- $$\text{Var}(\hat{\beta}) = (X^\top X)^{-1}$$ -->

$$\text{Var}(\hat{\beta}) = (X^\top X)^{-1} X^\top \text{Var}(Y) X (X^\top X)^{-1} = \sigma^2 (X^\top X)^{-1}$$

<!-- Let's also think about the expected prediction error, or the deviation between $X\hat{\beta}$ and the correct predictions $X\beta^*$:

$$\mathbb{E}[||X\beta^* - X\hat{\beta}||^2] = ||X\beta^* - X \mathbb{E}[\hat{\beta}]||^2 + \mathbb{E}[||XX \mathbb{E}[\hat{\beta}] - X\hat{\beta}||^2]$$

Hold on, this error decomposition is annoying, let me do this more generically because it will end up being bias + variance -->

So OLS is unbiased, and its variance is related to the inverse data covariance, $\Sigma = X^\top X$.
<!-- the variance is a function of both $\text{Var}(Y)$ and the data matrix $X$. -->

## Compressed least squares (CLS)

Now let's talk about a different approach to fitting a linear model. Rather than augmenting the least squares objective with a penalty term, an approach called *compressed least squares* (CLS) [2] regularizes the model by constraining it to operate on a projection of the features. It's called *compressed* because the projection is typically into a lower-dimensional space (similar to [compressed sensing](https://en.wikipedia.org/wiki/Compressed_sensing)).

Given a projection matrix $R \in \mathbb{R}^{d \times p}$ where $p < d$, CLS solves a least squares problem using the data $XR$:

<!-- $$\hat{\beta}_R = \argmin_\beta ||Y - XR\beta||^2 = (R^\top X^\top XR)^{-1} R^\top X^\top Y$$ -->

$$\argmin_\beta ||Y - XR\beta||^2$$

Solving this problem is equivalent to solving OLS (calculate the derivative and set it to zero), and we can denote the solution as $\hat{\beta}_R$:

$$\hat{\beta}_R = (R^\top X^\top XR)^{-1} R^\top X^\top Y$$

For simplicity, we're assuming that the projection matrix $R$ is full-rank, so that $R^\top X^\top XR$ is invertible. Once again, we can think about the estimator's quality by analyzing its bias and variance. Rather than analyzing $\hat{\beta}_R \in \mathbb{R}^p$, we'll calculate the bias and variance of $R\hat{\beta}_R \in \mathbb{R}^d$, which is effectively a coefficient vector applied to $X$:

$$\text{Bias}(R\hat{\beta}_R) = \mathbb{E}[R\hat{\beta}_R] - \beta^* = R (R^\top X^\top XR)^{-1} R^\top X^\top X \beta^* - \beta^*$$

<!-- $$\text{Var}(R\hat{\beta}_R) = R (R^\top X^\top X R)^{-1} X^\top \text{Var}(Y) X (R^\top X^\top X R)^{-1}R^\top $$ -->

<!-- $$\text{Var}(R\hat{\beta}_R) = \sigma^2 R (R^\top X^\top X R)^{-1} R^\top X^\top X R (R^\top X^\top X R)^{-1}R^\top $$ -->

$$\text{Var}(R\hat{\beta}_R) = \sigma^2 R (R^\top X^\top X R)^{-1}R^\top $$

The expressions are more complicated, and it's perhaps not yet obvious how CLS compares to OLS, except that its bias is non-zero. Next, we'll reframe the bias/variance analysis in a way that makes OLS and CLS easier to compare.

## Error analysis

Depending on whether you're a statistician or a computer scientist, you may be less interested in estimating $\beta^*$ than making accurate predictions. Here, we'll compare OLS and CLS by focusing on the latter. Fortunately, it turns out that these problems are closely related.

Consider a procedure for generating predictions based on the randomly sampled data $X, Y$, denoted $\hat{Y}$. For example, OLS produces the predictions $\hat{Y} = X (X^\top X)^{-1}X^\top Y$. Let's consider the expected error of such a procedure relative to the idealized predictions $X\beta^*$. The expected error is the following:

$$\text{Error}(\hat{Y}) = \mathbb{E}[||X\beta^* - \hat{Y}||^2]$$

We can decompose this as follows,

$$\mathbb{E}[||X\beta^* - \hat{Y}||^2] = ||X\beta^* - \mathbb{E}[\hat{Y}]||^2 + \mathbb{E}[ ||\hat{Y} - \mathbb{E}[\hat{Y}||^2 ]$$

which can be interpreted as a bias-variance decomposition:

$$\text{Error}(\hat{Y}) = ||\text{Bias}(\hat{Y})||^2 + \text{Tr}(\text{Var}(\hat{Y}))$$

Thus, the error relative to the idealized predictions is the sum of a bias term and a variance term. We can use this approach to compare the expected error from the OLS and CLS fitting procedures.

Beginning with OLS, we see that the error is the following:

$$\text{Error}(X \hat{\beta}) = 0 + \sigma^2 \text{Tr}(X (X^\top X)^{-1} X^\top) = \sigma^2 d$$

Next, the expected error for CLS is a bit more complicated, so let's handle it one term at a time. First, the variance term:

$$\text{Tr}(\text{Var}(XR\hat{\beta}_R)) = \sigma^2 \text{Tr}(X R (R^\top X^\top X R)^{-1}R^\top X^\top) = \sigma^2 p$$

Note that this depends on $p$ rather than $d$. Next, the bias term is the following:

$$||\text{Bias}(XR\hat{\beta}_R)||^2 = ||XR (R^\top X^\top XR)^{-1} R^\top X^\top X \beta^* - X\beta^*||^2 = (X\beta^*)^\top (XR(R^\top X^\top XR)^{-1}R^\top X^\top - I)^2 (X\beta^*)$$

If we let $H_R$ denote the *hat matrix* for our CLS model, or $H_R = XR(R^\top X^\top XR)^{-1} R^\top X^\top$, then we have:

$$||\text{Bias}(XR\hat{\beta}_R)||^2 = (X\beta^*)^\top (H_R - I)^2 (X\beta^*) = (X\beta^*)^\top (I - H_R) (X\beta^*)$$

Putting the bias and variance terms together, we have the following expected error from CLS:

$$\text{Error}(XR \hat{\beta}_R) = (X\beta^*)^\top (I - H_R) (X\beta^*) + \sigma^2 p$$

With this, we can now compare OLS and CLS more easily. In terms of the bias, OLS is unbiased whereas CLS has non-zero bias that is a function of the projection matrix $R$. Specificaly, its bias depends onlyon the hat matrix $H_R$, where its contribution to the error is related to how much $H_R$ deviates from the identity matrix. Intuitively, this looks like a measure of the model flexibility, because the bias is higher if the hat matrix is unable to reproduce the observed labels.

In terms of variance however, CLS has a clear advantage: its variance is $\sigma^2 p$ versus $\sigma^2 d$ from OLS, where $p$ is the projection dimension. This suggests a clear trade-off between OLS and CLS: CLS can incur less variance by operating in a lower-dimensional space, but using too small of a dimension will make the bias large.

Thus, the "sweet spot" for CLS is using a low-dimensional projection that minimizes the bias. If $\sigma^2$ is large enough, then a projection matrix $R$ where CLS outperforms OLS is likely to exist. On an abstract level, this is similar to ridge regression, where we often seek a penalty weight where the increase in bias over OLS is outweighed by the decrease in variance.

<!-- **TODO start editing here**

Consider any parameter estimate that depends on the data, denoted $\beta(\epsilon)$, and consider the ideal predictions $X\beta^*$. We can decompose the expected prediction error as follows:

$$\mathbb{E}_\epsilon [ ||X\beta^* - X\beta(\epsilon)||^2 ] = [ ||X\beta^* - X \mathbb{E}_\epsilon [\beta(\epsilon)] ||^2 ] + \mathbb{E}_\epsilon [ ||X \mathbb{E}_\epsilon[\beta(\epsilon)] - X\beta(\epsilon)||^2 ]$$

Utilizing the trace's cyclic permutation property, we can see how this depends on $\beta(\epsilon)$'s bias and variance:

$$\mathbb{E}_\epsilon [ ||X\beta^* - X\beta(\epsilon)||^2 ] = \text{Tr}(X \text{Bias}(\beta(\epsilon)) \text{Bias}(\beta(\epsilon))^\top X^\top) + \text{Tr}(X \text{Var}(\beta(\epsilon)) X^\top)$$

TODO I'm not sure how this could be wrong, but is it inconsistent with Thanei Theorem 1?

What does this yield for our two estimators, OLS and CLS? Let's start with OLS.

$$\text{Tr}(X \text{Bias}(\hat{\beta}) \text{Bias}(\hat{\beta})^\top X^\top) = 0$$

$$\text{Tr}(X \text{Var}(\hat{\beta}) X^\top) = \text{Var}(Y)$$

TODO maybe derive this more

So the total error we get is:

$$\mathbb{E}_\epsilon [ ||X\beta^* - X\hat{\beta}||^2 ] = \text{Var}(Y)$$

What about in the CLS case? Here, we get the following:

TODO -->

## Choosing the projection matrix: learned or random?

How do we find the sweet spot for CLS, or more specifically, how do we choose the right projection matrix $R$?

I'll discuss three possible choices: 1) choosing based on $Y$, 2) choosing based on $X$, and choosing at random.

### 1. Choosing based on $Y$

When choosing the projection matrix based on $R$, you'd expect that we're essentially choosing the projection matrix that minimizes the in-sample MSE. That's exactly what we get by minimizing the CLS bias term but with $X\beta^*$ replaced by $Y$.

To see this, consider the MSE from fitting CLS with projection matrix $R$:

$$||XR \hat{\beta}_R - Y||^2 = ||XR (R^\top X^\top XR)^{-1} R^\top X^\top Y - Y||^2 = ||(H_R - I) Y||^2 = Y^\top (I - H_R) Y$$

Thus, this is equivalent to minimizing $\text{Error}(X R \hat{\beta}_R)$, but replacing $X\beta^*$ with $Y$. 

**TODO this needs work**

### 2. Choosing based on $X$

Can we redo the analysis with $X$ in the place of $X\beta^*$?

TODO maybe instead of jumping to the end and asking about minimizing that, we should start at the beginning and say we're minimizing the prediction error (rather than expected prediction error) with a different label, either X or Y. That perspective at least introduces these learning-based approaches in a more sensible manner, and we can argue later (maybe) that they're implicitly like minimizing the CLS bias term, but using available proxies for $X\beta^*$.

<!-- First, it's helpful to simplify the bias term and mitigate the unidentifiability problem. Assume that $R^\top X^\top XR = I$. Or actually, should we assume $R^\top R = I$?

$$Y^\top (I - XRR^\top X^\top) Y = Y^\top Y - Y^\top XR R^\top X^\top Y = Y^\top Y - $$ -->

TODO three things need to fit into this story: 1) choose based on X (PCA/PCR), 2) choose based on Y (just choosing projection that minimizes MSE), 3) pick randomly

## Averaged CLS (ACLS)

Hopefully we can reduce the bias? I'm not even sure what I proved anymore, I need to spend some time revisiting my notes/pictures to recall. I'm not sure I ever thought about error relative to ideal predictions. I know I thought about both bias and variance, but I'm not sure I tied them in with the error in this ugly way. Maybe I focused on the bias and variance of predictions, which is just a left multiply?

## More results

Point to a couple papers, like Thanei and Kaban, that provide more results

## Conclusion

## Footnotes

1. People sometimes refer to column subsampling as *bagging*, but bagging technically stands for "bootstrap aggregating" (see [Breiman 1994](https://www.stat.berkeley.edu/~breiman/bagging.pdf)). It refers to subsampling along the row dimension, not the column dimension, even though both are commonly used in random forests. 

## References

1. Maillard
2. Thanei
3. Kaban
4. Slawski
5. Massey (PCR)
