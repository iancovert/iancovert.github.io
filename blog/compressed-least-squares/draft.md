# Averaged compressed least squares: column subsampling for linear models

Flexible ML models tend to overfit, and one way to mitigate that is *column subsampling*, or forcing the model to use only a subset of features. This approach is common in tree ensemble models like [XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html) and [random forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).$^1$

Why does column subsampling work? Intuitively, it's because a model trained with fewer features is less likely to overfit. On the other hand, it operates with less information, so it may be unable to express the correct model; this should increase the error, and that's why random forests average predictions across many models trained with different feature subsets.

I wondered if it was possible to justify column subsampling more precisely by analyzing a simpler model class, and it turns out it is. There's a technique called *averaged compressed least squares* (ACLS) [1] that's basically a linear analogue to random forests, and I'll share a couple basic results about it here.

First, here's how the post is organized:

1. Ordinary least squares (OLS)
2. Compressed least squares (CLS)
3. Error analysis: bias-variance decomposition
4. Averaged compressed least squares (ACLS)

Let's jump in.

## Ordinary least squares (OLS)

Assume a data matrix $X \in \mathbb{R}^{n \times d}$ and label vector $Y \in \mathbb{R}^n$ generated from the following ground truth model,

$$Y = X\beta^* + \epsilon,$$

where we have $\mathbb{E}[\epsilon] = 0$ and $\text{Var}(\epsilon) = \sigma^2 I_n$. Given a sampled response vector $Y$, we can fit an OLS model as follows:

$$\hat{\beta} = \argmin_\beta ||Y - X\beta||^2 = (X^\top X)^{-1} X^\top Y$$

For simplicity, we'll assume that $X$ is full-rank so that $X^\top X$ is invertible.

To understand if this is a good estimator, let's think about its bias and variance. We'll do so assuming a fixed design matrix $X$, which makes this analysis considerably easier. The bias and variance for our OLS estimator $\hat{\beta}$ are the following:

$$\text{Bias}(\hat{\hat{\beta}}) = \mathbb{E}[\hat{\beta}] - \beta^* = (X^\top X)^{-1} X^\top \mathbb{E}[Y] - \beta^* = 0$$

$$\text{Var}(\hat{\beta}) = (X^\top X)^{-1} X^\top \text{Var}(Y) X (X^\top X)^{-1} = \sigma^2 (X^\top X)^{-1}$$

This shows that OLS is unbiased, and that its variance is related to the inverse data covariance, $S = \frac{1}{n} X^\top X$.

## Compressed least squares (CLS)

Now let's talk about a different approach to fitting a linear model. Rather than augmenting the least squares objective with a penalty term, an approach called *compressed least squares* (CLS) [2] regularizes the model by constraining it to operate on a projection of the features. It's called *compressed* because the projection is typically into a lower-dimensional space, similar to [compressed sensing](https://en.wikipedia.org/wiki/Compressed_sensing).

Given a projection matrix $R \in \mathbb{R}^{d \times p}$ with $p < d$, CLS solves a least squares problem using the data $XR$:

$$\argmin_\beta ||Y - XR\beta||^2$$

Solving this problem is equivalent to solving OLS (calculate the derivative and set it to zero), and we can denote the solution as $\hat{\beta}_R$:

$$\hat{\beta}_R = (R^\top X^\top XR)^{-1} R^\top X^\top Y$$

For simplicity, we're assuming that the projection matrix $R$ is full-rank so that $R^\top X^\top XR$ is invertible.

Once again, we can think about the estimator's quality by analyzing its bias and variance. Rather than analyzing $\hat{\beta}_R \in \mathbb{R}^p$, we'll calculate the bias and variance of $R\hat{\beta}_R \in \mathbb{R}^d$, which is effectively a coefficient vector applied to $X$:

$$\text{Bias}(R\hat{\beta}_R) = \mathbb{E}[R\hat{\beta}_R] - \beta^* = R (R^\top X^\top XR)^{-1} R^\top X^\top X \beta^* - \beta^*$$

$$\text{Var}(R\hat{\beta}_R) = \sigma^2 R (R^\top X^\top X R)^{-1}R^\top $$

The expressions are more complicated, and it's perhaps not yet obvious how CLS compares to OLS, except that its bias is non-zero. Next, we'll reframe the bias/variance analysis in a way that makes OLS and CLS easier to compare.

## Error analysis

Depending on whether you're a statistician or a computer scientist, you may be less interested in estimating $\beta^*$ than making accurate predictions. Here, we'll compare OLS and CLS by focusing on the latter.

Consider a procedure for generating predictions based on the randomly sampled data $(X, Y)$, denoted $\hat{Y}$. For example, OLS produces the predictions $\hat{Y} = X (X^\top X)^{-1}X^\top Y$. Let's consider the expected error of such a procedure relative to the idealized predictions $X\beta^*$. The expected error is the following,

$$\text{Error}(\hat{Y}) = \mathbb{E}[||X\beta^* - \hat{Y}||^2],$$

where the expectation is taken relative to the noise $\epsilon$. We can decompose this as follows,

$$\mathbb{E}[||X\beta^* - \hat{Y}||^2] = ||X\beta^* - \mathbb{E}[\hat{Y}]||^2 + \mathbb{E}[ ||\hat{Y} - \mathbb{E}[\hat{Y}||^2 ]$$

which can be interpreted as a bias-variance decomposition:

$$\text{Error}(\hat{Y}) = ||\text{Bias}(\hat{Y})||^2 + \text{Tr}(\text{Var}(\hat{Y}))$$

Thus, the error relative to the idealized predictions is the sum of a bias term and a variance term. We can use this approach to compare the expected error from OLS and CLS.

Beginning with OLS, we see that the error is the following:

$$\text{Error}(X \hat{\beta}) = 0 + \sigma^2 \text{Tr}(X (X^\top X)^{-1} X^\top) = \sigma^2 d$$

Next, the expected error for CLS is a bit more complicated. Let's handle it one term at a time, starting with the the variance term:

$$\text{Tr}(\text{Var}(XR\hat{\beta}_R)) = \sigma^2 \text{Tr}(X R (R^\top X^\top X R)^{-1}R^\top X^\top) = \sigma^2 p$$

Note that this depends on $p$ rather than $d$. Next, the bias term is the following:

$$||\text{Bias}(XR\hat{\beta}_R)||^2 = ||XR (R^\top X^\top XR)^{-1} R^\top X^\top X \beta^* - X\beta^*||^2 = (X\beta^*)^\top (XR(R^\top X^\top XR)^{-1}R^\top X^\top - I)^2 (X\beta^*)$$

If we let $H_R$ denote the hat matrix for our CLS model, or $H_R = XR(R^\top X^\top XR)^{-1} R^\top X^\top$, then we have:

$$||\text{Bias}(XR\hat{\beta}_R)||^2 = (X\beta^*)^\top (H_R - I)^2 (X\beta^*) = (X\beta^*)^\top (I - H_R) (X\beta^*)$$

Putting the bias and variance terms together, we have the following expected error from CLS:

$$\text{Error}(XR \hat{\beta}_R) = (X\beta^*)^\top (I - H_R) (X\beta^*) + \sigma^2 p$$

With this, we can now compare OLS and CLS more easily. In terms of the bias, OLS is unbiased while CLS has non-zero bias that's a function of the projection matrix $R$. Specificaly, the bias term depends only on the hat matrix $H_R$, and its contribution to the error is related to how much $H_R$ deviates from the identity matrix. Intuitively, this looks like a measure of the model flexibility, because the bias is high if the hat matrix is unable to reproduce the observed labels.

On the other hand, CLS has a clear advantage in terms of variance: its variance is $\sigma^2 p$ versus $\sigma^2 d$ from OLS, where $p$ is the projection dimension. This suggests a trade-off between OLS and CLS: CLS can incur less variance by operating in a lower-dimensional space, but using too small of a dimension, or too information-destroying of a projection will make the bias large.

Thus, the "sweet spot" for CLS is using a low-dimensional projection that minimizes the bias. If $\sigma^2$ is large enough, then a projection matrix $R$ such that CLS outperforms OLS is likely to exist. On an abstract level, this is similar to ridge regression: in that case, we must seek a penalty weight where the increase in bias over OLS is outweighed by the decrease in variance.

## Choosing the projection matrix: learned or random?

How do we find the sweet spot for CLS, and how do we choose the right projection matrix $R$? It may be temping to fix the projection dimension $p$ and minimize the expected error from CLS as follows:

$$\min_R \ (X\beta^*)^\top (I - H_R) (X\beta^*)$$

However, this is impossible in practice because we don't have access to $\beta^*$. So what can we do instead? Let's discuss three possible options: 1) choosing $R$ based on $Y$, 2) choosing based on $X$, and choosing at random.

### 1. Choosing based on $Y$

Choosing the projection matrix based on $R$ is straightforward. We should simply minimize the squared error as follows:

$$\min_{R, \beta} \ ||Y - XR\beta||^2 = \min_R \ ||Y - XR(R^\top X^\top XR)R^\top X^\top Y||^2 = \min_R \ Y^\top (I - H_R) Y$$

Note that this looks a lot like our expression for the CLS bias term, only we've replaced $X\beta^*$ with $Y$, which is a reasonable substitution.

Overall, this seems like a reasonable approach. I'm not sure if this has a name, and I'm not covering how to solve for the optimal $R$ under this approach, but we can see a clear similarity in the optimization criterion to minimizing the expected error under CLS. The one missing piece is that this doesn't reflect the improvement in variance we get for using a low-dimensional projection; we could fix that by augmenting the objective with a penalty proportional to $p$, but because we don't know $\sigma^2$ in practice, it may be easier to repeatedly solve this problem for different $p$ and select the best value using held-out validation data.

### 2. Choosing based on $X$

Another approach is to select a projection matrix that preserves our ability to reconstruct the design matrix $X$. This sounds a lot like PCA, and indeed we can recover something like the CLS error by manipulating PCA's reconstruction-based characterization.

In this view, we aim to reconstruct $X$ using an embedding variable $Z$ and a projection matrix $W$:

$$\min_{W, Z} \ ||X - ZW||_F^2$$

If we let the embeddings be generated by a linear projection, or $Z = XR$, then we arrive at the following equivalent objective:

$$\min_{W, R} \ ||X - XRW||_F^2$$

Solving for the optimal $W$ given fixed $R$ is straightforward, so we can then update the objective as

$$\min_R \ ||X - XR (R^\top X^\top XR)^{-1} R^\top X^\top X||_F^2 = \text{Tr}(X^\top (I - H_R) X)$$

This resembles the CLS bias term, only we've replaced $X\beta^*$ with $X$ and taken the trace, which amounts to summing the prediction error over each of the data matrix columns.

The problem does not have a unique solution without applying additional constraints, but the solution given by PCA based on the SVD of $X$, or $X = U \Lambda V^\top$. The solution is given by $R = V_p$, and we can also see that $W = V_p^\top$, where $V_p \in \mathbb{R}^{d \times p}$ contains the first $p$ columns of $V$ corresponding to the $p$ largest singular values.

This is another reasonable solution to choose a projection matrix $R$, but it too also fails to account for the improvement in variance we incur for using a low-dimensional projection; once again, it may be easiest to fit PCA for different $p$ values and then select the best dimension using held-out validation data.

### 3. Choosing randomly

Finally, a simpler option is to select $R$ at random. 

**TODO: continue here**

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
