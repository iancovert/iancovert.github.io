# Averaged compressed least squares

Flexible models tend to overfit, and one way to mitigate that is *column subsampling*: that is, forcing the model to use only a subset of features. This approach is common in tree ensemble models, for example.

Why does column subsampling work? Intuitively, it's because a model trained with fewer features is less likely to overfit to noise in the data. On the other hand, it operates on less information, so it may be unable to express the true model; this will increase the error, and that's why random forests average the predictions across many models trained with different feature subsets.

I wondered whether it was possible to make this argument more precise in another context that's easier to analyze, and it turns out it is. There's a technique called *averaged compressed least squares* that's basically a linear analogue to random forests, and I found it pretty interesting so I'll share a couple basic results here.

<!-- Why does column subsampling work? To think about why doing so could improve a model's out-of-sample perforance, let's consider the standard bias-variance decomposition. 

[Need to explain this, including the notation.] -->

<!-- $$\mathbb{E}_{\mathcal{D}, \epsilon}[(f(x; \mathcal{D}) - y)^2] = \mathbb{E}_\mathcal{D}[f(x; \mathcal{D} - \mathbb{E}_\mathcal{D}[f(x; \mathcal{D})])^2] + (\mathbb{E}_\mathcal{D}[f(x; \mathcal{D})] - y)^2 + \sigma^2$$ -->

<!-- $$\mathbb{E}_{\mathcal{D}, \epsilon}[(\hat{f}(x; \mathcal{D}) - y)^2] = (\text{Bias}_\mathcal{D}[\hat{f}(x; \mathcal{D})] )^2 + \text{Var}_\mathcal{D}[\hat{f}(x; \mathcal{D})] + \sigma^2$$

$$\text{Bias}_\mathcal{D}[\hat{f}(x; \mathcal{D})] = \mathbb{E}_\mathcal{D}[\hat{f}(x; \mathcal{D})] - f(x)$$

$$\text{Var}(\hat{f}(x; \mathcal{D})) = \mathbb{E}_\mathcal{D}[(\hat{f}(x; \mathcal{D}) - \mathbb{E}_\mathcal{D}[\hat{f}(x; \mathcal{D})])^2]$$

When we train using a smaller number of features, two things should happen. First, the relative lack of flexibility means that our model may be unable to expression the true function $f(x)$, so the bias should increase; this should increase the error. Second,  -->

## Ordinary least squares (OLS)

Assume data $X \in \mathbb{R}^{n \times d}$ and $Y \in \mathbb{R}^n$ generated as follows:

$$Y = X\beta^* + \epsilon$$

Given a sampled response vector $Y$, we can fit an OLS model as follows:

$$\hat{\beta} = \argmin_\beta ||Y - X\beta||^2 = (X^\top X)^{-1} X^\top Y$$

To understand if this is a good estimator, let's think about its bias and variance:

$$\text{Bias}(\hat{\hat{\beta}}) = \mathbb{E}[\hat{\beta}] - \beta^* = 0$$

$$\text{Var}(\hat{\beta}) = (X^\top X)^{-1} X^\top \text{Var}(Y) X (X^\top X)^{-1}$$

<!-- Let's also think about the expected prediction error, or the deviation between $X\hat{\beta}$ and the correct predictions $X\beta^*$:

$$\mathbb{E}[||X\beta^* - X\hat{\beta}||^2] = ||X\beta^* - X \mathbb{E}[\hat{\beta}]||^2 + \mathbb{E}[||XX \mathbb{E}[\hat{\beta}] - X\hat{\beta}||^2]$$

Hold on, this error decomposition is annoying, let me do this more generically because it will end up being bias + variance -->

So it's unbiased, and the variance is a function of both $\text{Var}(Y)$ and the data matrix $X$.

## Compressed least squares (CLS)

In CLS, we have a projection matrix $R \in \mathbb{R}^{d \times p}$ where $p < d$. We then solve a standard least squares problem using the data $XR$:

$$\hat{\beta}_R = \argmin_\beta ||Y - XR\beta||^2 = (R^\top X^\top XR)^{-1} R^\top X^\top Y$$

Once again, we can think about the bias and variance of the estimator:

$$\text{Bias}(R\hat{\hat{\beta}_R}) = \mathbb{E}[R\hat{\beta}_R] - \beta^* = R (R^\top X^\top XR)^{-1} R^\top X^\top X - \beta^*$$

$$\text{Var}(R\hat{\beta}_R) = R (R^\top X^\top X R)^{-1} X^\top \text{Var}(Y) X (R^\top X^\top X R)^{-1}R^\top $$

The expressions are more complicated, and it's not obvious whether this is any better. Next, we'll reframe the bias/variance analysis in terms of the expected error.

## Error analysis

Our goal here is to analyze the prediction error rather than the bias/variance of the parameter estimates; it turns out that these problems are closely related.

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

TODO

## Averaged CLS (ACLS)

Hopefully we can reduce the bias? I'm not even sure what I proved anymore, I need to spend some time revisiting my notes/pictures to recall. I'm not sure I ever thought about error relative to ideal predictions. I know I thought about both bias and variance, but I'm not sure I tied them in with the error in this ugly way. Maybe I focused on the bias and variance of predictions, which is just a left multiply?

## More results

Point to a couple papers, like Thanei and Kaban, that provide more results

## Conclusion

## References

1. Maillard
2. Thanei
3. Kaban
4. Slawski
5. Massey (PCR)
