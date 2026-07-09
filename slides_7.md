# Machine Learning Practice Exam

**Instructions:**
- This test contains multiple-choice questions based strictly on the provided lecture slides.
- **CRITICAL:** For every question, the number of correct options can be **0, 1, or more than 1**. Carefully evaluate each option.
- Points are assigned based on the complexity and depth of the topic.

---

## Part 1: The Test

### Topic 1: Decision Trees (Properties & Construction)

**Question 1: Properties of Decision Trees [2 Points]**
Which of the following statements accurately describe the theoretical and practical properties of Decision Trees?
- [ ] Determining an optimal (minimal) decision tree is a computationally trivial task if the dataset is small.
- [ ] Decision trees are highly dependent on the scale of the features; thus, data normalization is strictly required.
- [ ] A decision tree's decision boundaries are invariant to the rotation of the training data.
- [ ] Decision trees are robust to small perturbations in the training dataset, ensuring stable topology across different folds.

**Question 2: Construction and Splitting Criteria [3 Points]**
During the recursive construction of a standard classification decision tree, how is the splitting of a node determined?
- [ ] A random subset of features is evaluated at each node, and the dimension that produces the highest entropy is chosen.
- [ ] All $d$ dimensions are checked for a node, and the algorithm determines the split that ultimately produces the least error based on measures like Gini index or entropy.
- [ ] The algorithm splits nodes randomly and relies entirely on post-pruning to establish the correct decision rules.
- [ ] The root node initially represents a randomly sampled subset of the data points, which are then split based on classification error.

**Question 3: Decision Trees for Regression [2 Points]**
When adapting decision trees for regression tasks, how does the model handle the continuous output space?
- [ ] The error measure typically used to evaluate the quality of a split is the Gini index.
- [ ] Each node or leaf in the tree represents a constant function, which is usually the average of all $y$ values of the data points contained in that node.
- [ ] The prediction boundary visually forms a smooth, continuous polynomial curve across the feature space.
- [ ] The regression tree aims to output the exact target value of the single nearest neighbor in the training data for any given leaf.

### Topic 2: Ensemble Learning & Bagging

**Question 4: Bootstrapping Characteristics [3 Points]**
Regarding the bootstrapping technique used in ensemble learning:
- [ ] A subsample of $m$ data points is selected from the original sample exclusively without replacement to ensure independent estimators.
- [ ] The $B$ bootstrap estimators generated ($\hat{\Theta}_1, \hat{\Theta}_2, ..., \hat{\Theta}_B$) are perfectly independent of one another.
- [ ] Bootstrapping allows us to estimate the distribution of an estimator $\hat{\Theta}$, which can be used to construct confidence intervals.
- [ ] The mathematical proofs regarding the estimators are simplified because the estimators are guaranteed to be independent and identically distributed (i.i.d.).

**Question 5: Bootstrap Aggregation (Bagging) Bias and Variance [3 Points]**
Which of the following accurately describes the statistical impact of bagging on simple classifiers/regressors?
- [ ] Bagging generally increases the variance of the individual estimators, leading to a wider spread in predictions.
- [ ] If all individual estimators have the same large bias, the aggregated bagging estimator $\hat{\Theta}_{bag}$ will unfortunately have this large bias too.
- [ ] Assuming the individual estimators are perfectly i.i.d. and have a variance of $\sigma^2$, the bagging estimator $\hat{\Theta}_{bag}$ will have a variance of $\sigma^2 / B$.
- [ ] Bagging is only applicable to classification problems and cannot be utilized for regression tasks.

### Topic 3: Random Forests

**Question 6: Random Forest Construction & Hyperparameters [4 Points]**
A Random Forest is constructed by ensembling multiple decision trees. Based on the provided slides, which of the following are standard choices or properties of this algorithm?
- [ ] The number of dimensions $p$ that are randomly selected for a potential split at each node is usually $p = d/3$.
- [ ] The minimum number of data points required to split a node is often set to $n_{min} = 1$, which typically makes the individual decision trees deep.
- [ ] While deep individual trees can easily overfit, this tendency is heavily reduced by the bagging process inherent to Random Forests.
- [ ] To output a regression prediction, a Random Forest returns the majority vote of all the individual decision trees $T_1, ..., T_B$.

### Topic 4: Evaluation Metrics (Regression)

**Question 7: Regression Metrics [2 Points]**
Which of the following statements about continuous target evaluation metrics are true?
- [ ] Mean Squared Error (MSE) penalizes small errors more strongly than large errors.
- [ ] Root Mean Squared Error (RMSE) has the same mathematical unit as the target variable.
- [ ] A Coefficient of Determination ($R^2$) of 0 indicates a perfect prediction model.
- [ ] If $R^2 < 0$, it implies the model is performing worse than simply predicting the mean of the target values.

**Question 8: MSE vs. MAE [2 Points]**
When choosing between Mean Squared Error (MSE) and Mean Absolute Error (MAE):
- [ ] MAE is considered more robust to outliers than MSE or RMSE.
- [ ] MSE is highly sensitive to outliers because the difference $(y^{(i)} - \hat{y}^{(i)})$ is squared.
- [ ] MAE calculates the average absolute deviation and is highly uninterpretable because it changes the units of the target variable.
- [ ] If a dataset has heavy outliers and we do not want to heavily penalize them, MSE is the preferred metric over MAE.

### Topic 5: Evaluation Metrics (Classification)

**Question 9: Binary Classification and Imbalance [3 Points]**
Consider a credit-card fraud detection problem where 1% of transactions are fraud and 99% are legitimate. A model is trained that simply always predicts "legitimate".
- [ ] The model will achieve an Accuracy of 99%.
- [ ] The model will have 0 True Positives ($TP = 0$).
- [ ] The model will have 0 False Positives ($FP = 0$).
- [ ] Because the accuracy is 99%, this metric objectively proves the model is highly effective for fraud detection.

**Question 10: Precision, Recall, and F1-Score [3 Points]**
Identify the correct mathematical and conceptual definitions of binary evaluation metrics:
- [ ] Precision is defined as $TP / (TP + FP)$.
- [ ] Recall (or sensitivity) is defined as $TP / (TP + FN)$.
- [ ] Specificity is defined as $TN / (TN + FN)$.
- [ ] The F1-score is defined as $\frac{1}{2} (\text{Precision} + \text{Recall})$.

**Question 11: Multiclass Averaging (Macro vs. Micro) [4 Points]**
When extending evaluation metrics to multiclass classification problems with $K$ classes:
- [ ] Macro-averaging treats each data point with equal weight, implicitly favoring the majority class.
- [ ] Micro-averaging computes the metric globally over all examples jointly.
- [ ] In a standard single-label multiclass classification task, Micro-Precision, Micro-Recall, Micro-F1, and Accuracy are mathematically equivalent.
- [ ] Macro-Precision is calculated by summing the global true positives and dividing by the global predicted positives.

### Topic 6: Significance Testing in ML

**Question 12: Permutation Test Null Hypothesis [2 Points]**
Which of the following aligns with the Null Hypothesis ($H_0$) in a permutation test for comparing two ML models A and B?
- [ ] The evaluation metrics for models A and B follow a normal distribution.
- [ ] Model A and Model B have the exact same true performance.
- [ ] Any observed difference between the models' performances on the test set is caused only by random variation.
- [ ] For a given test example, the predictions of A and B are fundamentally non-exchangeable.

**Question 13: Permutation Test Mechanics [3 Points]**
How is the permutation test explicitly carried out according to the slides?
- [ ] We randomly swap the actual true target labels $y^{(i)}$ independently for each test example with probability $1/2$.
- [ ] We randomly swap the *predictions* of Model A and Model B independently for each test example with probability $1/2$.
- [ ] We evaluate a permuted test statistic $s_{perm} = m(A') - m(B')$, and increment a counter $p_{out}$ if $s_{perm} \ge s_{obs}$.
- [ ] The final p-value is calculated as $p = \frac{1 + p_{out}}{1 + T}$, where $T$ is the number of permutations.

**Question 14: Interpreting the p-Value [3 Points]**
After running a permutation test, you calculate the p-value. What does this p-value actually measure and signify?
- [ ] A large p-value mathematically proves that Model A and Model B are equally good.
- [ ] A small p-value indicates that the observed difference $s_{obs}$ is unusually large under $H_0$, providing evidence against $H_0$.
- [ ] The test measures the absolute, fundamental quality of Model A independently of Model B.
- [ ] The p-value answers the question: "How often do we obtain a difference $\ge s_{obs}$ by chance under $H_0$?"

**Question 15: Advantages of the Permutation Test [2 Points]**
Why is the permutation test particularly useful in Machine Learning?
- [ ] It requires strict distributional assumptions, ensuring statistical rigor.
- [ ] It works for arbitrary evaluation metrics and arbitrary models.
- [ ] It naturally accounts for the paired structure of the test set.
- [ ] It is completely theoretical and highly difficult to implement in practical code.

---

## Part 2: Answer Key & Explanations

### Topic 1: Decision Trees (Properties & Construction)

**Question 1:**
- [ ] False. Slide 16 explicitly states that determining an optimal, i.e., minimal decision tree is NP-hard, making it computationally difficult, not trivial.
- [ ] False. Slide 16 notes that decision trees are "independent of feature scaling", meaning normalization is not required.
- [ ] False. Slide 16 emphasizes that decision trees are "very sensitive to the rotation of training data".
- [ ] False. Slide 16 states that decision trees "can be very sensitive to small changes in data", meaning they are not perfectly robust or stable.
*(Note: This question was designed where 0 options are correct.)*

**Question 2:**
- [ ] False. A standard decision tree does not randomly sample features at each node; this is a property of Random Forests (Slide 32). Standard DTs check all dimensions.
- [x] True. Slide 11 explicitly dictates to "check all $d$ dimensions for a node and determine the best split" and "choose the dimension that ultimately produces the least error."
- [ ] False. The construction is done recursively by choosing the best split (Slide 4), not randomly.
- [ ] False. The root node initially represents *all* data points, not a random subset (Slide 4).

**Question 3:**
- [ ] False. Gini index is for classification (Slide 10). Slide 18 states the error measure for a regression split is often the "squared error".
- [x] True. Slide 18 states "a node or a leaf in the tree represents a constant function" which is "often the average of all $y$ values of the data points contained in this node."
- [ ] False. The visuals on Slides 17-24 show that regression trees output step-functions (constant horizontal lines for each partition), not smooth polynomial curves.
- [ ] False. A leaf outputs the average of the $y$ values of *all* data points in that node, not just a single nearest neighbor (Slide 18).

### Topic 2: Ensemble Learning & Bagging

**Question 4:**
- [ ] False. Slide 28 states that a subsample of $m$ data points is selected "(with or without replacement)", not exclusively without.
- [ ] False. Slide 28 specifically warns that "the estimators are not independent, making the mathematics/proofs a bit more challenging."
- [x] True. Slide 27 mentions we are interested in the distribution of estimates to "construct confidence intervals."
- [ ] False. As stated on Slide 28, the estimators are not independent, which makes the proofs *more* challenging, not simplified.

**Question 5:**
- [ ] False. Slide 29 highlights as an advantage that $\hat{\Theta}_{bag}$ "can have a much smaller variance than any of the individual estimators".
- [x] True. Slide 31 clearly states: "if all individual estimators have the same large bias, unfortunately, the bagging estimator will have this too".
- [x] True. Slide 29 provides the formula: "if the estimators are i.i.d. and have a variance of $\sigma^2$, the bagging estimator $\hat{\Theta}_{bag}$ would have a variance of $\sigma^2/B$."
- [ ] False. Slide 30 explicitly states: "let's apply bagging to a classification or regression problem."

### Topic 3: Random Forests

**Question 6:**
- [x] True. Slide 34 mentions a good choice for the number of dimensions randomly selected for a split is "usually $d/3$".
- [x] True. Slide 34 notes that "often $n_{min} = 1$, so the decision trees become deep".
- [x] True. Slide 34 continues by saying deep trees can lead to overfitting of an individual tree, "but this is reduced by bagging".
- [ ] False. Slide 33 shows that for regression, a Random Forest returns the "average", whereas the "majority vote" is used strictly for classification.

### Topic 4: Evaluation Metrics (Regression)

**Question 7:**
- [ ] False. Slide 38 states that MSE "penalizes *large* errors more strongly" due to the squaring function, not small errors.
- [x] True. Slide 39 lists "same unit as the target variable" as a property of RMSE.
- [ ] False. Slide 40 shows that $R^2 = 1$ is a perfect prediction. $R^2 = 0$ is no better than predicting the mean.
- [x] True. Slide 40 states that $R^2 < 0$ means the prediction is "worse than predicting the mean".

**Question 8:**
- [x] True. Slide 41 explicitly lists that MAE is "more robust to outliers than MSE or RMSE."
- [x] True. Slide 38 lists that MSE is "sensitive to outliers" (which derives from squaring the $(y^{(i)} - \hat{y}^{(i)})^2$ term).
- [ ] False. Slide 41 states MAE is "easy to interpret: same unit as the target variable". It does not change the units.
- [ ] False. If we do *not* want to heavily penalize outliers, we should use MAE. MSE *does* penalize large errors (outliers) heavily, as per Slide 38.

### Topic 5: Evaluation Metrics (Classification)

**Question 9:**
- [x] True. Slide 44 shows that predicting exactly 99 true negatives (legitimate) out of 100 yields an accuracy of $99/100 = 99\%$.
- [x] True. Slide 44 shows $TP = 0$, since not a single fraud case is correctly predicted.
- [x] True. Slide 44 shows $FP = 0$, since no legitimate cases are falsely flagged as fraud.
- [ ] False. Slide 44 points out that "nevertheless, not a single fraud case is detected" and Slide 43 notes accuracy "can be misleading for imbalanced classes."

**Question 10:**
- [x] True. Slide 45 provides the formula: $\text{Precision} = TP / (TP + FP)$.
- [x] True. Slide 45 provides the formula: $\text{Recall} = TP / (TP + FN)$.
- [ ] False. Slide 46 defines Specificity as $TN / (TN + FP)$, not $FN$.
- [ ] False. Slide 45 defines F1 as the harmonic mean: $2 \cdot (\text{Precision} \cdot \text{Recall}) / (\text{Precision} + \text{Recall})$, not the arithmetic mean.

**Question 11:**
- [ ] False. Slide 49 provides the rule of thumb: "Macro: each class has equal weight". It is Micro that gives each *data point* equal weight.
- [x] True. Slide 49 defines micro average as "compute the metric globally over all examples jointly".
- [x] True. Slide 52 explicitly notes that for standard single-label multiclass classification: "micro-precision = micro-recall = micro-F1 = accuracy".
- [ ] False. Macro-Precision averages the *class-wise* metric values (Slide 49/53). Summing global TP and dividing by global predicted positives is the formula for *Micro*-Precision (Slide 53).

### Topic 6: Significance Testing in ML

**Question 12:**
- [ ] False. Slide 59 specifically notes that "no distributional assumptions are required" for the permutation test.
- [x] True. Slide 55 defines the null hypothesis $H_0$ as: "Model A and model B have the same true performance".
- [x] True. Slide 55 defines the second part of $H_0$ as: "their observed difference is caused only by random variation".
- [ ] False. Slide 55 states that if $H_0$ is true, "the predictions of A and B are exchangeable".

**Question 13:**
- [ ] False. Slide 56 Step 2 states we randomly swap the "predictions of A and B", not the true target labels.
- [x] True. Slide 56 Step 2 states to "randomly swap the predictions of A and B independently for each test example with probability $1/2$."
- [x] True. Slide 56 Steps 4 & 5 explicitly detail calculating $s_{perm} = m(A') - m(B')$ and incrementing $p_{out}$ if $s_{perm} \ge s_{obs}$.
- [x] True. Slide 56 Step 7 estimates the p-value with the formula $p = (1 + p_{out}) / (1 + T)$.

**Question 14:**
- [ ] False. Slide 58 explicitly warns: "important: a large p-value does not prove that the two models are equally good. It only means: 'We do not find evidence that they differ.'"
- [x] True. Slide 58 states under interpretation: "small p-value $\rightarrow$ $s_{obs}$ is unusually large under $H_0$ $\rightarrow$ evidence against $H_0$."
- [ ] False. Slide 58 clarifies that "the test does not measure the absolute quality of the models, but the evidence for a difference between them".
- [x] True. Slide 58 verbatim defines the p-value as measuring: "How often do we obtain a difference $\ge s_{obs}$ by chance under $H_0$?"

**Question 15:**
- [ ] False. Slide 59 states that "no distributional assumptions are required".
- [x] True. Slide 59 lists "works for arbitrary evaluation metrics" and "works for arbitrary models" as key advantages.
- [x] True. Slide 59 lists "naturally accounts for the paired structure of the test set" as an advantage.
- [ ] False. Slide 59 states it is "easy to implement: randomly swap predictions and recompute the metric", completely contradicting the claim that it is highly difficult.