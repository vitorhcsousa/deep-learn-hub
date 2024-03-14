





[TOC]

# Basics

***Exponents*** -> multiply the number by itself a specified number of times $2^3 = 2*2*2 = 8$ 

***Logarithms*** -> Is a math function that finds the power of a specific number and base. “2 raised to what power gives me 8?” $log_28= x; x = 3$ 

***Derivatives*** -> Tells the slope of a function, that measures the rate of change at any point in a function $\frac{d}{dx}$  indicates the derivative for x.  If we have $f(x)=x^2$ then 

$\frac{d}{dx}f(x) = \frac{d}{dx}x^2 = 2x \rightarrow \frac{d}{dx}f(2)=2(2)=4$

***Chain Rule*** -> The chain rule is a fundamental concept in calculus that allows us to find the derivative of composite functions. If we have a function composed of two or more functions, such as $f(g(x))$, where $f$ and $g$ are functions, then the chain rule states that the derivative of this composite function is the derivative of the outer function evaluated at the inner function, multiplied by the derivative of the inner function. Symbolically, if $y = f(u)$ and $u = g(x)$, then the chain rule is expressed as $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$.

For example, if we have $f(x) = (x^2 + 1)^3$, then $f'(x)$ can be found using the chain rule. Letting $u = x^2 + 1$, we find $\frac{df}{dx} = \frac{df}{du} \cdot \frac{du}{dx}$. Differentiating $u^3$ to $u$ gives $3u^2$, and differentiating $x^2 + 1$ concerning $x$ gives $2x$. Therefore, $f'(x) = 3(x^2 + 1)^2 \cdot 2x$.

***Integrals*** -> Integrals are the reverse operation of derivatives. **They compute the area under the curve of a function over a given interval.** The definite integral of a function $f(x)$ from $a$ to $b$ is denoted as $\int_{a}^{b} f(x) \, dx$, which represents the net signed area between the x-axis and the curve $f(x)$ from $x = a$ to $x = b$. The indefinite integral, or anti derivative, is denoted as $\int f(x) \, dx$, and it represents a family of functions whose derivative is $f(x)$.

For example, if we have $f(x) = 2x$ and we want to find the definite integral of $f(x)$ from $1$ to $3$, we would calculate $\int_{1}^{3} 2x \, dx$. This would give us the area under the curve of $f(x) = 2x$ from $x = 1$ to $x = 3$.

If we want to find the indefinite integral of $f(x) = 2x$, we would calculate $\int 2x \, dx$, which gives us $x^2 + C$, where $C$ is the constant of integration.

---

# Probability 

Some people feel more comfortable in expressing probabilities in probabilities but ***odds*** are useful as well.  An odd of 2.0  means that an event is 2x more likely to happen than not happen.



> ***Probability vs Statistics***: Probability is purely theoretical of how likely an event is to happen and does not requires data. Statistics cannot exist without data!

## Probability Math 



#### **Probability Theory**:

Probability theory deals with quantifying uncertainty. It defines the probability of an event occurring, denoted by $ P(A) $, where $ A $ is an event, as the ratio of the number of favourable outcomes to the total number of possible outcomes. Mathematically:

$$ P(A) = \frac{{\text{{Number of favorable outcomes}}}}{{\text{{Total number of possible outcomes}}}} $$

##### **Properties of Probability**

- **Addition Rule**: The probability of the union of two events \( A \) and \( B \) is the sum of the probabilities of each event, minus the probability of their intersection.
  $$ P(A \cup B) = P(A) + P(B) - P(A \cap B) $$
- **Multiplication Rule for Independent Events**: The probability of the intersection of two independent events \( A \) and \( B \) is the product of their probabilities.
  $$ P(A \cap B) = P(A) \times P(B) $$
- **Complement Rule**: The probability of the complement of an event \( A \) is 1 minus the probability of \( A \).
  $$ P(A') = 1 - P(A) $$
- **Conditional Probability**: The probability of an event \( A \) given that another event \( B \) has occurred is the probability of the intersection of \( A \) and \( B \) divided by the probability of \( B \).
  $$ P(A|B) = \frac{P(A \cap B)}{P(B)} $$

##### **Probability Distributions**

- **Probability Density Function (PDF)**: Describes the likelihood of a continuous random variable taking on a specific value.

  <img src="https://media.geeksforgeeks.org/wp-content/uploads/20230825121154/Probability-Density-Function.png" alt="Probability Density Function(PDF): Definition, Formula, Example" style="zoom:25%;" />

- **Probability Mass Function (PMF)**: Describes the likelihood of a discrete random variable taking on a specific value.

  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Discrete_probability_distrib.svg/1200px-Discrete_probability_distrib.svg.png" alt="Probability mass function - Wikipedia" style="zoom: 25%;" />

- **Cumulative Density Function (CDF)**: Gives the probability that a random variable is less than or equal to a certain value.

![Cumulative distribution function - Wikipedia](https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Normal_Distribution_CDF.svg/300px-Normal_Distribution_CDF.svg.png)

1. **De Morgan’s Law**:

   - States that the complement of the union of two sets is equal to the intersection of their complements, and the complement of the intersection of two sets is equal to the union of their complements.
     $$ (A \cup B)' = A' \cap B' $$
     $$ (A \cap B)' = A' \cup B' $$​​

   <img src="https://www.onlinemathlearning.com/image-files/de-morgans-theorem.png" alt="De Morgan's Theorem (solutions, examples, videos)" style="zoom:25%;" />

2. **Independent/Mutually Exclusive Events**:

   - **Independent Events**: Two events \( A \) and \( B \) are independent if the occurrence of one event does not affect the occurrence of the other. Mathematically, this is expressed as:
     $$ P(A \cap B) = P(A) \times P(B) $$
   - **Mutually Exclusive Events**: Two events \( A \) and \( B \) are mutually exclusive if they cannot both happen at the same time. Mathematically, this is expressed as:
     $$ P(A \cap B) = 0 $$

3. **Non-Mutually Exclusive Events**:

   - Events \( A \) and \( B \) are non-mutually exclusive if they can occur at the same time. Their intersection is not zero.

4. **Disjoint Events**:

   - Disjoint events are mutually exclusive events. They cannot occur simultaneously.

5. **Differences and Similarities**:

   - **Independent Events vs Mutually Exclusive Events**: While both concepts involve relationships between events, independent events focus on whether the occurrence of one event affects the probability of another, whereas mutually exclusive events focus on whether events can occur simultaneously.
   - **Non-Mutually Exclusive Events vs Disjoint Events**: Non-mutually exclusive events can overlap and occur together, whereas disjoint events cannot occur together and have no overlap in their outcomes.

#### Bayes' Theorem:

Bayes' theorem allows us to adjust our belief in the likelihood of an event occurring based on new evidence, by combining our prior knowledge with the probability of observing that evidence if the event were true.

**Definition:**

Bayes' theorem provides a way to update our beliefs about the probability of an event based on new evidence.

**Formula:**

It is stated as:

$$ P(A|B) = \frac{{P(B|A) \cdot P(A)}}{{P(B)}} $$

where:

- \( P(A|B) \) is the probability of event \( A \) given \( B \) has occurred,
- \( P(B|A) \) is the probability that event \( B \) given \( A \) has occurred,
- \( P(A) \) and \( P(B) \) are the probabilities of events \( A \) and \( B \) respectively.

**Application:**

Bayes' theorem finds applications in various real-world scenarios such as:

- **Medical Diagnosis**: Updating the probability of a disease given the results of a diagnostic test.
- **Spam Filtering**: Adjusting the likelihood of an email being spam based on certain keywords present in the email.
- **Fault Diagnosis**: Updating the probability of a particular fault in a system given observed symptoms.
- **Risk Assessment**: Revising the likelihood of a risk event occurring based on new information.

By incorporating prior knowledge and updating it with new evidence, Bayes' theorem enables a more accurate estimation of probabilities in various fields.

### **Probability Distributions**:

Probability distributions describe the likelihood of different outcomes in a random experiment. Some common distributions include:

#### **Gaussian (Normal) Distribution**

The Gaussian (Normal) distribution is defined by its probability density function (PDF):

$$f(x|\mu,\sigma^2) = \frac{1}{{\sqrt{2\pi\sigma^2}}} \cdot e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

where:

- $ \mu $ is the mean,
- $ \sigma^2 $ is the variance.

It’s common to rescale a normal distribution so that the mean is 0 and the standard deviation is 1, which is known as *standard normal distribution*, which allow to compare the spread of one normal distribution to another. 

**Properties**

- It’s symetrical; both sides are identically mirrored ate themean, whice is the center
- Most mass is at the center around the mean 
- It has a spread(being narrow or wide) that is specidfied by the standard deviation 
- It resembles a lot of phenomena in nature and daily life, and even generalized nonnormal problems because of the central limit theorem



**Examples of Applications:**

1. **Height of Individuals:** Heights of individuals in a population often follow a normal distribution, with the mean ($ \mu $) and variance ($ \sigma^2 $) representing the average height and the spread of heights, respectively.

2. **Measurement Errors:** Measurement errors in scientific experiments or industrial processes often follow a normal distribution, where $ \mu $ represents the true value and $ \sigma^2 $ represents the variability in the measurements.

3. **Financial Data:** Stock prices and financial returns are often modeled using a normal distribution, with $ \mu $ representing the expected return and $ \sigma^2 $ representing the volatility.

#### **Poisson Distribution**

The Poisson distribution describes the number of events occurring in a fixed interval of time or space. Its Probability Mass Function (PMF) is given by:

$$P(X=k) = \frac{{\lambda^k \cdot e^{-\lambda}}}{{k!}}$$

where:

- $ \lambda $ is the average rate of occurrence.

**Examples of Applications:**

1. **Traffic Flow:** The number of cars passing through a particular intersection in a given time period can be modeled using a Poisson distribution, with $ \lambda $ representing the average rate of cars passing through.

2. **Arrival of Customers:** The number of customers arriving at a service center in a given time period can be modeled using a Poisson distribution, with $ \lambda $ representing the average arrival rate.

3. **Defects in Manufacturing:** The number of defects found in a batch of manufactured products can be modeled using a Poisson distribution, with $ \lambda $ representing the average defect rate.

#### **Bernoulli Distribution**

The Bernoulli distribution represents a binary outcome (success/failure) with a probability $ p $ of success. Its Probability Mass Function (PMF) is defined as:

$$P(X=k) = \begin{cases} p & \text{if } k=1 \\ 1-p & \text{if } k=0 \end{cases}$$

where:

- $ k $ represents the outcome (1 for success, 0 for failure),
- $ p $ is the probability of success.

**Examples of Applications:**

1. **Coin Flipping:** The outcome of a coin flip can be modeled using a Bernoulli distribution, where "heads" might be considered a success (1) and "tails" a failure (0). The probability of getting heads ($ p $) is typically assumed to be 0.5 for a fair coin.

2. **Medical Diagnosis:** In medical diagnosis, a test result might be considered a success if it indicates the presence of a disease and a failure if it indicates the absence. The probability of a positive test result ($ p $) would depend on the sensitivity and specificity of the test.

3. **Customer Conversion:** In marketing, the conversion of a customer (e.g., making a purchase, signing up for a service) can be modeled using a Bernoulli distribution. The probability of conversion ($ p $) might be estimated based on historical data or experimental results.

#### **Exponential Distribution**

- The exponential distribution is often used to model the time until an event occurs in a Poisson process, where events occur continuously and independently at a constant average rate.
- Probability Density Function (PDF): 
  $$ f(x|\lambda) = \lambda e^{-\lambda x} $$
  where $ \lambda $ is the rate parameter.

**Examples of Applications:**

1. **Reliability Analysis:** The exponential distribution is commonly used to model the time until failure of electronic components or systems, where the rate parameter $ \lambda $ represents the failure rate.

2. **Queueing Theory:** In queueing systems, the exponential distribution is used to model inter-arrival times or service times, with $ \lambda $ representing the arrival rate or the service rate, respectively.

3. **Radioactive Decay:** The exponential distribution is used to model the time until decay of radioactive particles, where $ \lambda $ represents the decay constant.

#### **Uniform Distribution**

- In the uniform distribution, all outcomes in an interval are equally likely.
- Probability Density Function (PDF): 
  $$ f(x|a,b) = \frac{1}{b-a} $$
  where $ a $ and $ b $ are the lower and upper bounds of the interval, respectively.

**Examples of Applications:**

1. **Random Number Generation:** The uniform distribution is often used in simulations and random number generation, where each outcome in a given range has an equal probability of occurring.

2. **Probability Models:** In certain probability models, such as the discrete uniform distribution, where outcomes are integers within a specified range, the uniform distribution is used to assign equal probabilities to each outcome.

3. **Statistical Testing:** The uniform distribution serves as a reference distribution in statistical testing, such as the Kolmogorov-Smirnov test for goodness-of-fit, where observed data is compared against expected uniformity.

#### **Binomial Distribution**

- The binomial distribution characterizes the count of successes in a predetermined number of independent Bernoulli trials.
- It quantifies the likelihood of observing $k$ successes among $n$ trials, given a probability $p$ of success.
- The Probability Mass Function (PMF) is expressed as:  
  $$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$$
  where:
  - $n$ denotes the number of trials, and
  - $p$ represents the probability of success on each trial.

**Examples of Applications:**

1. **Coin Flipping:** Modeling the number of heads obtained in a series of coin flips.

2. **Quality Control:** Determining the number of defective items in a production batch based on a sample.

3. **Biological Studies:** Analyzing the number of successful attempts in a series of genetic crosses or drug trials.

#### Beta Distribution

- The Beta distribution is a continuous probability distribution defined on the interval [0, 1].
- It is commonly used to model random variables representing probabilities or proportions.
- The probability density function (PDF) of the Beta distribution is given by:

$$
f(x|a, b) = \frac{x^{a-1}(1-x)^{b-1}}{B(a, b)}
$$

where:

  - \( x $ is the random variable, which represents a probability or proportion.
  - \( a $ and$ b $ are shape parameters, with$ a, b > 0 $.
  - \( B(a, b) $ is the Beta function, a normalization constant ensuring the total area under the curve is 1.

- The mean of the Beta distribution is$ \frac{a}{a+b} $, and its variance is$ \frac{ab}{(a+b)^2(a+b+1)} $.

- The Beta distribution is often used as a conjugate prior for the parameter of a Bernoulli or Binomial distribution in Bayesian inference. This means that if the prior distribution of a parameter is Beta and the likelihood function is Binomial, then the posterior distribution is also Beta.

- It can also be interpreted as a distribution of the probability of success in a series of Bernoulli trials, where$ a $ represents the number of successes and$ b $ represents the number of failures.

- The Beta distribution is flexible and can take various shapes depending on the values of its parameters, allowing it to model a wide range of scenarios involving probabilities or proportions.

#### **Geometric Distribution**

- The geometric distribution models the number of trials needed to achieve the first success in a sequence of independent Bernoulli trials, each with probability *p* of success.
- Probability Mass Function (PMF): $ P(X=k) = (1-p)^{k-1}p  $
- where *k* is the number of trials needed to achieve the first success.

#### **Gamma Distribution**

- The gamma distribution generalizes the exponential distribution to allow for non-integer shape parameters.
- Probability Density Function (PDF): $f(x|k,\theta) = \frac{x^{k-1} e^{-\frac{x}{\theta}}}{\theta^k \Gamma(k)} $
- where *k* is the shape parameter, *θ* is the scale parameter, and Γ(*k*) is the gamma function.

#### How to choose the probability distribution? 

1. **Understand the Data:**
   - Start by thoroughly understanding the data you are working with. Examine the characteristics of the variables, such as their type (continuous, discrete, binary), range, and any patterns or trends present.

2. **Identify the Variable Type:**
   - Determine whether the variable you are modeling is continuous, discrete, or binary. This distinction will help narrow down the set of possible probability distributions.

3. **Consider Domain Knowledge:**
   - Leverage domain knowledge or subject matter expertise to gain insights into the underlying processes generating the data. Understanding the domain-specific context can guide the selection of appropriate distributions.

4. **Assess Data Distribution:**
   - Visualize the distribution of the data using histograms, density plots, or other statistical summaries. This exploration can provide clues about the shape and characteristics of the underlying distribution.

5. **Evaluate Distribution Assumptions:**
   - Consider the assumptions underlying each probability distribution and assess whether they align with the properties of your data. For example, Gaussian distributions assume symmetry and normality, while Poisson distributions assume counts of rare events.

6. **Evaluate Model Requirements:**
   - Consider the requirements of the model or analysis you intend to perform. Some models may have specific distributional assumptions or requirements, such as linear regression models assuming Gaussian errors.

7. **Compare Distributions:**
   - Compare candidate probability distributions based on their theoretical properties, goodness-of-fit measures, and practical considerations. You can use statistical tests or visual inspections to assess the fit of the distributions to the data.

8. **Iterate and Refine:**
   - It's often necessary to iterate and refine the selection process. Experiment with different distributions, model specifications, or transformations of the data to find the best-fitting distribution for your specific application.

9. **Validate the Chosen Distribution:**
   - After selecting a probability distribution, validate its appropriateness for your data through model validation techniques such as cross-validation or hypothesis testing. Assess the performance of models built using the chosen distribution to ensure they accurately represent the underlying data generating process.

#### Examples 

1. **Gaussian (Normal) Distribution**:

   - Examples: Heights of people in a population, errors in measurements, IQ scores.

   - Applications: Widely used in natural and social sciences, finance for modeling stock prices, in engineering for analyzing random noise, in quality control for manufacturing processes.

2. **Poisson Distribution**:

   - Examples: Number of phone calls received by a call center in an hour, number of emails arriving in an inbox per day, number of defects in a product.

   - Applications: Modeling rare events where occurrences are discrete and independent, such as in queuing theory, reliability engineering, and epidemiology.

3. **Bernoulli Distribution**:

   - Examples: Coin flips (success = heads, failure = tails), whether a patient recovers from a disease (success = recovery, failure = not recovered).

   - Applications: Modeling binary outcomes in experiments, such as success or failure of trials, customer churn prediction, click-through rates in online advertising.

4. **Exponential Distribution**:

   - Examples: Lifetimes of electronic components, time between arrivals of consecutive customers at a service point.

   - Applications: Reliability analysis, queuing theory, modeling waiting times and durations in various processes.

5. **Uniform Distribution**:

   - Examples: Rolling a fair die, selecting a random point within a square.

   - Applications: Simulations, random number generation, statistical sampling when each outcome is equally likely.

6. **Binomial Distribution**:

   - Examples: Number of heads obtained when flipping a coin multiple times, number of defective items in a sample from a production line.

   - Applications: Quality control, A/B testing in marketing, modeling success/failure experiments, estimating proportions in populations.

7. **Geometric Distribution**:

   - Examples: Number of attempts needed to score the first success in repeated Bernoulli trials, number of times a gambler needs to play to win for the first time.

   - Applications: Modeling waiting times until the first success, reliability analysis, analyzing the number of trials needed to achieve a certain outcome.

9. **Gamma Distribution**:

   - Examples: Time until a radioactive particle decays, time until a component fails in a system subject to wear and tear.

   - Applications: Survival analysis, reliability engineering, modeling continuous positive random variables with skewed distributions.

#  Random Variables

Random variables are variables whose possible values are outcomes of a random phenomenon. They can be categorized into two main types:

#### **Discrete Variables**

- Discrete random variables take on a countable number of distinct values.
- Examples include the number of children in a family, the outcome of rolling a die, or the number of heads in multiple coin flips.
- The probability distribution of a discrete random variable is described by a probability mass function (PMF).

#### **Continuous Variables**

- Continuous random variables can take on any value within a range or interval.
- Examples include height, weight, temperature, or time.
- The probability distribution of a continuous random variable is described by a probability density function (PDF).

Random variables are fundamental in probability theory and statistics, serving as a way to model uncertain outcomes in various real-world scenarios.



#### Joint and Marginal Distributions

1. **Joint Distribution**:
   - The joint distribution of two or more random variables describes the probability of observing specific combinations of values for those variables simultaneously.
   - Suppose you have two random variables, X and Y. The joint distribution function, denoted as P(X, Y), provides the probability that X takes on one value and Y takes on another value.
   - For discrete random variables, the joint distribution can be represented as a table or a function that assigns probabilities to all possible combinations of values of the variables.
   - For continuous random variables, the joint distribution is represented using a joint probability density function (PDF) or a joint cumulative distribution function (CDF).
2. **Marginal Distribution**:
   - The marginal distribution of a subset of random variables from a joint distribution describes the probability distribution of those variables individually, without considering the values of the other variables.
   - It's obtained by summing or integrating the joint distribution over the values of the variables not of interest.
   - For example, if you have a joint distribution P(X, Y), the marginal distribution of X is obtained by summing or integrating P(X, Y) over all possible values of Y. Similarly, the marginal distribution of Y is obtained by summing or integrating P(X, Y) over all possible values of X.
   - The marginal distribution provides information about the individual behavior of each variable regardless of the other variables.

**Example:**

Suppose we have two random variables, X and Y, representing the scores of two students in a class. The joint distribution \( P(X, Y) \) describes the probability of observing specific combinations of scores for both students. The marginal distributions provide the probability distributions of individual scores for each student.

Consider the following joint distribution table:

|       | Y = 0 | Y = 1 | Y = 2 |
| ----- | ----- | ----- | ----- |
| X = 0 | 0.1   | 0.2   | 0.1   |
| X = 1 | 0.2   | 0.3   | 0.1   |

### Mathematical Representation:

1. **Joint Distribution**:
   - The joint distribution \( P(X, Y) \) can be represented as:
     $$ P(X = x, Y = y) $$
     where \( x \) and \( y \) represent the possible values of X and Y, respectively.
   - For example, \( P(X = 0, Y = 1) = 0.2 \) indicates the probability of the first student scoring 0 and the second student scoring 1.

2. **Marginal Distributions**:
   - The marginal distribution of X is obtained by summing the joint distribution over all possible values of Y:
     $$ P(X = x) = \sum_{y} P(X = x, Y = y) $$
   - Similarly, the marginal distribution of Y is obtained by summing the joint distribution over all possible values of X:
     $$ P(Y = y) = \sum_{x} P(X = x, Y = y) $$

Using the joint distribution table provided, let's calculate the marginal distributions:

- Marginal distribution of X:
   - \( P(X = 0) = P(X = 0, Y = 0) + P(X = 0, Y = 1) + P(X = 0, Y = 2) = 0.1 + 0.2 + 0.1 = 0.4 \)
   - \( P(X = 1) = P(X = 1, Y = 0) + P(X = 1, Y = 1) + P(X = 1, Y = 2) = 0.2 + 0.3 + 0.1 = 0.6 \)

- Marginal distribution of Y:
   - \( P(Y = 0) = P(X = 0, Y = 0) + P(X = 1, Y = 0) = 0.1 + 0.2 = 0.3 \)
   - \( P(Y = 1) = P(X = 0, Y = 1) + P(X = 1, Y = 1) = 0.2 + 0.3 = 0.5 \)
   - \( P(Y = 2) = P(X = 0, Y = 2) + P(X = 1, Y = 2) = 0.1 + 0.1 = 0.2 \)

These distributions provide insights into the individual and joint probabilities of the scores of the two students.

---

# Descriptive Statistics

Descriptive statistics summarize and describe the features of a dataset. Common measures include:

- **Mean**: Average value of a dataset, calculated as:

  ​	$$bar{x} = \frac{{\sum_{i=1}^{n} x_i}}{n}$$

- **Weighted Mean**: The weighted mean is the average of a dataset, where each value is multiplied by a corresponding weight and then divided by the sum of the weights.

  $$ \text{Weighted Mean} = \frac{\sum_{i=1}^{n} w_i \cdot x_i}{\sum_{i=1}^{n} w_i} $$

  where$ x_i $ is the$ i^{th} $ value in the dataset,$ w_i $ is the weight corresponding to$ x_i $, and$ n $ is the number of observations.

- **Median**: The middle value of a dataset when it is sorted, or the average of the middle two values if the dataset has an even number of elements.

- **Mode**: Most frequent value(s) in a dataset.

- **Variance**: Measure of the spread of data points around the mean, calculated as:

  ​	$$sigma^2 = \frac{{\sum_{i=1}^{n} (x_i - \bar{x})^2}}{n}$$

- **Standard Deviation**: Square root of the variance.

- **Range**: The difference between the maximum and minimum values in a dataset, providing a simple measure of variability.

  $$ \text{Range} = \text{max}(x) - \text{min}(x) $$

- **Interquartile Range (IQR)**: The range between the first quartile (25th percentile) and the third quartile (75th percentile) of the dataset, which describes the spread of the middle 50% of the data.

  $$ \text{IQR} = Q3 - Q1 $$

- **Skewness**: A measure of the asymmetry of the distribution of data around its mean. Positive skewness indicates a longer right tail, while negative skewness indicates a longer left tail.

  $$ \text{Skewness} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^3}{(n-1) \cdot \sigma^3} $$

- **Kurtosis**: A measure of the "tailedness" of the distribution of data. High kurtosis indicates heavy tails or a sharp peak, while low kurtosis indicates light tails or a flat peak.

  $$ \text{Kurtosis} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^4}{(n-1) \cdot \sigma^4} $$

- **Percentiles**: Values below which a certain percentage of observations fall. Common percentiles include the median (50th percentile), quartiles (25th and 75th percentiles), and deciles (10th and 90th percentiles).

  $$ P_k = \text{Value of } x \text{ below which } k\% \text{ of observations fall} $$

- **Coefficient of Variation (CV)**: The ratio of the standard deviation to the mean, expressed as a percentage. It provides a measure of relative variability, allowing comparison of variability between datasets with different units or scales.

  $$ \text{CV} = \left( \frac{\sigma}{\bar{x}} \right) \times 100\% $$

- **Correlation Coefficient**: A measure of the strength and direction of the linear relationship between two variables. Commonly used correlation coefficients include Pearson's correlation coefficient (for linear relationships) and Spearman's rank correlation coefficient (for monotonic relationships).

  $$ \text{Pearson's correlation coefficient (} \rho \text{)} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}} $$

  $$ \text{Spearman's rank correlation coefficient (} \rho_s \text{)} = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)} $$
  where$ d_i $ is the difference between the ranks of corresponding observations in two variables.

- **Covariance**: A measure of the joint variability of two random variables. It indicates the direction of the linear relationship between variables but is sensitive to the scale of the variables.

  $$ \text{Cov}(X, Y) = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{n-1} $$​

- **Z-Score (Standard Score)**: The z-score measures the number of standard deviations a data point is from the mean of the dataset. It indicates how many standard deviations an observation is above or below the mean.

  $$ Z = \frac{x - \bar{x}}{\sigma} $$

  where$ x $ is the individual data point,$ \bar{x} $ is the mean of the dataset, and $ \sigma $​ is the standard deviation of the dataset.

  -  Example: Compare the variation of a house price relative to its neighbourhood. If we have two houses in different neighbourhoods we can compare the price variation relative to each neighborhood.

# Inferential Statistics



Inferential statistics is the branch of statistics concerned with making inferences or predictions about a population based on sample data. It involves generalizing from a sample to a population, drawing conclusions, and making predictions. Two key techniques in inferential statistics are:

## Hypothesis and A/B Testing

### Statistical Significance

Statistical significance in A/B testing refers to the likelihood that the differences observed between two variations (A and B) are not due to random chance. It helps determine whether the changes made (such as a new website design, a different marketing strategy, etc.) have a meaningful impact. When the difference between variations is statistically significant, it suggests that the observed effect is likely genuine and not simply a result of random variability in the data.

### Null and Alternative Hypotheses, with Examples

- **Null Hypothesis ($H_0$)**: This hypothesis assumes that there is no significant difference between the control (A) and treatment (B) groups in an A/B test. It suggests that any observed difference is due to chance.
  Example: $H_0$: The average time spent on a website is the same for users who see the old design (A) and the new design (B).

- **Alternative Hypothesis ($H_a$)**: This hypothesis contradicts the null hypothesis and suggests that there is a significant difference between the control and treatment groups.
  Example: $H_a$: The average time spent on the website differs between users who see the old design (A) and the new design (B).

### Type I and II Errors

- **Type I Error**: This occurs when the null hypothesis is incorrectly rejected when it is actually true. It represents a false positive result.
- **Type II Error**: This occurs when the null hypothesis is incorrectly accepted when it is actually false. It represents a false negative result.

In the context of A/B testing:

- Type I Error: Incorrectly concluding that there is a significant difference between variations (rejecting $H_0$) when there isn't one.
- Type II Error: Incorrectly concluding that there is no significant difference between variations (failing to reject $H_0$) when there actually is one.

### P-value, Statistical Power, and Confidence Level

- **p-values**:

  In hypothesis testing, the p-value is the probability of obtaining test results at least as extreme as the observed results under the assumption that the null hypothesis is true. A small p-value (typically less than a predetermined significance level, commonly 0.05) indicates strong evidence against the null hypothesis, leading to its rejection. Conversely, a large p-value suggests that the null hypothesis cannot be rejected. P-values provide a measure of the strength of evidence against the null hypothesis and help researchers assess the significance of their findings.

  **Example**:
  Suppose we conduct a hypothesis test to determine if the mean weight of a population is different from 150 pounds based on a sample of 100 individuals. If the calculated p-value is 0.03, we would interpret this as strong evidence against the null hypothesis, suggesting that the population mean weight is likely different from 150 pounds.

- **Statistical Power**: 

  Statistical power is the probability of correctly rejecting the null hypothesis when it is false. It is influenced by factors such as sample size, effect size, and the chosen significance level. Higher statistical power indicates a higher chance of detecting a true effect if it exists in the population. It is crucial in hypothesis testing as it helps researchers assess the sensitivity of their experiments to detect significant effects.

  **Example**: In a clinical trial testing the efficacy of a new drug, statistical power determines the likelihood of detecting a true difference in outcomes between the treatment and control groups. A study with low statistical power may fail to detect a significant effect even if the treatment truly has an impact, leading to false conclusions. Therefore, researchers often strive to achieve sufficient statistical power to increase the reliability of their findings.

- **Confidence Intervals**:

  Confidence intervals provide a range of plausible values for a population parameter, along with a level of confidence associated with that interval. They are calculated using sample data and are used to estimate the precision of an estimate. For example, a 95% confidence interval for the population mean represents the range of values within which we are 95% confident that the true population mean lies. Confidence intervals provide valuable information about the uncertainty associated with sample estimates and help researchers assess the reliability and precision of their findings.

  **Example**:
  If we calculate a 95% confidence interval for the mean height of a population to be ($65$ inches, $70$​ inches), we interpret this as follows: we are 95% confident that the true mean height of the population falls within this interval.

These measures are calculated based on the characteristics of the sample data, effect size, and the chosen level of confidence or significance. They help researchers assess the reliability and significance of their findings in A/B testing and hypothesis testing scenarios.

### **Central Limit Theorem (CLT)**

The Central Limit Theorem states that the sampling distribution of the sample mean approaches a normal distribution as the sample size increases, regardless of the shape of the population distribution. This theorem is fundamental in inferential statistics because it allows us to make inferences about population parameters based on sample means, even when the population distribution is non-normal. The CLT is widely used in hypothesis testing, confidence interval estimation, and other statistical analyses to justify the use of normal distribution-based methods.

The Central Limit Theorem states that when independent random variables are added, their properly normalized sum tends toward a normal distribution (commonly known as a bell curve), even if the original variables themselves are not normally distributed. In the context of sampling from a population:

1. The mean of sample means is equal to the population mean.

2. If the population is normally distributed, then the sample means will be normally distributed.

3. If the population is not normally distributed, but the sample size is greater than 30, the sample means will still roughly form a normal distribution.

4. The standard deviation of the sample means equals the population standard deviation divided by the square root of the sample size:

   $$ \text{sample\_standard\_deviation} = \frac{\text{population\_standard\_deviation}}{\sqrt{\text{sample\_size}}} $$ 

**Math Example**:

Suppose we have a population with a skewed distribution, such as an exponential distribution. We take multiple samples of size \( n \) from this population and calculate the sample means for each sample. According to the Central Limit Theorem, as the sample size \( n \) increases, the distribution of these sample means will approach a normal distribution. This allows us to use normal distribution-based methods for inference, such as calculating confidence intervals or performing hypothesis tests, even though the population distribution is not normal.