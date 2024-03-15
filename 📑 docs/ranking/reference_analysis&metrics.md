### Spearman Correlation Coefficient

The Spearman correlation coefficient, denoted by $ρ$ (rho), is a statistical measure that quantifies the strength and direction of association between two ranked variables. Unlike the Pearson correlation coefficient, which assesses linear relationships between variables, Spearman correlation assesses monotonic relationships. Monotonic relationships are those where the variables tend to change together but not necessarily at a constant rate.

### Formula

The formula to compute Spearman correlation coefficient is:

$$
ρ = 1 - \left[\frac{6 \sum{d^2}}{n(n^2 - 1)}\right]
$$

Where:
- $ρ$ is the Spearman correlation coefficient
- $\sum{d^2}$ is the sum of squared differences between the ranks of corresponding variables
- $n$ is the number of observations

### Example

Let's consider an example where we want to examine the relationship between the hours of study and the grades obtained by a group of students. Here's the data:

| Hours of Study (X) | Grades (Y) |
| ------------------ | ---------- |
| 5                  | 70         |
| 8                  | 85         |
| 3                  | 60         |
| 7                  | 80         |
| 6                  | 75         |

First, we need to rank the data:

| Hours of Study (X) | Grades (Y) | Rank(X) | Rank(Y) |
| ------------------ | ---------- | ------- | ------- |
| 5                  | 70         | 3       | 3       |
| 8                  | 85         | 5       | 5       |
| 3                  | 60         | 1       | 1       |
| 7                  | 80         | 4       | 4       |
| 6                  | 75         | 2       | 2       |

Now, we compute the differences between ranks:

| Hours of Study (X) | Grades (Y) | Rank(X) | Rank(Y) | d (Difference) | d² (Squared Difference) |
| ------------------ | ---------- | ------- | ------- | -------------- | ----------------------- |
| 5                  | 70         | 3       | 3       | 0              | 0                       |
| 8                  | 85         | 5       | 5       | 0              | 0                       |
| 3                  | 60         | 1       | 1       | 0              | 0                       |
| 7                  | 80         | 4       | 4       | 0              | 0                       |
| 6                  | 75         | 2       | 2       | 0              | 0                       |

Now, we sum the squared differences:

$$
\sum{d^2} = 0 + 0 + 0 + 0 + 0 = 0
$$

Finally, we plug the values into the Spearman correlation coefficient formula:

$$
ρ = 1 - \left[\frac{6 \times 0}{5(5^2 - 1)}\right] \\
ρ = 1 - \left[\frac{0}{120}\right] \\
ρ = 1 - 0 \\
ρ = 1
$$

### Interpretation

The Spearman correlation coefficient $ρ$ is 1, indicating a perfect monotonic relationship between hours of study and grades. This means that as the hours of study increase, the grades obtained also increase monotonically.

This is a simple example, and in real-world scenarios, Spearman correlation coefficient is used to analyze various types of data to understand the strength and direction of the relationship between variables.

---

### Normalized Discounted Cumulative Gain (NDCG)

Normalized Discounted Cumulative Gain (NDCG) is a metric commonly used in information retrieval and machine learning to evaluate the quality of a ranking algorithm's results. It measures the effectiveness of the ranking by considering both the relevance and the position of each item in the ranked list.

### Formula

The formula to compute NDCG is as follows:

$$
NDCG@k = \frac{DCG@k}{IDCG@k}
$$

Where:
- $NDCG@k$ is the Normalized Discounted Cumulative Gain at position $k$
- $DCG@k$ is the Discounted Cumulative Gain at position $k$
- $IDCG@k$ is the Ideal Discounted Cumulative Gain at position $k$

Discounted Cumulative Gain at position $k$ ($DCG@k$) is calculated as:

$$
DCG@k = rel_1 + \sum_{i=2}^{k} \frac{rel_i}{\log_2(i)}
$$

Ideal Discounted Cumulative Gain at position $k$ ($IDCG@k$) is the DCG that would be achieved if the documents were perfectly ranked according to relevance. It is computed by sorting the relevance scores in descending order and calculating DCG@k based on that order.

### Example

Let's consider an example where we have a ranked list of documents with their relevance scores:

| Rank | Relevance Score |
| ---- | --------------- |
| 1    | 3               |
| 2    | 2               |
| 3    | 3               |
| 4    | 0               |
| 5    | 1               |

We want to compute NDCG@3, which means we're considering the top 3 documents.

First, we compute $DCG@3$:

$$
DCG@3 = 3 + \frac{2}{\log_2(3)} + \frac{3}{\log_2(4)} = 3 + \frac{2}{1.585} + \frac{3}{2} \approx 3 + 1.26 + 1.5 = 5.76
$$

Next, we compute $IDCG@3$. We sort the relevance scores in descending order:

| Rank | Relevance Score |
| ---- | --------------- |
| 1    | 3               |
| 2    | 3               |
| 3    | 2               |

$$
IDCG@3 = 3 + \frac{3}{\log_2(2)} + \frac{2}{\log_2(3)} = 3 + \frac{3}{1} + \frac{2}{1.585} \approx 3 + 3 + 1.26 = 7.26
$$

Finally, we compute $NDCG@3$:

$$
NDCG@3 = \frac{DCG@3}{IDCG@3} = \frac{5.76}{7.26} \approx 0.793
$$

### Interpretation

The NDCG@3 value is approximately 0.793, indicating the effectiveness of the ranking algorithm in retrieving relevant documents within the top 3 positions. A higher NDCG value suggests a better ranking quality.

NDCG is widely used in evaluating search engines, recommendation systems, and other ranking-based applications.