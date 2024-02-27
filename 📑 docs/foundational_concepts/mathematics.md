

### Basics

***Exponents*** -> multiply the number by itself a specificed number of times $2^3 = 2*2*2 = 8$ 

***Logarithms*** -> Is a math function that finds the power of a specific number and base. “2 raised to what power gives me 8?” $log_28= x; x = 3$ 

***Derivatives*** -> Tells the slope of a function, that measures the rate of change at any point in a function $\frac{d}{dx}$  indicates the derivative for x.  If we have $f(x)=x^2$ then 

$\frac{d}{dx}f(x) = \frac{d}{dx}x^2 = 2x \rightarrow \frac{d}{dx}f(2)=2(2)=4$

***Chain Rule*** -> The chain rule is a fundamental concept in calculus that allows us to find the derivative of composite functions. If we have a function composed of two or more functions, such as $f(g(x))$, where $f$ and $g$ are functions, then the chain rule states that the derivative of this composite function is the derivative of the outer function evaluated at the inner function, multiplied by the derivative of the inner function. Symbolically, if $y = f(u)$ and $u = g(x)$, then the chain rule is expressed as $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$.

For example, if we have $f(x) = (x^2 + 1)^3$, then $f'(x)$ can be found using the chain rule. Letting $u = x^2 + 1$, we find $\frac{df}{dx} = \frac{df}{du} \cdot \frac{du}{dx}$. Differentiating $u^3$ to $u$ gives $3u^2$, and differentiating $x^2 + 1$ concerning $x$ gives $2x$. Therefore, $f'(x) = 3(x^2 + 1)^2 \cdot 2x$.

***Integrals*** -> Integrals are the reverse operation of derivatives. They compute the area under the curve of a function over a given interval. The definite integral of a function $f(x)$ from $a$ to $b$ is denoted as $\int_{a}^{b} f(x) \, dx$, which represents the net signed area between the x-axis and the curve $f(x)$ from $x = a$ to $x = b$. The indefinite integral, or antiderivative, is denoted as $\int f(x) \, dx$, and it represents a family of functions whose derivative is $f(x)$.

For example, if we have $f(x) = 2x$ and we want to find the definite integral of $f(x)$ from $1$ to $3$, we would calculate $\int_{1}^{3} 2x \, dx$. This would give us the area under the curve of $f(x) = 2x$ from $x = 1$ to $x = 3$.

If we want to find the indefinite integral of $f(x) = 2x$, we would calculate $\int 2x \, dx$, which gives us $x^2 + C$, where $C$ is the constant of integration.

---

# Probability 



















---

***ROC (Receiver Operating Characteristic) Curve***:
The ROC curve is a graphical representation of the performance of a binary classification model. It illustrates the trade-off between its true positive rate (sensitivity) and false positive rate (1 - specificity) as the discrimination threshold is varied.

- **True Positive Rate (TPR)**, also known as sensitivity, measures the proportion of actual positive cases that are correctly identified as positive. It can be mathematically expressed as:

    $TPR = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} $

- **False Positive Rate (FPR)** measures the proportion of actual negative cases that are incorrectly identified as positive. It can be mathematically expressed as:

    $FPR = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}} $

The ROC curve plots TPR against FPR at various threshold settings, providing insight into the model's performance across different levels of sensitivity and specificity.

***Area Under the Curve (AUC)***:
AUC quantifies the overall performance of a binary classification model represented by the ROC curve. It is a single scalar value that ranges from 0 to 1, where higher values indicate better performance.

- AUC = 1 implies a perfect classifier that achieves a TPR of 1 (sensitivity) and an FPR of 0 (specificity) across all thresholds.
- AUC = 0.5 suggests a random classifier that performs no better than chance.

In summary, the ROC curve visually represents how well a model can distinguish between the positive and negative classes at different threshold settings, while the AUC provides a single metric summarizing the model's overall performance in classification.