<<<<<<< HEAD
# Loss functions

When we train deep learning models the goal is to find the best parameters $\phi$  that produce the best mapping from the input $x$ to output $y$.

Having a training dataset ${x_i, y_i}$ of input/output pairs the **loss function** or **cost function** $L[\phi]$ returns a signle number that describes the mismatch between the model predictions $f[x_i,\phi]$ and the ground-truth $y_i$

## Maximum Likelihood

Instead of assuming that the model directly predicts a predictions $y$ consider that the model computes a **conditional probability distribution $Pr(y|x)$** over possible $y$ given input $x$. The loss encourages each training output $y_i$ to have a high probability of distribution $Pr(y_i|x_i)$.

### Computing distribution over outputs 

How can a model $f[x,\phi]$ be adapted to compute a probability distribution? 

1. Choose a parametrics distribution $Pr(y|\theta)$ defined on the output domain 
2. Use the network to compute one or more of the parameters $\theta$ of this distribution

### Maximum Likelihood Criterion

The models compute different distributions parameters $\theta = f[x_i, \phi]$ for each training example. 
Each training output $y_i$ should have a high probability under its corresponding distribution $Pr(y_i|\theta_i)$. We choose parameters $\phi$ so that they maximize the combined probability across all $I$ training examples:
$$
\hat{\phi} = \operatorname*{argmax}_{\phi} \Bigg[\prod_{i=1}^{I}Pr(y_|x_i)\Bigg]\\
= \operatorname*{argmax}_{\phi} \Bigg[\prod_{i=1}^{I}Pr(y_|\theta_i)\Bigg]\\
= \operatorname*{argmax}_{\phi} \Bigg[\prod_{i=1}^{I}Pr(y_|f[x_i,\phi])\Bigg]\\
$$
The combinaed probability term is the *likelihood* of parameters and is known as the ***maximum likelihood criterion***.

Here we make two assumptions:

1. Data is identically distributed
2. Conditional distribution $Pr(y_i,x_i)$ of the output given the input are independent; in other words, we assume that data is *independent and identically distributed* 

### Maximising log-likelihood

The maximum likelihood criterion is not very practical. Each term of $Pr(y_i[x_i,\phi])$ can be small so the product of many of these terms can be tiny. Fortunately, equivalently maximise the logarithm of the likelihood:
$$
\hat{\phi} = \operatorname*{argmax}_{\phi} \Bigg[\prod_{i=1}^{I}Pr(y_|f[x_i,\phi])\Bigg]\\
= \operatorname*{argmax}_{\phi}  \Bigg[\log\Big[\prod_{i=1}^{I}Pr(y_|f[x_i,\phi])\Big]\Bigg]\\
= \operatorname*{argmax}_{\phi}  \Bigg[\sum_{i=1}^I\log\Big[Pr(y_|f[x_i,\phi])\Big]\Bigg]\\
$$
This *log-likelihood* criterion is equivalent because the logarithm is a monotonically increasing function: if $z>z'$  then $log[z]>log[z']$​.

### Minimising negative log-likelihood 

> Model fitting problems are framed in terms of minimizing a loss.

 To convert the maximum log-likelihood criterion to a minimization we multiply by minus one, which gives us the *negative log-likelihood criterion:*
$$
\hat{y}= \operatorname*{argmin}_{\phi} \Bigg[-\sum_{i=1}^I \log\Big[Pr(y_i| f[x_i,\phi]) \Big]  \Bigg]\\
= \operatorname*{argmin}_{\phi} \Big[L[\phi]\Big]
$$


### Inference 

The network no longer directly predicts the outputs $y$ but instead determines the probability distribution $y$. When we perform inference, we often want a point estimate rather than a distribution, so we return the maximum of the distribution:
$$
\hat{y} = \operatorname*{argmax}_{y} \Big[Pr(y|f[x, \hat{\phi}])\Big]
$$

## Recipe for Construction Loss Functions

The recipe for construction loss functions for training data ${x_i, y_i}$ using maximum likelihood approach is hence:

1. Choose a suitable probability distribution $Pr(y| \theta)$ defined over the domain of the predictions y with distribution parameters $\theta$ 

2. Set the machine learning model $f[x,\phi ]$ to prediction one or more of these parameters so $\theta = f[x,\phi]$ and $Pr(y|\theta)= Pr(y|f[x,\phi])$

3. Train the model, find the network parameters $\hat{\phi}$ that minimize the negative log-likelihood loss function over the training dataset pairs ${x_i, y_i}$ :  
   $$
   \hat{\phi} = \operatorname*{argmin}_{\phi} \Big[L[\phi] \Big] = \operatorname*{argmin}_{\phi} \Bigg[ - \sum_{i=1}^I \log\Big[Pr(y_i|f[x_i,\phi])\Big] \Bigg]
   $$

4. To perform inference for a new test examined $x$ return either the full distribution either full distribution $Pr(y|f[x, \hat{\phi}])$ or the maximum of this distribution

