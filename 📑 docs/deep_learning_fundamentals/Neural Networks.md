---
tags:
  - nn
  - layers
  - hidden_layer
  - neurons
  - activations
  - mlp
  - shallow_nn
  - deep_nn
  - weights
  - biases
  - bias
  - offset
  - slope_param
  - neural_network
  - universal_approximation_theorem
---
# Neural Networks

### Common Terminology 

Neural networks sometimes are referred to in in terms of layers. The left is the *input layer*, the center is the *hidden layer* and the right is the *output layer*. The hidden layer contain x *hidden units*, that sometimes are referred as *neurons*.
When er pass data through the network , the vales of the inputs to the hidden layer are termed *pre-activations*. The values at hidden layers are termed *activations*. 

Any layer with at leat one hidden layer is also called *multi-layer perceptron* of *MLP* for short. Networks with one hidden layer are sometimes referred as *shallow neural networks*. Networks with multiple hidden layers are referred to as *Deep Neural Networks*. 

Neural networks in which the connections form an acyclic graph are referred as *feed-forward networks*. 
If every element in one layer connects to every element in the next, the network is *fully connected*. These connections represent slope parameters in the underlying equations and are referred to as network *weights*. The offset parameters are called *biases*. 

![image-20240121112330022](../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240121112330022.png)

### Neural Network Example

Neural Networks (this example is a shallow neural network) are functions $y=f[x,\phi]$ with parameters $\phi$ that map multivariate inputs $x$ to multivariate outputs $y$. 
Assuming that in this example the network $f[x, \phi]$ that maps a scalar input $x$ to a scalar output $y$ and has ten parameters $ \phi = [\phi_0, \phi_1,\phi_2, \phi_3, \theta_{10}, \theta_{11}, \theta_{20}, , \theta_{21}, \theta_{30}, \theta_{31}]$​ 
$$
y = f[x,\phi] \\
= \phi_0, + \phi_1a[\theta_{10}+\theta_{11}x]+ \phi_2a[\theta_{20}+\theta_{21}x]+ \phi_3a[\theta_{30}+\theta_{31}x]
$$


## Shallow Neural Networks

Shallow neural networks have one hidden layer. They compute several linear functions of the input, pass each result throught an activation function, and then take a linear combination of these activations to form the outputs. Shallow neural networks make predictions $y$ based on inputs $x$ by dividing the input space into a continuous surface of piecewise linear regions. With enough hidden units (neuros), shallow neural networks can approximate any continuous functions to arbitary precision. 

In general we can say that a shallow neural network $y=f[x, \phi]$ that maps a multidimensional input $x \in \mathbb{R}^{D_1}$ to a multi-dimensional output $y \in \mathbb{R}^{D_o}$ using $h \in \mathbb{R}^D$  hidden units. Each hidden unit is computed as:
$$
h_d = a[\theta_{d0} + \sum_{i=1}^{Di}{\theta_{di}x_i}]
$$
and these are combined linearly to create the output:
$$
y_j =\phi_{j0} +\sum_{d=1}^D{\phi}_{jd}h_d
$$

## Neural Networks intuition

----

Activation functions introduce non-linearity to the network, allowing it to learn complex patterns and relationships in the data. Piecewise  linear activation functions are a specific type of activation function  that consists of linear segments over different intervals.

One common example of a piecewise linear activation function is the Rectified Linear Unit (ReLU). The standard ReLU is defined as:

$f(x)=max⁡(0,x)$

This can be expressed as a piecewise linear function:
$$
\begin{cases}
0 & \text{if } x < 0 \\
x & \text{if } x \geq 0 \\
\end{cases}
$$


In this case, the ReLU activation function is linear for positive input values and zero for negative input values. It introduces non-linearity by "turning on" for positive inputs, allowing the network to learn complex representations.

Other piecewise linear activation functions, such as Leaky ReLU or Parametric ReLU, introduce a small slope for negative input values, providing a slight non-linearity even for negative inputs.

The use of piecewise linear activation functions in deep learning is crucial for enabling the training of deep neural networks. The non-linearities introduced by these functions allow the network to approximate more complex functions and capture intricate patterns in the data. Additionally, piecewise linear activation functions often help mitigate the vanishing gradient problem, which can occur with purely linear activations and make training difficult.
