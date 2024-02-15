---
tags:
  - normalization
  - nn
  - transformations
  - batch_norm
  - dl_model_components
  - rms_norm
---
> **Normalization is a technique used to change the values of the numeric columns in the dataset to use a common scale, without distorting differences in the ranges of values of losing information. Enforce the empirical mean and variance of groups of activation *(Francois Fleuret)*.**

In context of neural networks, normalization helps in stabilizing the learning process and dramatically reduces the number of training epochs required to train deep networks. It can **help mitigate problems such as exploding and vanishing gradients**, where the values of the gradientes become either too large or too small. 


## BatchNormalization
Batch normalization shifts and rescales each activation $h$ so that its mean and variance across the batch $\beta$ becomes values that are learned during training. First, the empirical mean $m_h$ and the standard deviation $s_h$ are computed:
$$
m_h = \frac{1}{|\beta|} \sum_{1\in\beta}h_i
$$
$$
s_h = \sqrt{ \frac{1}{|\beta|}\sum_{1\in\beta}(h_i -m_h)^2}
$$

Then we use these statistics to standardize the batch activations to have mean zero and a unit variant:
$$
h_i \leftarrow \frac{h_i -m_h}{s_h + \epsilon}    \forall i \in \beta
$$
where $epsilon$ is a small number that prevents division by zero if $h_1$ is the same for every member of the batch and $s_h$ =0.

Finally, the normalized variable is scaled by $\gamma$ and $\delta$ 
$$
h_1 \leftarrow \gamma h_i + \delta
$$

After this operations, the activations have mean $\delta$ and standard deviation $\gamma$ across all members of the batch. Both these quantities are learned during training.

> **Batch normalization is applied independently to each hidden units.**

In standard neural network with $K$ layers, each containing $D$ hidden units, there would be $KD$  learned offsets $\delta$ and $KD$ learned scales $\gamma$. 
In convolution networks, the normalizing statistics are computed over both the batch and the spatial position. If there were $K$ layers, each containing $c$ channels, there would be the $KC$ offsets and $KC$ scales. ** **

> **At tests time, we do not have a batch from which we can gather statistics. To resolve this, the statistics $m_h$ and $s_h$ are calculated across then whole training dataset(rather than just a batch) and frozen in the final network.**



![[batch_norm.png|Image from Understanding Deep Learning Book |500]]

### Costa and Benefits of batch Normalization
Batch normalization makes the network invariant to rescaling the weights and biases that contribute to each activation; If these are double, then the activation also double, the estimated standard deviation $s_H$ doubles and the normalization and the normalization in equation 3 compesates for theses changes. This happens separately for each hidden unit. Consequently, there will be a large family of weights and biases that all produce the same effect. Batch normalization also adds two parameters, $\gamma$ and $\delta$ at every hidden unit, which makes the model somewhat larger. Hence, it both create redundancy in the weight parameters and adds extra parameters to compensate for that redundancy. This is inefficient.
But the batch normalization also provides several benefits:
#### Stable forward Propagation
If 

#### Higher Learning Rates

#### Regularization



## RMSNorm (Root Mean Square Normalization)
- Compute the root mean square of the input tensor and uses it to normalize the tensor. This is done to ensure that the inputs to the layers of the model have a stable distribution, which can help improve the learning process.