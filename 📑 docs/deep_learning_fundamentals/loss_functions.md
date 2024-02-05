# Loss Functions




When training a machine learning model, we try to find the parameters that produce the the best possible mapping form the input to the output. 

To get the best possible mapping we need a training dataset ${X_i, Y_i}$ of input/output pairs. A *loss function* or *cost_function* $L[\phi]$  that returns a single number that describes the mismatch between the model predictions $ f[x_i,\phi]$  and their corresponign groud-thruth  outputs $y_1$. 

> The goal of train is to find parameters values for $\phi$ that minimize the loss and hence map the training inputs to the outputs as closely as possible. 



## Maximum likelihood

Considering a model $f[x,\phi]$ with parameters $\phi$ that computes an ouput from a input x. Instead of assum that the model directly predicts compute a prediction $y$, consider that the model compute a conditional proability distribuition $Pr(y|x)$ over possible output $y$ given input $x$ . The loss encourages each training output $y_i$ to have high probablility unde the distribuition  $Pr(y_i, x_i)$ computed from the corresponding $x_i$ .

### Computing distribution over outputs

But how can the model $f[x,\phi]$ be adapted to comput a probabilty distribution?

 - 1st we chosse a **parametric distribution** defined on the output domain $y$ 
 - Then we use the network to compute one or more of the parameters $\theta$ of this distribution



















































