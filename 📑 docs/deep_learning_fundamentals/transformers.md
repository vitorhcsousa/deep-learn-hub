

# Transformers

### Title: Unveiling the Power of Transformers: Revolutionizing Natural Language Processing

### Introduction:

- Briefly introduce the concept of transformers and their significance in the field of Natural Language Processing (NLP).
- Highlight the transformative impact of transformers on various applications, from language translation to text generation.

---

## Notes from Transformer with Lucas Beyer



1st paper https://arxiv.org/pdf/1409.0473v7.pdf



Attention is All you need as main reference

### Attention Mechanism

Attention is a funciton similar to a soft $kv$ dictionary lookup

- We have

  - Query $q$ ( a vector of floats)
  - Keys $k$ (vector of floats)
    - each key has a corresponding value
  - Values $v$ (vector of floats)

  1. Attettion weights $a_{1:N}$ are query-key similarities 
     $$
     â_i = q \cdot k_i \\
     \\
     
     normalized\ via\ softmax: a_i = \frac{e^{âi}}{\sum_j e^{âj}}
     $$

- 

2. Outpuz $z$ is attention-weighted average of values $v_{1:N}$
   $$
   z = \sum_iâ_iv_i = â \cdot v
   $$

3. Usually, $k$ annd $v$ are devired from the same inpu $x$ 

$$
k = W_k \cdot x 		\\
v = W_v \cdot x
$$

​	The quert $ q$ can com from a separate input $y$
$$
q = W_q \cdot y
$$
​	Or from the same input $x$ ! The we call it ***self attention***
$$
q = W_x \cdot x
$$


 <img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240127172312775.png" alt="image-20240127172312775" style="zoom:50%;" />

1. In many cases we use many queries $q_{1_M}$, not just one. Stacking them leads to the Attention Matrix $A_{1:N, 1:M}$ and the subsquently to many outputs:

$$
z_{1:M} = Attn(q_{1:M}, x)= [Attn(q_2,x)|...|Attn(q_M,x)]
$$

2. We usuallye use “multi-head” attention. This mens the operation is repeated K times and the resulst are concatenated along the feature dimension. $Ws$ differ.
   $$
   z_i = \left( \begin{array}{c}
     \text{Attn}_1(q_i, x) \\
     \text{Attn}_2(q_i, x) \\
     \vdots \\
     \text{Attn}_K(q_i, x)
   \end{array} \right)
   $$
   

3. The most commonly seen formulation:
   $$
   z = \frac{softmax(QK')}{\sqrt{d_{key}}V}
   $$
   Note that the complexity is $O(N^2)$​

### Transformer Architecture

- **Input (Tokenization and) and Embedding**: Convert text into a sequence of float vectors

  1. Tokenization: Convert text into pieces, can be charaters, word, “tokens” 
  2. Token are indices into the vocabulary 
     1. convert text into numbers
  3. Reach vocab entry corresponde to a learned $d_{model}$- dimensional vector
     1. Learn the embedding from the vocabulary 
     2. Randomly initialize

- **Positional Encoding**: Pass to model the order of things

  - Remember attention is permutationinvariant, but languande is not. Need to encode position of each word, just add something 

- **Multi-head Self-Attention**

  - Meaning the **input sequence is used to create the** **queries, keys and the values**
  - Each token can “look arround” the whole input, and decide how to update its representation based on what is sees

- **Point-wise MLP** (Feed Foward block)

  - A simple MLP applied to each token individually and indecently 
    $$
    z_i = W_2 GeLU(W_1x+b_1)+b_2
    $$

  - Think of it as each token pondering for itseltf about what it has observed previously. There’s some weak evidence this is where world knowledge is stored

  - Where most of paramenters are “stored”

- **Residual connections** (original from ResNet)

  - Each module’s output has the exact same same as the input. 

  - Folllwogin ResNest, the module conputes a residual instaed of new value:
    $$
    z_i = Module(x_i)+x_i
    $$

  - This was shoen to dramatically improve trainability 

  - Have a $x$ process $x$ on a math black and after update the $x$ based on the output of that model

- LayerNorm

  - Normalization also dramatically improve trainability. 
  - The post norm (original)
    - $z_i=LN(Module(x_i)+x_i)$​
  - And the pre-norm
    - $z_i=$​

  > Until now, all this components create the **Enconding/Encoder** 
  > Sice input and ouput shapes are identical we can stack N such blocks. Typically, N=6 (“base”), N=12(“large”) or more.
  >
  > Encoder output is “heavily processed” (think: “high level, contextualised”) version of the inputs tokens, i.e. sequence.

Decoding/ The decoder (alternatively Generationg /the Generator)

- What we want to model $p(z|x)$

  - for example, in translation : $p(z| \" the\ detective\ investigated\")  \forall z$​

- Seems impossible at first, byt we can exactly decompose into tokens:
  $$
  p(z|x) = p(z_1|x)p(z_2|z_1,x)p(z_3|z_2,x)
  $$

- Meaning, we can generate the answer one token at a time. Each p is full pass throught the model and take into consideration the previous ouptus 

- For generation $p(z_3|z_2,z_1, x)$:

  - $x$ comes  from the encoder 
  - $z_1, z_2$ is what we have predicted so far, goes into decoder.

- Once we have $p(z|x)$ we still need to actually sampe a sentence such as “le détective a enquête”. Many strategies, greedy, beam-search…

- **Mask Selft attention**

  - This is regular self-attention as in encoder, to process what’s been decoded so far, but with a trick
  - If we had to train on one single $p(z_3|z_2,z_1, x)$ at a time:SLOW
  - Instead, train on all $p(z_i|z_{1:i}, x)$ simultaneously
  - How? In the attention weights for $z_i$, set all entries $i:N$ to 0. 
    - This way, each token only sees the already generated ones
  - At generatarion time there is no such trick. We need to generate on $z_i$ at a time. This is why autoregressive decoding is extremely slow

- **Cross attention**

  - Each decoded token can “look at” the encoder’s output:

  $$
  Attn(q=W_qx_{dec}, k=W_kx_{enc}, v=W_vx_{enc})
  $$

  - This is the same as in the 2014 paper
  - This is where $|x$ in $p(z_3|z_2,z_1,x)$ come from

- **Feedfoward and stack layers**

- **Output layers**

  - Assume we have already generated K tokens, generate the next one
  - The decoder was used to gather all information necessary to predict a probability distribution to the next token (K), over the whole vocab
  - Simple:
    - ***Linear projection of token K*** 
    - ***Softmax normalization***





- Works very well with a lot of data
- Very efficient 
- Very prone to overfitting with few data
-  

---

# NLP with Transformers

## Architecture 

Transformer is based on the *encoder-decoder* architecture that is widely used for tasks like machine transalations, where a sequence of word is translated from one languange to another. The architecture consists of two main components:

- Encoder
  - Converts an input sequence of tokens into a sequence of embedding vectors often called the *hidden state* or *context*.

- Decoder
  - Uses the encoder’s hidden state to iteratively generate an output sequence of tokens, on token at a time.

***Main things that characterize the Transformers:***

- The input text is tokenized and converted to *token embeddings*. Sice the attention mechanisms is not aware of the relative positions of the tokens, we need a way to inject some informationabout token positions into the input to model the sequential nature of text. The token embeddings are thus combined with *positional embeddings* that contain positional information for each token. 
- The encoder is composed of stack of *encoder layers* or “blocks”, which is analogous to stacking convolutional layers in computer vision. The same is true if the decoder, which has its own stack of *decoder layers*
- The encoder’s output is def to each decoder layer, and the decoder then generates a predcition for the most probable next tolen in the sequence. The output of this step is then fed 







---

- 


### 1. Understanding the Basics:

#### 1.1 What are Transformers?

- Explain the fundamental idea behind transformers in NLP.
- Mention the shift from traditional sequence models to the transformer architecture.

#### 1.2 Components of Transformers:

- Discuss key components like self-attention mechanism, feedforward neural networks, and layer normalization.
- Illustrate how these components work together to process information in parallel.

### 2. Self-Attention Mechanism:

#### 2.1 How Does Self-Attention Work?

- Explain the concept of attention and its role in capturing relationships between words in a sequence.
- Describe the self-attention mechanism's ability to assign different weights to different words.

#### 2.2 Benefits of Self-Attention:

- Discuss how self-attention improves the model's ability to capture long-range dependencies in sequences.
- Highlight the efficiency of parallel processing in self-attention.

### 3. Transformer Architecture:

#### 3.1 Encoder-Decoder Structure:

- Introduce the encoder-decoder architecture and its role in tasks like machine translation.
- Explain how the encoder processes input sequences, and the decoder generates output sequences.

#### 3.2 Multi-Head Attention:

- Describe the concept of multi-head attention and its role in capturing different types of relationships in parallel.

### 4. Pre-training and Fine-tuning:

#### 4.1 Pre-training with Language Models:

- Explain the pre-training phase using large language models like BERT and GPT.
- Discuss how transformers learn contextualized representations during pre-training.

#### 4.2 Fine-tuning for Specific Tasks:

- Illustrate how pre-trained models can be fine-tuned for specific NLP tasks.
- Highlight the advantages of transfer learning in NLP using pre-trained transformers.

### 5. Applications:

#### 5.1 Natural Language Understanding:

- Explore how transformers have revolutionized tasks like sentiment analysis, named entity recognition, and text classification.

#### 5.2 Language Generation:

- Discuss the role of transformers in language generation tasks, including text completion and story generation.

#### 5.3 Machine Translation:

- Highlight the success of transformers in improving the accuracy of machine translation systems.

### Conclusion:

- Summarize the key points discussed in the blog post.
- Emphasize the transformative impact of transformers on NLP and the broader field of artificial intelligence.

### References:

- Include citations to relevant papers, articles, and resources for readers interested in exploring further.

Remember to use clear and concise language, provide visual aids if necessary, and engage your audience with examples and real-world applications.





