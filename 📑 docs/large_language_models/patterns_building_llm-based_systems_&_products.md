# Intro LLM 

https://www.youtube.com/watch?v=zjkBMFhNj_g

https://drive.google.com/file/d/1pxx_ZI7O-Nwl7ZLNk5hI3WzAsTLwvNU7/view?pli=1

***LLM open source -> basically is just two files parameters file / file to run the neural network. No need to internet*** 

## Train 

- How do we get the parameters? The magic of LLMs
  - Training  is more involved, its like compressing the internet (ex 10TB of data); we need a GPUs (a lot)  and let training for some days (eg Llama 2 70B took 6000GPUs and 12 days, costing ~$2M generating a parameters file with 140GB)
- Neural Network Training (To predict the next work in the sequence)
  - In the task of predicting the next word during the  train the model is *forced* to learn about the world (personalities, evens, shows…) and that knowledge is compressed in the weights 



## How they work?

Using Transformer NN architecture- we know the math computations, we can measure that is performance but we don´t know how they are interact. 

Litter is know in full detail…

- Billion of parameters are dispersed through the network
- We know how to iteratively adjust them to make it better at prediction
- We can measure that this works, but we don´t really known how the billions of parameters collaborate to do it 
- e.g. the reversal curse with the Tom Cruise mother question

>  **LLMs are mostly inscrutable artifacts, develop correspondingly sophisticated evaluations** 

The filed of interpretability is trying to explore this in a deeper level 



## Fine Tuning into a Assistant 

we don´t want a document generator. We, generally, want give equations and e want answers to that. 

- We train a model with labeled documents like question&answers  + quality -quantity
- Pre-training stage - Train on a lots of data: adds the knowledge to the model; This step ***Fine Tuning*** is like alignment to the goal that we want, in this case an assistant. 

##### How to train our ChatGPT

###### Stage 1: Pretraining  (every month)

1. Download tons of data from internet (text)
2. Get a cluster of XXXX GPUs
3. Compress the text into a neural network, pay $MM and wait Xx days/weeks
4. with this we get the **base model**

##### Stage 2: Finetuning (every week)

1. Write labeling instructions 
2. Collect 100k(per example) high quality Q&A responses, and/or comparisons
   1. A second kind of label: comparisons - it is often muche easier to compare Answers instead of writing Answers (reinforment learning from humam feedback)
3. Fine tune base model based on this data (~1/day)
4. With that we obtain the assistant model 
5. Run a lot of evaluations 
6. Deploy
7. Monitor, collect misbehaviors (go to step 1)  

### LLM Leaderboard from Chatbot Arena

https://chat.lmsys.org/

---

# LLM Scaling Laws

Performance of LLMs is a smooth, well-behaved, predictable function of:

- **N** the number of parameters in the network
- **D** the amount of text we train on 

And trend do no show signs of topping out

> We can expect mode intelligence for free by scalling 

<img src="/home/vitor.costasousa/Documents/Personal/deep-learn-hub/📑 docs/large_language_models/assets/image-20240308140756577.png" alt="image-20240308140756577" style="zoom:50%;" />

with more data we can see that the perfomance improves

<img src="/home/vitor.costasousa/Documents/Personal/deep-learn-hub/📑 docs/large_language_models/assets/image-20240308140845163.png" alt="image-20240308140845163" style="zoom: 50%;" />



---

## Thinking, System 1/2

Currently LLMs only work on system 1 (of book think fast & slow), its automatic almost instinctive. A lot of people are wondering if these kinds of systems can think about the problem/question that we present to them (like the system 2). For that we probably need to convert time in accuracy, *here my question take 10mins to think and answer*. 

### Self Improvement

#alphaGo

AlphaGo had to major stages:

- Learn by imitating expert human players
- Learn by self-improvement(reward + win the game) 

# Patterns for Building LLM-based Systems & Products

