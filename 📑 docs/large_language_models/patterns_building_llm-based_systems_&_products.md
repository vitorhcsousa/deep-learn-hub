# Intro LLM 

https://www.youtube.com/watch?v=zjkBMFhNj_g

https://drive.google.com/file/d/1pxx_ZI7O-Nwl7ZLNk5hI3WzAsTLwvNU7/view?pli=1

***LLM open source -> basically is just two files parameters file/file to run the neural network. No need for internet*** 

## Train 

- How do we get the parameters? The magic of LLMs
  - Training  is more involved, it’s like compressing the internet (ex 10TB of data); we need GPUs (a lot)  and let training for some days (eg Llama 2 70B took 6000GPUs and 12 days, costing ~$2M generating a parameters file with 140GB)
- Neural Network Training (To predict the next work in the sequence)
  - In the task of predicting the next word during the  train the model is *forced* to learn about the world (personalities, events, shows…) and that knowledge is compressed in the weights 



## How do they work?

Using Transformer NN architecture- we know the math computations, and we can measure that performance but we don´t know how they interact. 

Litter is known in full detail…

- Billion of parameters are dispersed through the network
- We know how to iteratively adjust them to make them better at prediction
- We can measure that this works, but we don´t really known how the billions of parameters collaborate to do it 
- e.g. the reversal curse with the Tom Cruise mother question

>  **LLMs are mostly inscrutable artefacts, develop correspondingly sophisticated evaluations** 

The field of interpretability is trying to explore this on a deeper level 



## Fine Tuning into an Assistant 

we don´t want a document generator. We, generally, want to give equations and we want answers to that. 

- We train a model with labelled documents like question&answers  + quality -quantity
- Pre-training stage - Train on a lot of data: add the knowledge to the model; This step of ***Fine Tuning*** is like alignment to the goal that we want, in this case, an assistant. 

##### How to train our ChatGPT

###### Stage 1: Pretraining  (every month)

1. Download tons of data from the internet (text)
2. Get a cluster of XXXX GPUs
3. Compress the text into a neural network, pay $MM and wait Xx days/weeks
4. with this, we get the **base model**

##### Stage 2: Finetuning (every week)

1. Write labelling instructions 
2. Collect 100k(per example) high quality Q&A responses, and/or comparisons
   1. A second kind of label: comparisons - it is often much easier to compare Answers instead of writing Answers (reinforcement learning from human feedback)
3. Fine-tune base model based on this data (~1/day)
4. With that, we obtain the assistant model 
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

And trends do no show signs of topping out

> We can expect mode intelligence for free by scalling 

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/Screenshot%202024-03-10%20at%2010.50.07.png" alt="Screenshot 2024-03-10 at 10.50.07" style="zoom:33%;" />

with more data, we can see that the performance improves

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/Screenshot%202024-03-10%20at%2010.51.12.png" alt="Screenshot 2024-03-10 at 10.51.12" style="zoom:25%;" />



---

## Thinking, System 1/2

Currently, LLMs only work on system 1 (of book think fast & slow), it’s automatic almost instinctive. A lot of people are wondering if these kinds of systems can think about the problem/question that we present to them (like system 2). For that we probably need to convert time to accuracy, *here my question takes 10 minutes to think and answer*. 

### Self Improvement

#alphaGo

AlphaGo had two major stages:

- Learn by imitating expert human players
- Learn by self-improvement(reward = win the game) 
  - Says if what the model ended up doing was a good or a bad move


In LLMs, we just have the imitating part. The big question is what Step 2 looks like in the open domain of language. The main challenge is the lack of reward criteria. 
In narrow domains, this could be possible, but in a general approach is hard to find reward criteria. 



### LLMs OS

![image-20240310105742873](../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240310105742873.png)

### LLM Security 



# Patterns for Building LLM-based Systems & Products

