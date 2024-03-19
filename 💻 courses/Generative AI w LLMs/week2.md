# Fine-tuning and evaluating large language models

### Learning Objectives

------

- Describe how fine-tuning with instructions using prompt datasets can improve performance on one or more tasks
- Define catastrophic forgetting and explain techniques that can be used to overcome it
- Define the term Parameter-efficient Fine Tuning (PEFT)
- Explain how PEFT decreases computational cost and overcomes catastrophic forgetting
- Explain how fine-tuning with instructions using prompt datasets can increase LLM performance on one or more tasks

## Instruction fine-tuning

ICL have some limitations:

- In smaller models, even with one/few shots inference, the model isn’t capable of performing
- Examples take up space in the context window

To overcome these limitations we can **fine-tune** the model. 

### LLM fine-tuning at a high level

The process is known as fine-tuning to further train a base model. In contrast to pre-training, where you train the LLM using vast amounts of unstructured textual data via self-supervised learning, fine-tuning is a supervised learning process where you use a data set of labelled examples to update the weights of the LLM. The labelled examples are prompt completion pairs, the fine-tuning process extends the training of the model to improve its ability to generate good completions for a specific task. One strategy, instruction fine-tuning, is particularly good at improving a model's performance on various tasks. 

> **Instruction fine-tuning trains the model using examples demonstrating how it should respond to a specific instruction.** 

 Instruction fine-tuning, where all of the model's weights are updated is known as full fine-tuning. The process results in a new version of the model with updated weights. It is important to note that just like pre-training, full fine tuning requires enough memory and compute budget to store and process all the gradients, optimizers and other components that are being updated during training. So you can benefit from the memory optimization and parallel computing strategies

The 1st step is to prepare the dataset ( there are a lot of templates of prompt instruction to fine-tune models). After that the process is the same, divide the dataset in train, validation and test and use that to train/evaluate the model.

>  The fine-tuning process results in a new version of the base model, often called an instruct model that is better at the tasks you are interested in. Fine-tuning with instruction prompts is the most common way to fine-tune LLMs these days.

## Fine-tuning on a single task

LLMs have become famous for their ability to perform many different language tasks within a single model, your application may only need to perform a single task. In this case, you can fine-tune a pre-trained model to improve performance on only the task that is of interest to you.  

### Catastrophic forgetting

However, there is a potential downside to fine-tuning on a single task. The process may lead to a phenomenon called ***catastrophic forgetting.*** Catastrophic forgetting happens because the full fine-tuning process modifies the weights of the original LLM. While this leads to great performance on a single fine-tuning task, it can degrade performance on other tasks. For example, while fine-tuning can improve the ability of a model to perform sentiment analysis on a review and result in a quality completion, the model may forget how to do other tasks.



#### How to avoid catastrophic forgetting?

- First note that you might not have to!
- Fine-tune on  **multiple tasks** at the same time
- Consider ***Parameter Efficient Fine-tuning(PETF)***

First of all, it's important to decide whether catastrophic forgetting impacts your use case. If all you need is reliable performance on the single task you fine-tuned on, it may not be an issue that the model can't generalize to other tasks. If you do want or need the model to maintain its multitask generalized capabilities, you can perform fine-tuning on multiple tasks at one time. **Good multitask fine-tuning may require 50-100,000 examples across many tasks,** and so will require more data and compute to train. Will discuss this option in more detail shortly. Our second option is to perform parameter efficient fine-tuning, or PEFT for short instead of full fine-tuning. **PEFT is a set of techniques that preserves the weights of the original LLM and trains only a small number of task-specific adapter layers and parameters**. **PEFT shows greater robustness to catastrophic forgetting since most of the pre-trained weights are left unchanged. PEFT is an exciting and active area of research that we will cover later this week. In the meantime, let's move on to the next video and take a closer look at multitask fine-tuning.**

## Multi-task instruction fine-tuning

> Multitask fine-tuning is an extension of single-task fine-tuning, where the training dataset is comprised of example inputs and outputs for multiple tasks.

You train the model on this mixed dataset so that it can improve the performance of the model on all the tasks simultaneously, thus avoiding the issue of catastrophic forgetting. Over many epochs of training, the calculated losses across examples are used to update the weights of the model, resulting in an *instruction tuned model that is learned how to be good at many different tasks simultaneously.* One drawback to multitask fine-tuning is that it requires a lot of data. You may need as many as 50-100,000 examples in your training set.

### Instruction fine-tuning FLAN

***FLAN***, which stands for *fine-tuned language net,* is a specific set of instructions used to fine-tune different models. Because their FLAN fine-tuning is the last step of the training process the authors of the original paper called it the metaphorical dessert to the main course of pre-training quite a fitting name. FLAN-T5, is the FLAN instruct version of the T5 foundation model while FLAN-PALM is the flattening struct version of the palm foundation model. 

- FLAN models refer to a specific set of instructions used to perform instruction fine-tuning

FLAN-T5 is a great general-purpose instruction model. In total, it's been fine-tuned on 473 datasets across 146 task categories.

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/Screenshot%202024-03-19%20at%2007.29.26.png" alt="Screenshot 2024-03-19 at 07.29.26" style="zoom:33%;" />

### Sample FLAN-T5 prompt templates

The template is comprised of several different instructions that all ask the model to do the same thing. Summarize a dialogue. For example, briefly summarize that dialogue. What is a summary of this dialogue? What was going on in that conversation? Including different ways of saying the same instruction helps the model generalize and perform better. The summary is used as the label. After applying this template to each row in the SAMSum dataset, you can use it to fine-tune a dialogue summarization task.

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240319074658803.png" alt="image-20240319074658803" style="zoom: 50%;" />

### Improve FLAN-T5’s summarisation capabilities

You can perform additional fine-tuning of the FLAN-T5 model using a dialogue dataset that is much closer to the conversations that happened with your bot. This is the exact scenario that you'll explore in the lab this week. You'll make use of an additional domain-specific summarization dataset called dialogsum to improve FLAN-T5's is ability to summarize support chat conversations. 

In practice, you'll get the most out of fine-tuning by using your company's own internal data. For example, the support chat conversations from your customer support application. This will help the model learn the specifics of how your company likes to summarize conversations and what is most useful to your customer service colleagues. 

---

[This paper](https://arxiv.org/abs/2210.11416) introduces FLAN (Fine-tuned LAnguage Net), an instruction finetuning method, and presents the results of its application. The study demonstrates that by fine-tuning the 540B PaLM model on 1836 tasks while incorporating Chain-of-Thought Reasoning data, FLAN achieves improvements in generalization, human usability, and zero-shot reasoning over the base model. The paper also provides detailed information on how each these aspects was evaluated.



<img src="https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/mu9BmR3dSRuEBq2t64mL0A_f9f43fc146bb42779a56c467bd929df1_image.png?expiry=1710979200000&hmac=n0CwPE2_8orl2Dfrb0iT7up-bJ9CzWmOG-G5LGwSkoQ" alt="img" style="zoom:25%;" />

Here is the image from the lecture slides that illustrates the fine-tuning tasks and datasets employed in training FLAN. The task selection expands on previous works by incorporating dialogue and program synthesis tasks from Muffin and integrating them with new Chain of Thought Reasoning tasks. It also includes subsets of other task collections, such as T0 and Natural Instructions v2. Some tasks were held-out during training, and they were later used to evaluate the model's performance on unseen tasks.