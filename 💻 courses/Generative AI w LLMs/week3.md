# Reinforcement learning and power LLM-powered Applications

[TOC]



### Learning Objectives

------

- Describe how RLHF uses human feedback to improve the performance and alignment of large language models
- Explain how data gathered from human labellers is used to train a reward model for RLHF
- Define chain-of-thought prompting and describe how it can be used to improve LLMs’ reasoning and planning abilities
- Discuss the challenges that LLMs face with knowledge cut-offs, and explain how information retrieval and augmentation techniques can overcome these challenges



# Reinforcement Learning From Humam Feedback

## Align models with hum values

> Important human values, **helpfulness**, **honesty**, and **harmlessness** are sometimes collectively called ***HHH***, and are a set of principles that guide developers in the responsible use of AI

Additional fine-tuning with human feedback helps to better align models with human preferences and to increase the helpfulness, honesty, and harmlessness of the completions. This further training can also help to decrease the toxicity, often model responses and reduce the generation of incorrect information. 

## Reinforcement learning from human feedback (RLHF)

RLHF uses reinforcement learning, or RL for short, to finetune the LLM with human feedback data, resulting in a model that is better aligned with human preferences.

This can help in:

- Maximised helpfulness
- Minimize harm 
- Avoid dangerous topics

One potentially exciting application of RLHF is the personalization of LLMs, where models learn the preferences of each individual user through a continuous feedback process. This could lead to exciting new technologies like individualized learning plans or personalized AI assistants.



### Reinforcement Learning 

Reinforcement learning is a type of machine learning in which an agent learns to make decisions related to a specific goal by taking actions in an environment, to maximize some notion of a cumulative reward.
In this framework, the agent continually learns from its experiences by taking action, observing the resulting changes in the environment, and receiving rewards or penalties, based on the outcomes of its actions. By iterating through this process, the agent gradually refines its strategy or policy to make better decisions and increase its chances of success.

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240325193243433.png" alt="image-20240325193243433" style="zoom:25%;" />

#### Example of fine-tuning LLMS

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240325193713561.png" alt="image-20240325193713561" style="zoom:33%;" />

The agent's policy that guides the actions is the LLM, and its objective is to generate text that is perceived as being aligned with human preferences. This could mean that the text is, for example, helpful, accurate, and non-toxic. *The environment is the context window of the model, the space in which text can be entered via a prompt.* *The state that the model considers before taking an action is the current context.* That means any text currently contained in the context window. The action here is the act of generating text. This could be a single word, a sentence, or a longer form text, depending on the task specified by the user. 
*The action space is the token vocabulary, meaning all the possible tokens that the model can choose from to generate the completion.* How an LLM decides to generate the next token in a sequence, depends on the statistical representation of language that it learned during its training. At any given moment, the action that the model will take, meaning which token it will choose next, depends on the prompt text in the context and the probability distribution over the vocabulary space. The reward is assigned based on how closely the completions align with human preferences.
***Given the variation in human responses to language, determining the reward is more complicated than in the Tic-Tac-Toe example.*** 

One way you can do this is to have a human evaluate all of the completions of the model against some alignment metric, such as determining whether the generated text is toxic or non-toxic. This feedback can be represented as a scalar value, either a zero or a one. The LLM weights are then updated iteratively to maximize the reward obtained from the human classifier, enabling the model to generate non-toxic completions. ***However, obtaining human feedback can be time-consuming and expensive***. 

As a practical and scalable alternative, you can use an ***additional model, known as the reward model***, to classify the outputs of the LLM and evaluate the degree of alignment with human preferences. You'll start with a smaller number of human examples to train the secondary model by your traditional supervised learning methods. *Once trained, you'll use the reward model to assess the output of the LLM and assign a reward value, which in turn gets used to update the weights of the LLM and train a new human-aligned version.* 

Lastly, note that in the context of language modelling, *the sequence of actions and states is called a rollout,* instead of the term playout that's used in classic reinforcement learning. The reward model is the central component of the reinforcement learning process. It encodes all of the preferences that have been learned from human feedback, and it plays a central role in how the model updates its weights over many iterations.



### RLHF: Obtaining feedback from humans

- Define your model alignment criterion 
- For the prompt-response sets you generated obtain human feedback through the label workforce.

Before you start to train the reward model, however, you need to convert the ranking data into a pairwise comparison of completions. In other words, all possible pairs of completions from the available choices to a prompt should be classified as 0 or 1 score.

In the example shown here, there are three completions to a prompt, and the ranking assigned by the human labellers was 2, 1, 3, as shown, where 1 is the highest rank corresponding to the most preferred response. With the three different completions, there are three possible pairs purple-yellow, purple-green and yellow-green. Depending on the number N of alternative completions per prompt, you will have N choose two combinations. For each pair, you will assign a reward of 1 for the preferred response and a reward of 0 for the less preferred response. Then you'll reorder the prompts so that the preferred option comes first. This is an important step because the reward model expects the preferred completion, which is referred to as Yj first. Once you have completed this data, restructuring, the human responses will be in the correct format for training the reward model. Note that while thumbs-up and thumbs-down feedback is often easier to gather than ranking feedback, ranked feedback gives you more prom completion data to train your reward model. As you can see, here you get three prompt completion pairs from each human ranking.

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240325194519914.png" alt="image-20240325194519914" style="zoom:33%;" />

# LLM-powered applications

## Model optimizations for deployment

There are several important questions to ask at this stage. The first set is related to how your LLM will function in deployment. So how fast do you need your model to generate completions? What compute budget do you have available? And are you willing to trade off model performance for improved inference speed or lower storage? The second set of questions is tied to additional resources that your model may need. Do you intend for your model to interact with external data or other applications? And if so, how will you connect to those resources? Lastly, there's the question of how your model will be consumed. What will the intended application or API interface that your model will be consumed through look like?



### LLM optimization techniques

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327070733689.png" alt="image-20240327070733689" style="zoom:25%;" />

**Distillation** uses a larger model, the teacher model, to train a smaller model, the student model. You then use the smaller model for inference to lower your storage and compute budget. 
Similar to **quantization-**aware training, post-training quantization transforms a model's weights to a lower precision representation, such as a 16-bit floating point or 8-bit integer, this reduces the memory footprint of your model. 
The third technique, **Model Pruning**, removes redundant model parameters that contribute little to the model's performance. 



#### Distillation 

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327071122636.png" alt="image-20240327071122636" style="zoom:25%;" />

Model Distillation is a technique that focuses on having a larger teacher model train a smaller student model. The student model learns to statistically mimic the behaviour of the teacher model, either just in the final prediction layer or in the model's hidden layers as well. You'll focus on the first option here. You start with your fine-tuning LLM as your teacher model and create a smaller LLM for your student model. You freeze the teacher model's weights and use it to generate completions for your training data. At the same time, you generate completions for the training data using your student model. The knowledge distillation between the teacher and student model is achieved by minimizing a loss function called the distillation loss. To calculate this loss, distillation uses the probability distribution over tokens that is produced by the teacher model's softmax layer. Now, the teacher model is already fine-tuned on the training data. So the probability distribution likely closely matches the ground truth data and won't have much variation in tokens. That's why Distillation applies a little trick adding a temperature parameter to the softmax function. As you learned in lesson one, a higher temperature increases the creativity of the language the model generates. With a temperature parameter greater than one, the probability distribution becomes broader and less strongly peaked. This softer distribution provides you with a set of tokens that are similar to the ground truth tokens. In the context of Distillation, the teacher model's output is often referred to as soft labels and the student model's predictions as soft predictions. In parallel, you train the student model to generate the correct predictions based on your ground truth training data. Here, you don't vary the temperature setting and instead use the standard softmax function. Distillation refers to the student model outputs as the hard predictions and hard labels. The loss between these two is the student loss. The combined distillation and student losses are used to update the weights of the student model via backpropagation. The key benefit of distillation methods is that the smaller student model can be used for inference in deployment instead of the teacher model. In practice, distillation is not as effective for generative decoder models. It's typically more effective for encoder-only models, such as Burt that have a lot of representation redundancy. Note that with Distillation, you're training a second, smaller model to use during inference. You aren't reducing the model size of the initial LLM in any way.

#### Post-Training Quantization (PTQ)

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327071215928.png" alt="image-20240327071215928" style="zoom:25%;" />

Specifically Quantization Aware Training, or QAT for short. However, after a model is trained, you can perform post training quantization, or PTQ for short to optimize it for deployment. PTQ transforms a model's weights to a lower precision representation, such as 16-bit floating point or 8-bit integer. To reduce the model size and memory footprint, as well as the compute resources needed for model serving, quantization can be applied to just the model weights or to both weights and activation layers. In general, quantization approaches that include the activations can have a higher impact on model performance. Quantization also requires an extra calibration step to statistically capture the dynamic range of the original parameter values. As with other methods, there are tradeoffs because sometimes quantization results in a small percentage reduction in model evaluation metrics. However, that reduction can often be worth the cost savings and performance gains.

#### Pruning 

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327071320156.png" alt="image-20240327071320156" style="zoom:25%;" />

The last model optimization technique is pruning. At a high level, the goal is to reduce model size for inference by eliminating weights that are not contribute much to overall model performance. These are the weights with values very close to or equal to zero. Note that some pruning methods require full retraining of the model, while others fall into the category of parameter-efficient fine-tuning, such as LoRA. Some methods focus on post-training Pruning. In theory, this reduces the size of the model and improves performance. In practice, however, there may not be much impact on the size and performance if only a small percentage of the model weights are close to zero.

## Cheat Sheet

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327071544411.png" alt="image-20240327071544411" style="zoom: 50%;" />

As you saw earlier, pre-training a large language model can be a huge effort. This stage is the most complex you'll face because of the model architecture decisions, the large amount of training data required, and the expertise needed. Remember though, that in general, you will start your development work with an existing foundation model. You'll probably be able to skip this stage. If you're working with a foundation model, you'll likely start to assess the model's performance through prompt engineering, which requires less technical expertise, and no additional training of the model. If your model isn't performing as you need, you'll next think about prompt tuning and fine-tuning. Depending on your use case, performance goals, and compute budget, the methods you'll try could range from full fine-tuning to parameter-efficient fine-tuning techniques like LoRA or prompt tuning. Some level of technical expertise is required for this work. However since fine-tuning can be very successful with a relatively small training dataset, this phase could potentially be completed in a single day. Aligning your model using reinforcement learning from human feedback can be done quickly, once you have your train reward model. You'll likely see if you can use an existing reward model for this work, as you saw in this week's lab. However, if you have to train a reward model from scratch, it could take a long time because of the effort involved in gathering human feedback. Finally, the optimization techniques you learned about in the last video, typically fall in the middle in terms of complexity and effort but can proceed quite quickly assuming the changes to the model don't impact performance too much. 



## Using the LLM in applications

### Models having difficulty 

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327071759137.png" alt="image-20240327071759137" style="zoom:33%;" />

### LLM Powered applications

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327073608946.png" alt="image-20240327073608946" style="zoom:33%;" />

## RAG -Retrieval Augmented Generation

Retrieval Augmented Generation, or RAG for short, is a framework for building LLM-powered systems that make use of external data sources. And applications to overcome some of the limitations of these models. RAG is a great way to overcome the knowledge cutoff issue and help the model update its understanding of the world. While you could retrain the model on new data, this would quickly become very expensive. And require repeated retraining to regularly update the model with new knowledge. A more flexible and less expensive way to overcome knowledge cutoffs is to give your model access to additional external data at inference time. RAG is useful in any case where you want the language model to have access to data that it may not have seen. This could be new information documents not included in the original training data, or proprietary knowledge stored in your organization's private databases. Providing your model with external information can improve both the relevance and accuracy of its completions. 

Retrieval augmented generation isn't a specific set of technologies, but rather a framework for providing LLMs access to data they did not see during training. Several different implementations exist, and the one you choose will depend on the details of your task and the format of the data you have to work with. 
At the heart of this implementation is a model component called the Retriever, which consists of a query encoder and an external data source. The x takes the user's input prompt and encodes it into a form that can be used to query the data source. In the Facebook paper, the external data is a vector store, which we'll discuss in more detail shortly. But it could instead be an SQL database, CSV files, or other data storage format. These two components are trained together to find documents within the external data that are most relevant to the input query. The Retriever returns the best single or group of documents from the data source and combines the new information with the original user query. The newly expanded prompt is then passed to the language model, which generates a completion that makes use of the data.

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327073913838.png" alt="image-20240327073913838" style="zoom:33%;" />

### External Information sources

- DOcument
- wikis
- Expert Systems
- Web pagaes 
- Databases
  - contains vector representations of text. This is a particularly useful data format for language models, since internally they work with vector representations of language to generate text. Vector stores enable a fast and efficient kind of relevant search based on similarity.

### Data Preparation for vector source RAG

Starting with the ***size of the context window.*** Most text sources are too long to fit into the limited context window of the model, which is still at most just a few thousand tokens. Instead, the external data sources are chopped up into many chunks, each of which will fit in the context window. Packages like *Langchain* can handle this work for you. 

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327074215987.png" alt="image-20240327074215987" style="zoom:33%;" />

***The data must be available in a format that allows for easy retrieval of the most relevant text.*** Recall that large language models don't work directly with text, but instead create vector representations of each token in an embedding space. These embedding vectors allow the LLM to identify semantically related words through measures such as cosine similarity, which you learned about earlier. Rag methods take the small chunks of external data and process them through the large language model, to create embedding vectors for each. These new representations of the data can be stored in structures called vector stores, which allow for fast searching of datasets and efficient identification of semantically related text. Vector databases are a particular implementation of a vector store where each vector is also identified by a key. This can allow, for instance, the text generated by RAG to also include a citation for the document from which it was received.

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327074452348.png" alt="image-20240327074452348" style="zoom:25%;" />

## Interacting with external applications

- Trigger API call
- Perform calculations

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327074937898.png" alt="image-20240327074937898" style="zoom:25%;" />

### Helping LLMs reason and plan with chain-of-thought

It is important that LLMs can reason through the steps that an application must take, to satisfy a user request. Unfortunately, complex reasoning can be challenging for LLMs, especially for problems that involve multiple steps or mathematics. These problems exist even in large models that show good performance at many other tasks.



One strategy that has demonstrated some success is prompting the model to think more like a human, by breaking the problem down into steps.
These intermediate calculations form the reasoning steps that a human might take, and the full sequence of steps illustrates the chain of thought that went into solving the problem. Asking the model to mimic this behavior is known as ***chain of thought prompting.*** It works by including a series of intermediate reasoning steps into any examples that you use for one or few-shot inference. By structuring the examples in this way, you're essentially teaching the model how to reason through the task to reach a solution. 

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327075251741.png" alt="image-20240327075251741" style="zoom:25%;" />

## Program-aided languange models (PAL)

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327075646186.png" alt="image-20240327075646186" style="zoom:33%;" />

The strategy behind PAL is to have the LLM generate completions where reasoning steps are accompanied by computer code. This code is then passed to an interpreter to carry out the calculations necessary to solve the problem. You specify the output format for the model by including examples for one or few short inference in the prompt.

> Chain of though prompting but with python code to be executed in the pyhton interperter

Steps:

- To prepare for inference with PAL, you'll format your prompt to contain one or more examples. Each example should contain a question followed by reasoning steps in lines of Python code that solve the problem.
- Next, you will append the new question that you'd like to answer to the prompt template. Your resulting PAL formatted prompt now contains both the example and the problem to solve.
- Next, you'll pass this combined prompt to your LLM, which then generates a completion that is in the form of a Python script having learned how to format the output based on the example in the prompt.
- hand off the script to a Python interpreter, which you'll use to run the code and generate an answer.

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327080010101.png" alt="image-20240327080010101" style="zoom:33%;" />

- Now append the text containing the answer, which you know is accurate because the calculation was carried out in Python to the PAL formatted prompt you started with.
- Now when you pass the updated prompt to the LLM, it generates a completion that contains the correct answer.

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327080132259.png" alt="image-20240327080132259" style="zoom:33%;" />

How to automate this process so that you don't have to pass information back and forth between the LLM, and the interpreter by hand. This is where the orchestrator that you saw earlier comes in. The orchestrator shown here as the yellow box is a technical component that can manage the flow of information and the initiation of calls to external data sources or applications. It can also decide what actions to take based on the information contained in the output of the LLM. 

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327080247355.png" alt="image-20240327080247355" style="zoom:33%;" />

Remember, the LLM is your application's reasoning engine. Ultimately, it creates the plan that the orchestrator will interpret and execute. In PAL there's only one action to be carried out, the execution of Python code. The LLM doesn't really have to decide to run the code, it just has to write the script which the orchestrator then passes to the external interpreter to run. However, most real-world applications are likely to be more complicated than the simple PAL architecture. Your use case may require interactions with several external data sources.You may need to manage multiple decision points, validation actions, and calls to external applications.

## ReAct: Combining reasiong and action 

ReAct that can help LLMs plan out and execute these workflows. ReAct is a prompting strategy that combines chain of thought reasoning with action planning

ReAct uses structured examples to show a large language model how to reason through a problem and decide on actions to take that move it closer to a solution. The example prompts start with a question that will require multiple steps to answer.

The ability of the model to reason well and plan actions depends on its scale. Larger models are generally your best choice for techniques that use advanced prompting, like PAL or ReAct.

## Building up a ReAct prompt

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327085538559.png" alt="image-20240327085538559" style="zoom: 33%;" />



## LangChain

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327085735466.png" alt="image-20240327085735466" style="zoom:25%;" />

LangChain framework provides you with modular pieces that contain the components necessary to work with LLMs. These components include prompt templates for many different use cases that you can use to format both input examples and model completions. And memory that you can use to store interactions with an LLM. The framework also includes pre-built tools that enable you to carry out a wide variety of tasks, including calls to external datasets and various APIs. Connecting a selection of these individual components together results in a chain. The creators of LangChain have developed a set of predefined chains that have been optimized for different use cases, and you can use these off the shelf to quickly get your app up and running. Sometimes your application workflow could take multiple paths depending on the information the user provides. In this case, you can't use a pre-determined chain, but instead we'll need the flexibility to decide which actions to take as the user moves through the workflow. LangChain defines another construct, known as an agent, that you can use to interpret the input from the user and determine which tool or tools to use to complete the task. LangChain currently includes agents for both PAL and ReAct, among others. Agents can be incorporated into chains to take an action or plan and execute a series of actions. LangChain is in active development, and new features are being added all the time, like the ability to examine and evaluate the LLM's completions throughout the workflow. 

---

introduces ReAct, a novel approach that integrates verbal reasoning and interactive decision making in large language models (LLMs). While LLMs have excelled in language understanding and decision making, the combination of reasoning and acting has been neglected. ReAct enables LLMs to generate reasoning traces and task-specific actions, leveraging the synergy between them. The approach demonstrates superior performance over baselines in various tasks, overcoming issues like hallucination and error propagation. ReAct outperforms imitation and reinforcement learning methods in interactive decision making, even with minimal context examples. It not only enhances performance but also improves interpretability, trustworthiness, and diagnosability by allowing humans to distinguish between internal knowledge and external information.

In summary, ReAct bridges the gap between reasoning and acting in LLMs, yielding remarkable results across language reasoning and decision making tasks. By interleaving reasoning traces and actions, ReAct overcomes limitations and outperforms baselines, not only enhancing model performance but also providing interpretability and trustworthiness, empowering users to understand the model's decision-making process.

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/1sTOKhNdQZ6PrrOKANlbRQ_f0c0bdfe18414de681e207ad6b23cef1_image.png" alt="img" style="zoom: 50%;" />

***Image***: The figure provides a comprehensive visual comparison of different prompting methods in two distinct domains. The first part of the figure (1a) presents a comparison of four prompting methods: Standard, Chain-of-thought (CoT, Reason Only), Act-only, and ReAct (Reason+Act) for solving a HotpotQA question. Each method's approach is demonstrated through task-solving trajectories generated by the model (Act, Thought) and the environment (Obs). The second part of the figure (1b) focuses on a comparison between Act-only and ReAct prompting methods to solve an AlfWorld game. In both domains, in-context examples are omitted from the prompt, highlighting the generated trajectories as a result of the model's actions and thoughts and the observations made in the environment. This visual representation enables a clear understanding of the differences and advantages offered by the ReAct paradigm compared to other prompting methods in diverse task-solving scenarios.



## LLM application architectures

***Infrastructure***:  This layer provides the compute, storage, and network to serve up your LLMs, as well as to host your application components. You can make use of your on-premises infrastructure for this or have it provided for you via on-demand and pay-as-you-go Cloud services.

**Models**: you'll include the large language models you want to use in your application.These could include foundation models, as well as the models you have adapted to your specific task. The models are deployed on the appropriate infrastructure for your inference needs.

You may also have the need to retrieve information from external sources, such as those discussed in the retrieval augmented generation section.

Your application will return the completions from your large language model to the user or consuming application. Depending on your use case, you may need to implement a mechanism to capture and store the outputs.

-  For example, you could build the capacity to store user completions during a session to augment the fixed contexts window size of your LLM. You can also gather feedback from users that may be useful for additional fine-tuning, alignment, or evaluation as your application matures.

Next, you may need to use additional tools and frameworks for large language models that help you easily implement some of the techniques discussed in this course.

- As an example, you can use LangChain built-in libraries to implement techniques like pow react or chain of thought prompting.
- You may also utilize model hubs which allow you to centrally manage and share models for use in applications.

In the final layer, you typically have some type of user interface that the application will be consumed through, such as a website or a rest API. This layer is where you'll also include the security components required for interacting with your application. 

<img src="../../%F0%9F%96%BC%EF%B8%8F%20images/image-20240327081133259.png" alt="image-20240327081133259" style="zoom:33%;" />



# Summary 

How to align your models with human preferences, such as helpfulness, harmlessness, and honesty by fine-tuning using a technique called reinforcement learning with human feedback, or RLHF for short. Given the popularity of RLHF, there are many existing RL reward models and human alignment datasets available, enabling you to quickly start aligning your models. In practice, RLHF is a very effective mechanism that you can use to improve the alignment of your models, reduce the toxicity of their responses, and let you use your models more safely in production. You also saw important techniques to optimize your model for inference by reducing the size of the model through distillation, quantization, or pruning. This minimizes the amount of hardware resources needed to serve your LLMs in production. Lastly, you explored ways that you can help your model perform better in deployment through structured prompts and connections to external data sources and applications. LLMs can play an amazing role as the reasoning engine in an application, exploiting their intelligence to power exciting, useful applications. Frameworks like LangChain are making it possible to quickly build, deploy, and test LLM powered applications, and it's a very exciting time for developers. 

# Resources

## **Generative AI Lifecycle**

- [**Generative AI on AWS: Building Context-Aware, Multimodal Reasoning Applications**](https://www.amazon.com/Generative-AI-AWS-Multimodal-Applications/dp/1098159225/) - This O'Reilly book dives deep into all phases of the generative AI lifecycle including model selection, fine-tuning, adapting, evaluation, deployment, and runtime optimizations.

## **Reinforcement Learning from Human-Feedback (RLHF)**

- [**Training language models to follow instructions with human feedback**](https://arxiv.org/pdf/2203.02155.pdf) **-** Paper by OpenAI introducing a human-in-the-loop process to create a model that is better at following instructions (InstructGPT).
- [**Learning to summarize from human feedback**](https://arxiv.org/pdf/2009.01325.pdf) - This paper presents a method for improving language model-generated summaries using a reward-based approach, surpassing human reference summaries.

## **Proximal Policy Optimization (PPO)**

- [**Proximal Policy Optimization Algorithms**](https://arxiv.org/pdf/1707.06347.pdf) - The paper from researchers at OpenAI that first proposed the PPO algorithm. The paper discusses the performance of the algorithm on a number of benchmark tasks including robotic locomotion and game play.
- [**Direct Preference Optimization: Your Language Model is Secretly a Reward Model**](https://arxiv.org/pdf/2305.18290.pdf) - This paper presents a simpler and effective method for precise control of large-scale unsupervised language models by aligning them with human preferences.

## **Scaling human feedback**

- [**Constitutional AI: Harmlessness from AI Feedback**](https://arxiv.org/pdf/2212.08073.pdf) - This paper introduces a method for training a harmless AI assistant without human labels, allowing better control of AI behavior with minimal human input.

## **Advanced Prompting Techniques**

- [**Chain-of-thought Prompting Elicits Reasoning in Large Language Models**](https://arxiv.org/pdf/2201.11903.pdf) -  Paper by researchers at Google exploring how chain-of-thought prompting improves the ability of LLMs to perform complex reasoning.
- [**PAL: Program-aided Language Models**](https://arxiv.org/abs/2211.10435) - This paper proposes an approach that uses the LLM to read natural language problems and generate programs as the intermediate reasoning steps.
- [**ReAct: Synergizing Reasoning and Acting in Language Models**](https://arxiv.org/abs/2210.03629) This paper presents an advanced prompting technique that allows an LLM to make decisions about how to interact with external applications.

## **LLM powered application architectures**

- [**LangChain Library (GitHub)**](https://github.com/hwchase17/langchain) - This library is aimed at assisting in the development of those types of applications, such as Question Answering, Chatbots and other Agents. You can read the documentation [here](https://docs.langchain.com/docs/).
- [**Who Owns the Generative AI Platform?**](https://a16z.com/2023/01/19/who-owns-the-generative-ai-platform/) - The article examines the market dynamics and business models of generative AI.