# Basics

The **Transformer** architecture outperformed the recurrent neural networks(RNNS) on machine translations tasks both in quality and costs.

***ULMFiT*** , an effective transfer learning method, showed that training long short-term memory(LSTM) networks on very large and diverse corpus could produce sota classifiers with little labeled data. 



These two advances were the major catalysts for today well-known transformers:

- Generative Pretrained Transformer (**GPT**) 
- Bidirectional Encoder Representation from Transformers (**BERT**)

> By combining the Transformer Architecture with unsupervised these models remove the need to train task-specific architecture from scratch

To understand the Transformers we first need to understand:

- The encoder-decoder framework
- Attention Mechanisms
- Transfer Learning

## The Encoder-Decoder Framework

Before the Transformers, recurrent architectures (LSTM, GRU) were the sota. These architectures contain a feedback loop in the network connections that allows information to propagate from one step to another, making ideal form modeling sequential data, like text.  As

```mermaid
flowchart TD;

    input --> rnn_cell;
    rnn_cell --> rnn_cell;
	rnn_cell --> state_t

    classDef red fill:red,stroke:#cc0000,stroke-width:2px;
    classDef blue fill:#9999ff,stroke:#0000cc,stroke-width:2px;
    classDef green fill:#ccffcc,stroke:#006600,stroke-width:2px;
    classDef orange fill:#ffcc99,stroke:#cc6600,stroke-width:2px;
    
    linkStyle 0 stroke-width:4px,stroke:green;
    linkStyle 1 stroke-width:4px,stroke:purple;
    linkStyle 2 stroke-width:4px,stroke:blue;
    
    class input green;
	class rnn_cell purple; 
	class state_t blue; 
```

***Unrolled*** 



```mermaid
graph TD;

    input_1--> a[rnn_cell];
    a --> a;
	a --> state_1
	
	input_2 --> b[rnn_cell];
	a --> b
    b --> b
	b --> state_2
	
	input_3 --> c[rnn_cell];
	b --> c
    c --> c
	c --> state_3
	
	input_t --> t[rnn_cell];
	c --> t
    t --> t
	t --> state_t
	linkStyle default stroke:green;
	
	    %% Add IDs for reference
    classDef red fill:#ffcccc,stroke:#cc0000,stroke-width:2px;
    classDef blue fill:#9999ff,stroke:#0000cc,stroke-width:2px;
    classDef green fill:#ccffcc,stroke:#006600,stroke-width:2px;
    classDef orange fill:#ffcc99,stroke:#cc6600,stroke-width:2px;
    classDef purple fill:#4444ff,stroke:#660099,stroke-width:2px;
    classDef red fill:red,stroke:#cc0000,stroke-width:2px;
    classDef blue fill:#9999ff,stroke:#0000cc,stroke-width:2px;
    classDef green fill:#ccffcc,stroke:#006600,stroke-width:2px;
    classDef orange fill:#ffcc99,stroke:#cc6600,stroke-width:2px;
    
    linkStyle 0 stroke-width:4px,stroke:green;
    linkStyle 1 stroke-width:4px,stroke:purple;
    linkStyle 2 stroke-width:4px,stroke:blue;
	linkStyle 3 stroke-width:4px,stroke:green;
	linkStyle 4 stroke-width:4px,stroke:yellow;
    linkStyle 5 stroke-width:4px,stroke:purple;
    linkStyle 6 stroke-width:4px,stroke:blue;
    linkStyle 7 stroke-width:4px,stroke:green;
    linkStyle 9 stroke-width:4px,stroke:blue;
    
    class input_1,input_2,input_3,input_t green;
	class a,b,c,t yellow; 
	class state_1,state_2,state_3,state_t blue; 

```



