# Sentiment Analysis Predictor

This project is a personal endeavor aimed at exploring the intricacies of Large Language Models (LLMs) and their underlying mechanisms. Through this initiative, I delve into how LLMs generate meaningful and accurate predictions by analyzing text data and uncovering the layers that contribute to their training process.

## Model's Capability
The model is designed to predict the accuracy of sentiment classifications with a current confidence level of **XXXX**. It achieves this by leveraging a robust training dataset and advanced machine learning architecture. With the current configuration, the model is optimized for high-accuracy predictions in diverse sentiment analysis tasks, including:  
- Customer feedback analysis  
- Social media sentiment monitoring  
- Product reviews classification    


## Dataset

The dataset used for this project is the [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), a popular resource for sentiment analysis tasks.
### Dataset Distribution & Analysis

| Sentiment | Number of Reviews | Average Length of Text | 
|-----------|-------------------|------------------------|
| Positive  | 12,500            | 236.7065               |
| Negative  | 12,500            | 230.86784              |
![image](https://github.com/user-attachments/assets/a466fe23-2e42-4009-9d70-1c0018fc2d50)

## Model Architecture  
The core of the model is built on a **Transformer architecture** with an integrated **attention mechanism**, which offers the following advantages:  
- **Focus on Key Features:** The attention mechanism identifies and emphasizes the most relevant parts of the input data, ensuring critical information is prioritized during sentiment analysis.  
- **Handle Sequential Data Effectively:** Transformers process sequences in parallel, significantly improving efficiency compared to traditional models like RNNs or LSTMs.  
- **Enhance Context Understanding:** By analyzing relationships between words in a sentence, the model captures nuanced sentiment, even in complex phrasing or ambiguous language.  
<img src="https://github.com/user-attachments/assets/9cd96477-14a2-4cc3-a57c-1d31e2fb0cfc" alt="image" width="300"/>

## Hypertuning of Parameters


## Future Improvements  
To further improve the model's capabilities, we plan to:  
1. Expand the training dataset to include diverse linguistic and cultural contexts.  
2. Fine-tune the model on domain-specific sentiment data for specialized applications.  
3. Enhance interpretability by visualizing attention weights, enabling users to understand how the model makes predictions.  
