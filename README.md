# Skip-gram Word Embedding Model

## Overview
This repository contains an implementation of a **Skip-gram Word Embedding Model** designed to analyze the impact of **window size** and **embedding dimensions** on the quality of word embeddings. The model is trained using **news headlines** and evaluated based on **cosine similarity** between word vectors.

## Features
- Implements **Skip-gram with Negative Sampling**.
- Evaluates different configurations of **window sizes (1, 2, 3)** and **embedding dimensions (50, 100, 200)**.
- Uses **cosine similarity** to analyze word relationships.
- Identifies the best configuration based on **average similarity scores**.
- Provides **word similarity rankings** for interpretability.

## Best Configuration
Based on experiments, the **best configuration** found was:
- **Window Size = 3**
- **Embedding Dimension = 50**
- **Average Similarity Score = 0.2964**

This configuration achieves a balance between **semantic coherence, generalization, and computational efficiency**.

## Installation
```bash
# Clone the repository
git clone https://github.com/fliw/Skipgram-Natural-Language-Processing.git
cd Skipgram-Natural-Language-Processing

# Install dependencies
pip install -r requirements.txt
```

## Usage in Jupyter Notebook
To use the model in Jupyter Notebook, follow these steps:

1. **Open Jupyter Notebook** and navigate to the repository folder.
2. **Run the provided notebook**: `Skipgram_Model.ipynb`
3. If needed, manually execute the following steps in a new notebook:
    
    - **Import necessary libraries**:
      ```python
      import numpy as np
      import pandas as pd
      ```
    
    - **Load the notebook functions** (since everything is in a single notebook, define the required functions inside a cell before using them).
    
    - **Train the model**:
      ```python
      API_KEY = "your_newsapi_key"  # Replace with your actual NewsAPI key
      model, word2idx = train_model(API_KEY, window_size=3, embedding_dim=50)
      ```
    
    - **Compute word similarity**:
      ```python
      sample_word = "economy"
      similar_words = compute_similarity(sample_word, model, word2idx)
      print(f"Top similar words to '{sample_word}':")
      for word, similarity in similar_words:
          print(f"  {word}: {similarity:.4f}")
      ```

## Included Files
- **`Skipgram_Model.ipynb`** - A Jupyter Notebook demonstrating training and evaluation of the model.
- **`requirements.txt`** - List of dependencies.

## Future Enhancements
- Experimenting with **larger datasets**.
- Integrating **pre-trained embeddings** (e.g., Word2Vec, FastText).
- Implementing **subword information** to handle out-of-vocabulary (OOV) words.
- Applying **regularization techniques** to improve embedding stability.
