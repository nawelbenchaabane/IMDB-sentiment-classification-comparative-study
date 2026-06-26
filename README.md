# IMDB Sentiment Classification — Comparative Study

This project presents a comparative study of several approaches for binary sentiment classification on the IMDB movie reviews dataset.

The goal is to predict whether a movie review is **negative** or **positive** using different NLP and deep learning techniques, ranging from classical machine learning models to Transformer-based architectures.

The project compares the following approaches:

- TF-IDF + Logistic Regression
- TensorFlow Hub Swivel Embedding + Dense Classifier
- TensorFlow Hub Universal Sentence Encoder + Dense Classifier
- TensorFlow Hub NNLM 128D + Dense Classifier
- BiLSTM
- Improved BiLSTM
- Fine-tuned DistilBERT

The best-performing model was **DistilBERT**, achieving a test accuracy of **90.76%**.

---

## Project Overview

Sentiment analysis is a common Natural Language Processing task that aims to identify the emotional polarity of a text. In this project, the task is formulated as a binary classification problem:

- `0` → Negative review
- `1` → Positive review

The main objective was not only to train a single model, but to compare several modeling strategies and analyze their strengths and limitations.

This project shows that classical NLP methods such as TF-IDF can be highly competitive, while fine-tuning a Transformer-based model such as DistilBERT can provide the best final performance.

---

## Dataset

The project uses the IMDB Reviews dataset, which contains movie reviews labeled as positive or negative.

The dataset was loaded using `tensorflow_datasets`.

The data was split as follows:

| Split | Number of samples |
|---|---:|
| Training set | 15,000 |
| Validation set | 10,000 |
| Test set | 25,000 |

The classes are balanced between positive and negative reviews, which makes accuracy a meaningful first metric. However, additional metrics such as precision, recall, F1-score, and confusion matrix were also used for evaluation.

---

## Exploratory Data Analysis

Before training the models, the dataset was explored to better understand the structure of the reviews.

The analysis included:

- Checking the number of samples in each split
- Inspecting positive and negative review examples
- Analyzing the class distribution
- Measuring review lengths in terms of characters and words
- Visualizing the distribution of review lengths
- Checking for missing or empty reviews

This step confirmed that the dataset was clean, balanced, and suitable for binary sentiment classification.

---

## Models Implemented

### 1. TF-IDF + Logistic Regression

The first model is a classical machine learning baseline.

The raw text reviews were transformed into numerical vectors using TF-IDF. The vectorizer used unigrams and bigrams in order to capture both individual words and short expressions such as:

- `not good`
- `very bad`
- `waste time`
- `really enjoyed`

The resulting TF-IDF matrix was used to train a Logistic Regression classifier.

This model provided a strong baseline and outperformed several deep learning architectures.

---

### 2. TensorFlow Hub Swivel Embedding + Dense Classifier

The first deep learning model used a pre-trained TensorFlow Hub embedding layer:

```text
https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1
```

This model transforms each review into a dense vector of 20 dimensions. The embedding is then passed through a small dense neural network for binary classification.

The architecture can be summarized as:

```text
Raw review → Swivel embedding → Dense layer → Output layer
```

Although simple and efficient, this model did not outperform the TF-IDF baseline. One possible reason is that the embedding dimension is relatively small and may not capture complex sentiment patterns.

---

### 3. TensorFlow Hub Universal Sentence Encoder + Dense Classifier

The second TensorFlow Hub model used the Universal Sentence Encoder:

```text
https://tfhub.dev/google/universal-sentence-encoder/4
```

Unlike the Swivel model, the Universal Sentence Encoder produces a richer sentence-level embedding of 512 dimensions.

The architecture can be summarized as:

```text
Raw review → Universal Sentence Encoder → Dense layers with dropout → Sentiment probability
```

The Universal Sentence Encoder provides semantic sentence representations, but in this experiment it did not outperform the TF-IDF baseline. This suggests that a frozen general-purpose embedding may not always be optimal for a specific sentiment classification task.

---

### 4. TensorFlow Hub NNLM 128D + Dense Classifier

The third TensorFlow Hub model used the NNLM 128D embedding model:

```text
https://tfhub.dev/google/nnlm-en-dim128/2
```

This model transforms each review into a dense vector of 128 dimensions. Unlike the Universal Sentence Encoder, this embedding layer was set as trainable, allowing its weights to be adapted to the IMDB sentiment classification task.

The architecture can be summarized as:

```text
Raw review → NNLM 128D embedding → Dense layers with dropout → Sentiment probability
```

This model performed better than the Universal Sentence Encoder model, but still remained below the TF-IDF + Logistic Regression baseline.

---

### 5. BiLSTM

A Bidirectional LSTM model was implemented to process the reviews as sequences of tokens.

The preprocessing pipeline was:

```text
Raw review → TextVectorization → Integer token sequence → Embedding layer → BiLSTM
```

The `TextVectorization` layer was used to build a vocabulary from the training data and convert each review into a sequence of integer token IDs. These IDs were then passed to an Embedding layer, which learned dense word representations during training.

The BiLSTM layer reads the sequence in both directions, allowing the model to capture contextual information from both the beginning and the end of the review.

The initial BiLSTM model underperformed compared to the TF-IDF baseline.

---

### 6. Improved BiLSTM

An improved BiLSTM architecture was also tested.

The improvements included:

- Increasing the maximum vocabulary size
- Increasing the sequence length
- Reducing dropout
- Using a lower learning rate
- Improving the input shape definition

This improved version performed better than the first BiLSTM, but it still did not outperform the classical TF-IDF baseline.

---

### 7. Fine-tuned DistilBERT

The final model used in this project was:

```text
distilbert-base-uncased
```

DistilBERT is a smaller and faster version of BERT obtained through knowledge distillation. It keeps the main Transformer-based architecture while reducing the number of layers.

The pipeline was:

```text
Raw review
→ DistilBERT tokenizer
→ input_ids + attention_mask + token_type_ids
→ Dynamic padding
→ DistilBERT embeddings
→ Transformer encoder layers
→ Classification head
→ Sentiment prediction
```

Each review was tokenized using the DistilBERT tokenizer. Long reviews were truncated to a maximum length of 256 tokens. Dynamic padding was applied with `DataCollatorWithPadding`, so padding was added only at batch creation time.

Inside DistilBERT, each token ID is converted into a 768-dimensional embedding. The model combines token embeddings with positional embeddings, then processes them through Transformer encoder layers. A classification head is added on top of the pre-trained model to predict whether the review is negative or positive.

DistilBERT achieved the best performance in this project.

---

## Results

| Model | Validation Accuracy | Test Accuracy |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.8830 | 0.87628 |
| TensorFlow Hub + Dense | 0.8578 | 0.84560 |
| TensorFlow Hub USE + Dense | 0.8563 | 0.85752 |
| TensorFlow Hub NNLM 128D + Dense | 0.8732 | 0.85992 |
| BiLSTM | 0.8568 | 0.84264 |
| Improved BiLSTM | 0.8785 | 0.86260 |
| DistilBERT | **0.9062** | **0.90764** |

DistilBERT achieved the best overall performance, with a test accuracy of **90.76%**.

Compared to the strongest classical baseline, TF-IDF + Logistic Regression, DistilBERT improved test accuracy by approximately **3.14 percentage points**..

---

## Key Findings

The main findings of this project are:

1. **TF-IDF + Logistic Regression is a strong baseline**  
   The classical machine learning model achieved strong results and outperformed several deep learning models.

2. **TensorFlow Hub embeddings did not outperform TF-IDF**  
   Although pre-trained embeddings provide dense semantic representations, the tested TensorFlow Hub models did not improve over the TF-IDF baseline.

3. **BiLSTM models were not the best fit for this setup**  
   The BiLSTM models learned sequential patterns but did not generalize as well as TF-IDF or DistilBERT.

4. **DistilBERT achieved the best performance**  
   Fine-tuning a pre-trained Transformer model led to the highest validation and test accuracy.

5. **Model comparison is essential**  
   This project highlights the importance of comparing classical machine learning, deep learning, and Transformer-based methods instead of assuming that more complex models are always better.

---

## Embedding Visualization

To better understand how neural models represent text, embedding visualizations were performed on sample reviews.

For the BiLSTM model, the output of the Embedding layer was inspected for one review. The review was first converted into integer token IDs using the `TextVectorization` layer. Then, each token ID was transformed into a dense vector representation.

The embedding output for one review has the following shape:

```text
(sequence_length, embedding_dim)
```

A heatmap was used to visualize part of the embedding matrix, where:

- each row corresponds to a token,
- each column corresponds to an embedding dimension,
- each color intensity corresponds to an embedding value.

PCA was also applied to reduce the token embeddings from 128 dimensions to 2 dimensions, allowing the token representations to be plotted in a 2D space.


---

## Project Structure

```text
imdb-sentiment-classification-comparative-study/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── imdb_sentiment_classification_comparative_study.ipynb
│
├── src/
│   └── app.py
│
├── results/
│   ├── model_comparison.png
│   ├── confusion_matrix_distilbert.png
│   └── sample_embedding_visualization.png
│
└── assets/
    └── project_overview.png
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/imdb-sentiment-classification-comparative-study.git
cd imdb-sentiment-classification-comparative-study
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

The main libraries used in this project are:

```text
numpy
pandas
matplotlib
scikit-learn
tensorflow
tensorflow-datasets
tensorflow-hub
tf-keras
torch
transformers
datasets
evaluate
accelerate
streamlit
```

---

## How to Run

The main notebook is located in:

```text
notebooks/imdb_sentiment_classification_comparative_study.ipynb
```

You can run it locally or in Google Colab.

The notebook includes:

- dataset loading,
- exploratory data analysis,
- TF-IDF baseline,
- TensorFlow Hub models,
- BiLSTM models,
- DistilBERT fine-tuning,
- model comparison,
- confusion matrices,
- embedding visualization.

---

## Future Improvements

Possible future improvements include:

- Fine-tuning a larger Transformer model such as BERT or RoBERTa
- Performing hyperparameter optimization for DistilBERT
- Adding explainability methods for text classification
- Deploying the final model with Streamlit or Gradio
- Saving and loading the best model for inference
- Testing the pipeline on other sentiment analysis datasets
- Adding error analysis by review length and sentiment confidence

---

## Conclusion

This project compared multiple NLP approaches for IMDB sentiment classification.

The classical TF-IDF + Logistic Regression model achieved strong performance and proved to be a competitive baseline. However, the fine-tuned DistilBERT model achieved the best overall results, reaching **90.76% test accuracy**.

The project demonstrates that while classical NLP methods remain highly effective, Transformer-based models can provide superior performance when fine-tuned on the target task.
