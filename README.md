# 📚 Next Word Prediction Using LSTM

![Model Architecture](images/CAPTUR.PNG)

This project applies Long Short-Term Memory (LSTM) networks to predict the **next word** in a given text sequence using Shakespeare’s *Hamlet*. The model learns the structure and patterns of language and generates the next word based on context.

---

## 🚀 Project Overview

Next Word Prediction is a core problem in Natural Language Processing (NLP), often used in autocomplete systems. In this project, we:

- Collected and saved Shakespeare's *Hamlet* using NLTK’s Gutenberg corpus  
- Preprocessed and tokenized the dataset into sequences  
- Built and trained an LSTM model using TensorFlow/Keras  
- Evaluated the model by generating predictions on sample sequences  
- Built a Streamlit app to interactively predict the next word  

---

## 🛠️ Tools & Technologies

| Tool             | Logo |
|------------------|------|
| Python           | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| Pandas           | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) |
| NumPy            | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) |
| TensorFlow       | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) |
| Keras            | ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white) |
| Streamlit        | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) |
| NLTK             | ![NLTK](https://img.shields.io/badge/NLTK-76AADB?style=for-the-badge&logo=nltk&logoColor=white) |
| Jupyter Notebook | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white) |

---

## 🧠 Model Architecture

- Embedding Layer  
- LSTM Layer 1  
- LSTM Layer 2  
- Dense Output Layer with Softmax Activation

---

## 📊 Dataset

- **Source**: Shakespeare's *Hamlet* from the Gutenberg corpus via NLTK  
- **Extraction Code**:

```python
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg

data = gutenberg.raw('shakespeare-hamlet.txt')
with open('data/hamlet.txt', 'w') as file:
    file.write(data)
