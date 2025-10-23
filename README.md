# Natural Language Processing System for Consumer Complaint Classification

A machine learning and Natural Language Processing (NLP) solution designed to automatically categorize raw, unstructured consumer complaint narratives into standardized product or service categories, enabling efficient routing and large-scale trend analysis.

---

## Project Title & Short Description

**Title:** Automated Consumer Complaint Classification and Product Categorization

**Description:** This project implements a **Text Classification** pipeline using classic machine learning models (**Multinomial Naive Bayes** and **SGD Classifier**) to accurately map the consumer complaint text to its corresponding financial **Product** category based on the Consumer Financial Protection Bureau (CFPB) dataset.

---

## Problem Statement / Goal

The primary objective is to develop a high-throughput system for **automated complaint triage**. By accurately classifying incoming complaints into predefined **Product** categories, the system minimizes manual effort, accelerates the assignment of complaints to the correct department, and allows for rapid identification of emerging issues and risk areas related to specific products.

---

## Tech Stack / Tools Used

The project is implemented in Python and leverages industry-standard libraries for text processing and classical machine learning:

| Category | Tool / Library | Purpose |
| :--- | :--- | :--- |
| **Data Handling** | Pandas NumPy | Data loading cleaning and numerical operations |
| **Text Processing**| NLTK re string | Tokenization stopword removal and text cleaning (stopwords are imported) |
| **Feature Extraction**| CountVectorizer | Creating the Bag-of-Words (BoW) model for text features |
| **Modeling** | Scikit-learn | Training and evaluation of classification models (MNB, SGDClassifier) |

---

## Approach / Methodology

1.  **Data Acquisition and Cleaning**: The `complaints.csv` dataset is loaded. The pipeline includes dropping an irrelevant index column (`Unnamed: 0`) and removing rows with missing values (`dropna`) to prepare the data for training.
2.  **Text Preprocessing**: Libraries like **NLTK** and the `re` module are imported for cleaning the complaint narratives, typically involving lowercasing and eliminating **stopwords** and punctuation to prepare the text for modeling.
3.  **Feature Engineering**: Text features are converted into a sparse numerical matrix using the **CountVectorizer** (Bag-of-Words) approach.
4.  **Data Splitting**: The dataset is split into training and testing sets using `train_test_split`.
5.  **Model Implementation**: Two powerful and efficient classification algorithms are implemented and trained: **Multinomial Naive Bayes (MNB)** and the **Stochastic Gradient Descent (SGD) Classifier**.

---

## Results / Key Findings

* The project successfully establishes a foundational text classification pipeline, ready to handle large volumes of unstructured data.
* The model training focuses on comparing the effectiveness of a **probabilistic model (MNB)** against a **linear model (SGD)** for product categorization.
* The system is prepared to deliver high-accuracy classification predictions, enabling the automation of the complaint routing process.

---

## Topic Tags

NaturalLanguageProcessing NLP TextClassification MachineLearning MultinomialNB SGDClassifier ConsumerComplaints Scikit-learn NLTK

---

## How to Run the Project

### 1. Install Requirements

Install all necessary packages using the provided `requirements.txt` file and download the required NLTK data:

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
