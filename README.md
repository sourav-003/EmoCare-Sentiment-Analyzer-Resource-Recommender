# EmoCare: AI Sentiment & Recommendation System for Healthcare

*<p align="center">(https://huggingface.co/spaces/Sourav-003/EmoCare)</p>*

##  Project Overview

This project is an end-to-end system designed to provide empathetic support for a vulnerable population: **cancer survivors and their caregivers**.

As outlined in the "EmoCare" project brief, this application has two primary functions:
1.  **Sentiment Analysis:** It uses a fine-tuned Large Language Model (LLM) to analyze the sentiment of a user's post, classifying it as `positive`, `neutral`, `negative`, or `very negative`.
2.  **Recommendation System:** Based on the predicted sentiment, the app provides a context-aware, empathetic response, recommending helpful resources such as crisis hotlines, support groups, or trusted medical information.

This repository contains the full source code for the fine-tuned model and the deployable Gradio web application.

##  Key Features

* ** Nuanced Sentiment Analysis:** The model is not a generic sentiment classifier. It is a `Bio_ClinicalBERT` model fine-tuned specifically on the "Mental Health Insights" dataset, enabling it to understand the complex, domain-specific language of patients and caregivers.
* ** Empathetic Recommendation System:** The app's primary goal is to help. It automatically suggests actionable resources, providing the most critical support (e.g., crisis hotlines) for users expressing the highest level of distress (`very negative`).
* ** Deployed & Shareable:** The entire pipeline is packaged into a user-friendly Gradio web app and deployed on Hugging Face Spaces for public demonstration.

##  Technical Methodology

This project's core challenge was not just training a model, but training one that performs well on a highly difficult and imbalanced dataset.

### 1. The Challenge: Severe Class Imbalance

The "Mental Health Insights" dataset is representative of real-world data: the vast majority of posts are `neutral` or `negative`, while truly `positive` or `very negative` (crisis) posts are rare.

| Sentiment | Post Count |
| :--- | :--- |
| neutral | ~4375 |
| negative | ~4112 |
| **very negative** | **~1155** |
| **positive** | **~750** |

A standard model trained on this data would become "lazy." It would learn to be very good at predicting `neutral` and `negative` but would fail to identify the rare `positive` or `very negative` posts—which are arguably the most important ones.

### 2. Model Choice: `emilyalsentzer/Bio_ClinicalBERT`

Instead of a generic model like `bert-base-uncased`, this project uses **`emilyalsentzer/Bio_ClinicalBERT`**. This model is pre-trained on a massive corpus of biomedical and clinical text (MIMIC-III), giving it a foundational understanding of medical terminology ("scanxiety," "chemo," "prognosis") right from the start.

### 3. The Solution: Weighted Loss Finetuning

To solve the class imbalance problem, I implemented a custom `WeightedLossTrainer`.

During training, this custom trainer applies **Class Weights** to the loss function. This effectively "forces" the model to pay significantly more attention to the rare `positive` and `very negative` samples, penalizing it heavily for misclassifying them.

This technique shifts the optimization goal from simple "accuracy" to a high **`f1_macro` score**, resulting in a more balanced, fair, and—most importantly—useful model that doesn't ignore minority classes.

##  Dataset

This project uses the publicly available **"Mental Health Insights: Vulnerable Cancer Survivors & Caregivers Data"**.

* **Source:** [Mendeley Data](https://data.mendeley.com/datasets/69dcnv2gzd/1)
* **DOI:** 10.17632/69dcnv2gzd.1
* **Citation:** Orchi, Irin Hoque; Tabassum, Nafisa; Hossain, Jaeemul; Tajrin, Sabrina; Alam, Iftekhar (2023), “Mental Health Insights: Vulnerable Cancer Survivors & Caregivers Data”, Mendeley Data, V1, doi: 10.17632/69dcnv2gzd.1

##  How to Run This Project Locally

These instructions assume you have `git` and `python` installed.

1.  **Clone the repository:**
    ```bash
    git clone [(https://github.com/sourav-003/EmoCare-Sentiment-Analyzer-Resource-Recommender)
    cd EmoCare-Sentiment-Analyzer-Resource-Recommender
    ```

2.  **Set up a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    This project requires the `transformers` library, `torch`, and `gradio`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    The `app.py` script will automatically load the model from the local `my-sentiment-model-biomed-weighted-final` directory.
    ```bash
    python app.py
    ```

5.  **Open the app:**
    Once it's running, open your web browser and go to `http://127.0.0.1:7860`.
