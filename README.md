# Adaptive Credit Card Fraud Detection with Concept Drift and Uncertainty (IEEE‑CIS)

This project implements an adaptive deep learning pipeline for credit card fraud detection on the IEEE‑CIS Fraud Detection dataset. It focuses on concept drift, verification latency (delayed labels), and uncertainty‑aware learning, then exposes the trained model through a simple Flask web app for demo purposes.[1][2]

***

## Project Overview

Real‑world fraud detection systems face three major challenges:  
- Concept drift – user behavior and fraud patterns change over time.[3][4]
- Verification latency – only a subset of transactions are reviewed quickly, so labels arrive with delay.[1]
- Imbalanced data – fraud is rare compared to legitimate transactions.[5]

This project addresses these issues by:

- Using a deep MLP model trained on the IEEE‑CIS dataset for fraud probability estimation.[2][5]
- Simulating a streaming setting using `TransactionDT`, where transactions arrive chronologically and labels are revealed after a fixed delay.[6][1]
- Applying online fine‑tuning when delayed labels arrive (continual learning).[4]
- Detecting concept drift with the ADWIN algorithm and triggering stronger adaptation when performance degrades.[7][8]
- Using Monte Carlo Dropout to estimate predictive uncertainty and prioritizing high‑uncertainty transactions for simulated “manual review” and model updates.[1]

A lightweight Flask web app is provided to score individual transactions using the trained model, suitable for demonstrations and UI integration.[9][10]

***

## Main Components

- Data and Preprocessing (Colab)  
  - Download and merge IEEE‑CIS transaction and identity tables.[2]
  - Select key features; handle missing values, encode categoricals, and standardize numerics.

- Deep Learning Model (PyTorch)  
  - MLP with BatchNorm, ReLU, and Dropout, trained using binary cross‑entropy and Adam.[5]
  - Monte Carlo Dropout for uncertainty estimation at inference.[1]

- Adaptive Learning Pipeline  
  - Streaming simulation based on `TransactionDT`.[6]
  - Delayed label queue to model verification latency.[1]
  - Online fine‑tuning on recent labeled samples (replay buffer).[4]
  - ADWIN drift detector over prediction error stream for drift‑triggered adaptation.[8][7]
  - Uncertainty‑gated update policy: high‑uncertainty samples are always used for updates.[1]

- Baselines Implemented  
  - Static MLP (one‑shot training, no updates).[5]
  - Periodic retraining with sliding window.[3]
  - Online fine‑tuning without drift detection.[4]
  - Drift‑triggered fine‑tuning without uncertainty.[7]

- Flask Web App  
  - Loads exported model weights and preprocessing artifacts.[10][9]
  - Simple HTML form to enter transaction features.  
  - Returns fraud probability and a textual risk assessment (for example, “Likely Fraud”).

***

## Evaluation

Experiments are run in a prequential manner: each streamed transaction is first scored, then (after delay) used for updating the model.[1]
Metrics are computed over time windows (for example, every 50k transactions):

- ROC‑AUC and PR‑AUC per time chunk.[5]
- Recall at fixed false positive rate (for example, FPR ≤ 5 percent).[3]
- Recovery time after detected drift events.[11][7]

Results show that the full method (delay‑aware, drift‑triggered, uncertainty‑gated) maintains more stable performance under drift compared to static and periodically retrained baselines, and uses manual review capacity more efficiently.[4][1]

***

## How to Use

1. Training and Experiments (Colab)  
   - Open the provided Colab notebook.  
   - Run data preprocessing, model training, and streaming experiments.  
   - Export `fraud_mlp.pth`, `scaler.pkl`, `encoders.pkl`, and `feature_cols.pkl`.

2. Web App  
   - Place exported artifacts in `model/` and `artifacts/`.  
   - Install dependencies: `pip install flask torch scikit-learn joblib`.[9][10]
   - Run `python app.py` and open `http://127.0.0.1:5000/`.

***

## Academic Use

This repository supports a 6–8 page research paper structure:

- Introduction: fraud, concept drift, delayed labels, motivation.[4][1]
- Related Work: deep fraud detection, drift detection, uncertainty in deep learning.[7][5][1]
- Method: dataset, streaming simulation, model, drift detection, uncertainty.[2][6]
- Experimental Setup: splits, delay settings, baselines, metrics.[11][3]
- Results: plots and tables comparing baselines versus full method.  
- Discussion and Conclusion: implications, limitations, future reinforcement learning based extensions.[12][4]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/8beda63d-33b4-42ca-a415-d514d728e19e/2107.13508v1.pdf)
[2](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/63434d8b-fd7f-42fc-af26-75c36d3230d2/2409.13406v1.pdf)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/d3bb967b-9e79-4649-868b-3611e77dd5ae/2504.03750v1.pdf)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/f0e08a98-856d-47a8-9351-7f237694cd03/2012.03754v1.pdf)
[6](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203)
[7](https://riverml.xyz/dev/api/drift/ADWIN/)
[8](https://riverml.xyz/0.21.0/api/drift/ADWIN/)
[9](https://www.meritshot.com/example-deploying-a-pytorch-model/)
[10](https://www.python-engineer.com/posts/pytorch-model-deployment-with-flask/)
[11](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/c067868e-252b-4d0e-9f2a-c22257592b30/2506.10842v1.pdf)
[12](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/6343cf5e-cdd7-4136-9f09-645ff412e0c2/2504.08183v1.pdf)
[13](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/f7ee0cfe-0d9b-4232-9582-7737b7672cff/2503.22681v1.pdf)
[14](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/5ccaab56-9449-4bfb-a9d5-a7d52389e12f/1-s2.0-S187705092030065X-main.pdf)
[15](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/93020ccf-b05c-4fe9-8a6e-4e3f7f517871/2205.15300v1.pdf)
[16](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/8f62bf0e-03b9-442f-bb7b-d0dcfa847da6/doc.pdf)
