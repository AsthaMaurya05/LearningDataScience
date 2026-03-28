# 🧠 ML → DL Learning Roadmap

> A structured, project-based journey from Classical Machine Learning to Advanced Deep Learning.  
> Every project is checked off as completed — following a deliberate sequence with mathematical foundations first.

---

## 📌 How to use this roadmap

- Work through each **Phase** in order — every phase builds on the previous one
- Check off `[ ]` boxes as you complete each project
- Each project folder should be committed to this repo under the matching phase directory
- Math foundations come **first**, before any code

---

## 🗂️ Repository structure

```
ml-dl-roadmap/
│
├── README.md                  ← This file
├── math-foundations/          ← Notes, notebooks on prerequisite math
│
├── phase-1-classical-ml/
│   ├── 01-house-price-prediction/
│   ├── 02-iris-classifier/
│   └── 03-customer-churn/
│
├── phase-2-data-skills/
│   ├── 04-titanic-survival/
│   ├── 05-fraud-detection/
│   └── 06-movie-recommender/
│
├── phase-3-ann/
│   ├── 07-mnist-from-scratch/
│   ├── 08-diabetes-prediction/
│   └── 09-stock-price-mlp/
│
├── phase-4-cnn/
│   ├── 10-cifar10-classifier/
│   ├── 11-dog-vs-cat/
│   └── 12-brain-tumor-detection/
│
├── phase-5-rnn-lstm/
│   ├── 13-stock-forecasting-lstm/
│   ├── 14-sentiment-analysis/
│   └── 15-text-generation-rnn/
│
├── phase-6-transformers/
│   ├── 16-text-classifier-bert/
│   ├── 17-question-answering/
│   └── 18-neural-translation/
│
├── phase-7-generative/
│   ├── 19-digit-gan/
│   ├── 20-face-vae/
│   └── 21-diffusion-model/
│
├── phase-8-vision/
│   ├── 22-object-detection-yolo/
│   ├── 23-image-segmentation-unet/
│   └── 24-pose-estimation/
│
├── phase-9-rl/
│   ├── 25-cartpole-qlearning/
│   ├── 26-atari-dqn/
│   └── 27-trading-agent-ppo/
│
├── phase-10-llms/
│   ├── 28-rag-chatbot/
│   ├── 29-finetune-llm-lora/
│   └── 30-image-captioning-clip/
│
└── capstone/
    └── 31-end-to-end-ml-system/
```

---

## 📐 Phase 0 — Math foundations (do this first)

> Before writing a single line of ML code, revise these mathematical concepts.  
> You don't need to master everything — get comfortable, then learn the rest as projects demand it.

```
Math Foundations
│
├── 1. Linear Algebra
│   ├── [ ] Vectors & dot product
│   ├── [ ] Matrix multiplication & transpose
│   └── [ ] Eigenvalues & eigenvectors (for PCA)
│
├── 2. Calculus
│   ├── [ ] Derivatives & chain rule
│   ├── [ ] Partial derivatives & gradient vectors
│   └── [ ] Gradient descent intuition
│
├── 3. Probability & Statistics
│   ├── [ ] Mean, variance, standard deviation
│   ├── [ ] Conditional probability & Bayes' theorem
│   └── [ ] MLE — Maximum Likelihood Estimation
│
├── 4. Key Mathematical Functions
│   ├── [ ] Sigmoid & softmax
│   ├── [ ] Logarithms (why log-loss?)
│   └── [ ] ReLU & activation functions
│
├── 5. Loss Functions
│   ├── [ ] MSE & MAE (regression)
│   ├── [ ] Cross-entropy (classification)
│   └── [ ] L1 / L2 regularization
│
└── 6. Evaluation Metrics
    ├── [ ] R², RMSE, MAE
    ├── [ ] Precision, Recall, F1, Confusion matrix
    └── [ ] Bias-variance tradeoff
```

---

## 🌿 Phase 1 — Classical ML foundations

> **Stack:** Python, NumPy, pandas, scikit-learn, matplotlib, seaborn  
> **Goal:** Learn the full ML workflow — EDA → preprocessing → train → evaluate → interpret

```
Phase 1 Flow
│
├── [ ] Project 1: House price prediction
│   ├── Dataset   : California Housing (sklearn)
│   ├── Model     : Linear Regression
│   ├── Concepts  : EDA, feature scaling, StandardScaler
│   ├── Metrics   : RMSE, MAE, R²
│   └── Upgrade   : Polynomial features, Ridge regression
│
├── [ ] Project 2: Iris flower classifier
│   ├── Dataset   : Iris (sklearn)
│   ├── Model     : Logistic Regression, KNN
│   ├── Concepts  : Multi-class classification, decision boundaries
│   ├── Metrics   : Accuracy, confusion matrix
│   └── Upgrade   : Compare KNN at different k values
│
└── [ ] Project 3: Customer churn prediction
    ├── Dataset   : Telco Customer Churn (Kaggle)
    ├── Model     : Decision Tree, Random Forest
    ├── Concepts  : Feature importance, overfitting, tree depth
    ├── Metrics   : Precision, Recall, F1
    └── Upgrade   : Hyperparameter tuning with GridSearchCV
```

**Key things you will learn in this phase:**
- The full data science workflow end to end
- How to read and clean a dataset with pandas
- Why feature scaling matters and when to apply it
- The difference between classification and regression
- How to interpret model coefficients and feature importances

---

## 🔧 Phase 2 — Data skills & feature engineering

> **Stack:** pandas, scikit-learn, imbalanced-learn, surprise  
> **Goal:** Get good at data manipulation, handling messy real-world datasets, and building features

```
Phase 2 Flow
│
├── [ ] Project 4: Titanic survival prediction
│   ├── Dataset   : Titanic (Kaggle)
│   ├── Model     : Logistic Regression, Random Forest
│   ├── Concepts  : Missing data imputation, feature engineering
│   ├── Metrics   : Accuracy, F1
│   └── Upgrade   : One-hot encoding, name-based features
│
├── [ ] Project 5: Credit card fraud detection
│   ├── Dataset   : Credit Card Fraud (Kaggle)
│   ├── Model     : Logistic Regression, XGBoost
│   ├── Concepts  : Imbalanced datasets, SMOTE, class weights
│   ├── Metrics   : AUC-ROC, Precision-Recall curve
│   └── Upgrade   : Isolation Forest for anomaly detection
│
└── [ ] Project 6: Movie recommendation system
    ├── Dataset   : MovieLens 100K
    ├── Model     : SVD (Collaborative Filtering)
    ├── Concepts  : User-item matrix, latent factors, cold start
    ├── Metrics   : RMSE on rating predictions
    └── Upgrade   : Content-based filtering with TF-IDF
```

---

## 🧬 Phase 3 — Intro to neural networks (ANN)

> **Stack:** NumPy, PyTorch, TensorFlow/Keras  
> **Goal:** Understand how neural networks work from first principles before using frameworks

```
Phase 3 Flow
│
├── [ ] Project 7: MNIST digit classifier (from scratch)
│   ├── Dataset   : MNIST (torchvision / keras)
│   ├── Model     : 2-layer ANN built in NumPy
│   ├── Concepts  : Forward pass, backprop, weight updates
│   ├── Metrics   : Accuracy
│   └── Upgrade   : Rebuild in PyTorch, add dropout
│
├── [ ] Project 8: Diabetes prediction
│   ├── Dataset   : Pima Indians Diabetes (Kaggle)
│   ├── Model     : Feedforward Neural Network (PyTorch)
│   ├── Concepts  : Sigmoid output, BCE loss, Adam optimizer
│   ├── Metrics   : Accuracy, AUC-ROC
│   └── Upgrade   : Batch normalization, learning rate scheduler
│
└── [ ] Project 9: Stock price regression (MLP)
    ├── Dataset   : Yahoo Finance via yfinance
    ├── Model     : MLP with Keras/TensorFlow
    ├── Concepts  : Regression with NN, early stopping, callbacks
    ├── Metrics   : RMSE, MAE
    └── Upgrade   : Feature engineering with technical indicators
```

> ⚠️ **Before this phase:** Make sure you understand derivatives, chain rule, and gradient descent from Phase 0.

---

## 🖼️ Phase 4 — Convolutional neural networks (CNN)

> **Stack:** PyTorch, torchvision, OpenCV, albumentations  
> **Goal:** Learn to work with image data and understand spatial feature extraction

```
Phase 4 Flow
│
├── [ ] Project 10: CIFAR-10 image classifier
│   ├── Dataset   : CIFAR-10 (torchvision)
│   ├── Model     : Custom CNN from scratch
│   ├── Concepts  : Conv layers, pooling, feature maps, padding
│   ├── Metrics   : Top-1 accuracy
│   └── Upgrade   : Batch norm, dropout, data augmentation
│
├── [ ] Project 11: Dog vs cat classifier
│   ├── Dataset   : Dogs vs Cats (Kaggle)
│   ├── Model     : Transfer learning — ResNet-18 / VGG-16
│   ├── Concepts  : Pretrained weights, fine-tuning, freeze layers
│   ├── Metrics   : Accuracy, loss curves
│   └── Upgrade   : EfficientNet, Grad-CAM visualization
│
└── [ ] Project 12: Brain tumor MRI detection
    ├── Dataset   : Brain Tumor MRI (Kaggle)
    ├── Model     : CNN + Transfer learning
    ├── Concepts  : Medical imaging, class imbalance, augmentation
    ├── Metrics   : Sensitivity, specificity, AUC
    └── Upgrade   : Grad-CAM to explain predictions
```

---

## 🔁 Phase 5 — Recurrent networks & time series (RNN / LSTM)

> **Stack:** PyTorch, TensorFlow/Keras, pandas  
> **Goal:** Handle sequential and time-ordered data

```
Phase 5 Flow
│
├── [ ] Project 13: Stock price forecasting
│   ├── Dataset   : Yahoo Finance (AAPL or any stock)
│   ├── Model     : LSTM (PyTorch)
│   ├── Concepts  : Sequences, hidden state, BPTT, look-back window
│   ├── Metrics   : RMSE, MAE
│   └── Upgrade   : Stacked LSTM, attention mechanism
│
├── [ ] Project 14: Sentiment analysis
│   ├── Dataset   : IMDB Movie Reviews
│   ├── Model     : Bidirectional LSTM
│   ├── Concepts  : Word embeddings, GloVe, padding/masking
│   ├── Metrics   : Accuracy, F1
│   └── Upgrade   : Pre-trained embeddings (GloVe 100D)
│
└── [ ] Project 15: Text generation (character-level RNN)
    ├── Dataset   : Shakespeare / any large text corpus
    ├── Model     : Char-level RNN (Vanilla or LSTM)
    ├── Concepts  : Language modeling, temperature sampling
    ├── Metrics   : Perplexity, qualitative output
    └── Upgrade   : Beam search decoding
```

---

## 🤖 Phase 6 — Attention & transformers (NLP)

> **Stack:** HuggingFace Transformers, PyTorch, datasets  
> **Goal:** Learn the transformer architecture and fine-tune pretrained language models

```
Phase 6 Flow
│
├── [ ] Project 16: Text classifier with BERT
│   ├── Dataset   : SST-2 / AG News / custom dataset
│   ├── Model     : BERT (bert-base-uncased) — fine-tuned
│   ├── Concepts  : Tokenization, [CLS] token, fine-tuning strategy
│   ├── Metrics   : Accuracy, F1
│   └── Upgrade   : DistilBERT for speed, RoBERTa for accuracy
│
├── [ ] Project 17: Question answering system
│   ├── Dataset   : SQuAD 2.0
│   ├── Model     : BERT / DistilBERT for extractive QA
│   ├── Concepts  : Span prediction, start/end logits
│   ├── Metrics   : Exact Match (EM), F1
│   └── Upgrade   : Build a simple RAG pipeline on top
│
└── [ ] Project 18: Neural machine translation
    ├── Dataset   : Multi30k (EN→DE) or OPUS
    ├── Model     : Seq2Seq transformer (trained from scratch)
    ├── Concepts  : Encoder-decoder, cross-attention, positional encoding
    ├── Metrics   : BLEU score
    └── Upgrade   : Label smoothing, beam search
```

> 💡 **Milestone:** After this phase you understand the architecture behind ChatGPT, BERT, and every modern NLP system.

---

## 🎨 Phase 7 — Generative models (GAN / VAE / Diffusion)

> **Stack:** PyTorch, torchvision  
> **Goal:** Learn to generate data — images, faces, new samples

```
Phase 7 Flow
│
├── [ ] Project 19: Digit generation with DCGAN
│   ├── Dataset   : MNIST / CIFAR-10
│   ├── Model     : Deep Convolutional GAN
│   ├── Concepts  : Generator, discriminator, adversarial loss
│   ├── Metrics   : FID score, visual quality
│   └── Upgrade   : Conditional GAN (cGAN) — generate specific digits
│
├── [ ] Project 20: Face generation with VAE
│   ├── Dataset   : CelebA
│   ├── Model     : Variational Autoencoder
│   ├── Concepts  : Encoder, decoder, latent space, KL divergence
│   ├── Metrics   : Reconstruction loss, interpolation quality
│   └── Upgrade   : Interpolate between two faces in latent space
│
└── [ ] Project 21: Denoising diffusion model
    ├── Dataset   : MNIST or CIFAR-10
    ├── Model     : DDPM with UNet backbone
    ├── Concepts  : Forward/reverse diffusion, noise schedules
    ├── Metrics   : FID score
    └── Upgrade   : Conditional generation with class labels
```

---

## 👁️ Phase 8 — Object detection & segmentation

> **Stack:** PyTorch, ultralytics (YOLOv8), OpenCV, albumentations  
> **Goal:** Go beyond classification — locate and outline objects in images

```
Phase 8 Flow
│
├── [ ] Project 22: Object detection with YOLO
│   ├── Dataset   : COCO subset or custom labeled dataset (Roboflow)
│   ├── Model     : YOLOv8 (ultralytics)
│   ├── Concepts  : Bounding boxes, anchors, NMS, IoU
│   ├── Metrics   : mAP@0.5, mAP@0.5:0.95
│   └── Upgrade   : Train on a custom dataset with Roboflow
│
├── [ ] Project 23: Image segmentation with UNet
│   ├── Dataset   : CarvanaDataset or medical image dataset
│   ├── Model     : UNet architecture
│   ├── Concepts  : Encoder-decoder, skip connections, pixel-wise loss
│   ├── Metrics   : IoU (Jaccard), Dice coefficient
│   └── Upgrade   : DeepLabV3+ for semantic segmentation
│
└── [ ] Project 24: Pose estimation
    ├── Dataset   : Live webcam or COCO keypoints
    ├── Model     : MediaPipe Pose
    ├── Concepts  : Keypoint detection, skeleton graph, confidence scores
    ├── Metrics   : PCK (Percentage of Correct Keypoints)
    └── Upgrade   : Build a real-time exercise counter app
```

---

## 🕹️ Phase 9 — Reinforcement learning

> **Stack:** OpenAI Gymnasium, PyTorch, stable-baselines3  
> **Goal:** Train agents to make decisions through trial and reward

```
Phase 9 Flow
│
├── [ ] Project 25: CartPole balancing agent
│   ├── Environment : CartPole-v1 (Gymnasium)
│   ├── Algorithm   : Q-learning (tabular → then DQN)
│   ├── Concepts    : State, action, reward, Q-value, epsilon-greedy
│   ├── Metrics     : Episode reward, convergence episodes
│   └── Upgrade     : Deep Q-Network (DQN) with replay buffer
│
├── [ ] Project 26: Atari game agent (DQN)
│   ├── Environment : Atari Breakout or Pong (Gymnasium)
│   ├── Algorithm   : DQN with experience replay & target network
│   ├── Concepts    : Convolutional state encoding, frame stacking
│   ├── Metrics     : Mean episode reward over last 100 episodes
│   └── Upgrade     : Double DQN, Dueling DQN
│
└── [ ] Project 27: Stock trading agent (PPO)
    ├── Environment : Custom trading env (FinRL or from scratch)
    ├── Algorithm   : Proximal Policy Optimization (PPO)
    ├── Concepts    : Policy gradient, actor-critic, value function
    ├── Metrics     : Cumulative return, Sharpe ratio
    └── Upgrade     : Multi-stock portfolio management
```

---

## 🚀 Phase 10 — LLMs & multimodal (advanced)

> **Stack:** HuggingFace, LangChain, OpenAI API, PEFT, FAISS  
> **Goal:** Work at the frontier — RAG systems, LLM fine-tuning, vision-language models

```
Phase 10 Flow
│
├── [ ] Project 28: RAG chatbot
│   ├── Stack     : LangChain + FAISS + OpenAI / Mistral
│   ├── Concepts  : Document chunking, embeddings, vector search
│   ├── Use case  : Chat with a PDF / documentation
│   └── Upgrade   : Add memory, multi-document retrieval
│
├── [ ] Project 29: Fine-tune an LLM with LoRA
│   ├── Model     : Mistral-7B or LLaMA-3 8B
│   ├── Stack     : HuggingFace PEFT, bitsandbytes, trl
│   ├── Concepts  : LoRA adapters, QLoRA (4-bit), instruction tuning
│   ├── Dataset   : Alpaca / custom Q&A dataset
│   └── Upgrade   : Deploy with vLLM or Ollama
│
└── [ ] Project 30: Multimodal image captioning
    ├── Model     : CLIP (OpenAI) + GPT-2 / T5
    ├── Concepts  : Vision-language alignment, contrastive learning
    ├── Dataset   : MS-COCO captions
    ├── Metrics   : BLEU, CIDEr
    └── Upgrade   : BLIP-2 or LLaVA fine-tuning
```

---

## 🏆 Capstone — End-to-end production ML system

> **Goal:** Take a full ML project from experiment to deployed, monitored production system.

```
Capstone Flow
│
├── [ ] Choose a real problem (classification or regression)
├── [ ] Experiment tracking with MLflow
├── [ ] Data versioning with DVC
├── [ ] Build a REST API with FastAPI
├── [ ] Containerize with Docker
├── [ ] Deploy to cloud (AWS / GCP / Railway / Render)
├── [ ] Model monitoring (data drift, performance degradation)
└── [ ] CI/CD pipeline with GitHub Actions
```

**Stack:** MLflow · DVC · FastAPI · Docker · GitHub Actions · any cloud provider

---

## 📊 Progress tracker

| Phase | Projects | Status |
|---|---|---|
| 📐 Math foundations | 6 topics | `[ ] Not started` |
| 🌿 Phase 1 — Classical ML | Projects 1–3 | `[ ] Not started` |
| 🔧 Phase 2 — Data skills | Projects 4–6 | `[ ] Not started` |
| 🧬 Phase 3 — ANN | Projects 7–9 | `[ ] Not started` |
| 🖼️ Phase 4 — CNN | Projects 10–12 | `[ ] Not started` |
| 🔁 Phase 5 — RNN/LSTM | Projects 13–15 | `[ ] Not started` |
| 🤖 Phase 6 — Transformers | Projects 16–18 | `[ ] Not started` |
| 🎨 Phase 7 — Generative | Projects 19–21 | `[ ] Not started` |
| 👁️ Phase 8 — Vision | Projects 22–24 | `[ ] Not started` |
| 🕹️ Phase 9 — RL | Projects 25–27 | `[ ] Not started` |
| 🚀 Phase 10 — LLMs | Projects 28–30 | `[ ] Not started` |
| 🏆 Capstone | Project 31 | `[ ] Not started` |

---

## 🛠️ Core tools & libraries

| Category | Tools |
|---|---|
| Data manipulation | `numpy` `pandas` |
| Visualization | `matplotlib` `seaborn` `plotly` |
| Classical ML | `scikit-learn` `xgboost` `lightgbm` |
| Deep learning | `pytorch` `tensorflow` `keras` |
| NLP / LLMs | `transformers` `datasets` `tokenizers` `peft` `langchain` |
| Computer vision | `torchvision` `opencv-python` `albumentations` `ultralytics` |
| RL | `gymnasium` `stable-baselines3` |
| MLOps | `mlflow` `dvc` `fastapi` `docker` |
| Experiment tracking | `wandb` `mlflow` |

---

## 📝 Per-project commit convention

Each project folder should contain:

```
project-name/
├── README.md          ← What the project does, dataset, results
├── notebook.ipynb     ← Main Jupyter notebook with full walkthrough
├── requirements.txt   ← Python dependencies
├── src/               ← Clean Python scripts (optional)
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
└── results/           ← Saved plots, metrics, model checkpoints
```

Commit message convention:
```
✅ Complete: [Project Name] — [key metric achieved]
🔧 WIP: [Project Name] — [what you're working on]
📝 Notes: [topic] — [what you learned]
```

