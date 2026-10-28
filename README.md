# Machine Learning — From Mathematics to Implementation

Welcome to my Machine Learning repository! 👋

This repository documents my journey of learning, understanding, and implementing machine learning algorithms, with a strong emphasis on the mathematical foundations behind them.

As a BSc Mathematics student building practical data science skills, I use this repository to connect mathematical theory with Python implementations. My goal is to understand how algorithms work internally, implement their core ideas using NumPy where applicable, and compare them with established implementations such as Scikit-learn where available.

Rather than treating machine learning models as black boxes, I aim to understand the mathematics, assumptions, calculations, and decision-making processes that make them work.

## 🎯 Objectives

- Understand the mathematical foundations of machine learning algorithms.
- Implement algorithms from scratch using Python and NumPy where applicable.
- Practise using Scikit-learn implementations and compare approaches where available.
- Develop practical experience with supervised and unsupervised learning.
- Strengthen my understanding of model training, prediction, and evaluation.
- Build a documented portfolio that demonstrates continuous learning and practical problem-solving.

## 📚 Algorithms and Topics

The repository is organized into folders dedicated to individual algorithms and concepts.

### 1. Supervised Learning

| Algorithm | Main concepts |
|---|---|
| [Linear Regression](linear_regression/) | Linear relationships, prediction, and error minimization |
| [Logistic Regression](logistic_regression/) | Probability, the sigmoid function, and binary classification |
| [k-Nearest Neighbour](k-nearest%20neighbour/) | Distance metrics, nearest neighbours, and classification |
| [Decision Trees](Decision%20Trees/) | Entropy, Gini impurity, information gain, and tree-based decisions |
| [Naive Bayes](Naive-Bayes/) | Bayes' theorem, conditional probability, and classification |
| [Random Forest](Random%20Forest/) | Bootstrap sampling, feature randomness, and ensemble learning |
| [Support Vector Machine](Support%20Vector%20Machine/) | Maximum-margin classification and decision boundaries |
| [XGBoost](XGBOOST/) | Gradient boosting, regularization, and sequential tree learning |
| [Neural Networks](neural_network/) | Weights, biases, activation functions, and learning |

### 2. Unsupervised Learning and Dimensionality Reduction

| Algorithm | Main concepts |
|---|---|
| [K-Means Clustering](K-Means%20clusterring/) | Centroids, cluster assignments, and iterative optimization |
| [Expectation-Maximization (EM)](EM%20(Expectation-maximization)/) | Latent variables, responsibilities, and parameter estimation |
| [Principal Component Analysis (PCA)](PCA%20(Principal%20Component%20Analysis/) | Variance, eigenvectors, eigenvalues, and dimensionality reduction |
| [Independent Component Analysis (ICA)](ICA%20(Independent%20Component%20Analysis)/) | Statistical independence and source separation |

## 🛠️ Technologies Used

- **Python** — primary programming language.
- **NumPy** — numerical computing and mathematical implementations.
- **Pandas** — data manipulation and preprocessing where needed.
- **Scikit-learn** — established machine learning implementations and model evaluation where applicable.
- **Matplotlib** — data visualization where needed.

## 🧠 Learning and Implementation Approach

My approach combines mathematical understanding with practical coding.

1. **Mathematical foundations:** Study the equations, assumptions, and reasoning behind an algorithm.
2. **Implementation:** Translate the mathematical concepts into Python and NumPy where applicable.
3. **Library implementation:** Use Scikit-learn or other relevant libraries where included.
4. **Testing and experimentation:** Explore model behaviour, predictions, and results using examples or datasets.
5. **Documentation:** Explain important concepts, implementation decisions, and observations as the work develops.

The availability of these steps varies by algorithm. Not every folder necessarily contains both a from-scratch implementation and a library-based version.

## ⚙️ Getting Started

### Prerequisites

You should have Python installed. A virtual environment is recommended for managing dependencies.

### Clone the repository

```bash
git clone https://github.com/lewiskagunya/machine_learning.git
cd machine_learning
```

### Create a virtual environment

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install the libraries

```bash
python -m pip install numpy pandas matplotlib scikit-learn
```

Some implementations may require additional dependencies. Install those as specified in the relevant algorithm folder.

### Explore an algorithm

Open the folder of the algorithm you want to study, read its code and documentation, and run the relevant Python script or notebook if provided.

For example, to explore Decision Trees:

```powershell
cd "Decision Trees"
```

Follow the instructions in that folder, if available, to run its implementation.

## 📈 Current Focus

My current focus is strengthening my understanding of machine learning algorithms through mathematical study, implementation, experimentation, and documentation.

I am particularly interested in understanding how different algorithms learn from data, how their mathematical foundations influence their behaviour, and how to evaluate their performance appropriately.

## 🚀 Future Improvements

As the repository develops, I intend to:

- Improve explanations of the mathematical foundations of each algorithm.
- Add more experiments using real-world datasets.
- Document model evaluation metrics and experimental results.
- Compare from-scratch implementations with established machine learning libraries.
- Include visual explanations of important mathematical concepts.
- Develop more complete, reproducible machine learning projects.

## 👨‍💻 About Me

I am a BSc Mathematics student developing practical skills in data science and machine learning.

My learning approach combines mathematics, programming, and experimentation. I am interested in understanding not only how to use machine learning algorithms, but also why they work and how their underlying principles can be implemented in code.

**GitHub:** [@lewiskagunya](https://github.com/lewiskagunya)

**Repository:** [machine_learning](https://github.com/lewiskagunya/machine_learning)

---

*This repository is an ongoing learning and implementation project. Its contents will evolve as I deepen my understanding of machine learning and develop more experiments and implementations.*
