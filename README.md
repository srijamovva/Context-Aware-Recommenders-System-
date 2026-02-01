# Context-Aware-Recommenders-System-
This project studies Context-Aware Recommender Systems using deep learning models like NCF, NeuMF, and NeuCMF. We evaluate multiple context representation strategies, including PCC and neural embeddings, on TripAdvisor and Frappe datasets, showing that individual context embeddings improve recommendation accuracy.

Context-Aware Recommendation with Deep Learning

This project investigates how context representation impacts the performance of Context-Aware Recommender Systems (CARS). Traditional recommender systems often rely on limited, predefined contextual variables, which restrict their ability to adapt to real-world scenarios where user preferences vary dynamically.

We evaluate multiple deep learning–based recommendation models, including Neural Collaborative Filtering (NCF), Neural Matrix Factorization (NeuMF), and Neural Contextual Matrix Factorization (NeuCMF), across two real-world datasets: TripAdvisor and Frappe. To effectively integrate context, we explore three representation strategies:

Pearson Correlation Coefficient (PCC)

Combined context embeddings

Individual context embeddings with independent neural representations

Using Mean Absolute Error (MAE) as the evaluation metric and 5-fold cross-validation, our experiments demonstrate that neural embeddings consistently outperform binary encodings, and that modeling each context feature individually leads to the best performance, particularly in NeuCMF variants. The results highlight the importance of fine-grained context modeling for improving recommendation accuracy in complex, real-world environments.
