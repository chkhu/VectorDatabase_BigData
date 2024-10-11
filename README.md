# Large-scale Vector Database Retrieval Algorithm Optimization

## Overview

This project focuses on the development and optimization of algorithms for efficient and accurate retrieval of vectors from large-scale vector databases. It explores multiple methods, including FAISS (Facebook AI Similarity Search) and HNSW (Hierarchical Navigable Small World), to achieve high-performance vector similarity search.

## Project Goals

	•	Implement large-scale vector similarity search using FAISS and HNSW algorithms.
	•	Optimize the algorithms for speed and accuracy in databases with millions of high-dimensional vectors.
	•	Benchmark the performance on different datasets and evaluate accuracy and query time.

## Datasets

The project uses two datasets for evaluation:

	•	Dataset A:
	•	500 query vectors
	•	500,000 vectors in the database
	•	512-dimensional vectors
	•	Goal: Return the top 50 most similar vectors.
	•	Dataset B:
	•	5,000 query vectors
	•	5,000,000 vectors in the database
	•	Same dimensions and retrieval goal as Dataset A.

## Algorithms

The following algorithms were implemented and tested:

	1.	FAISS (Flat Index):
A brute-force search algorithm that compares each query vector against all vectors in the database, achieving high accuracy but at the cost of slower query times on larger datasets.
	2.	HNSW (via nmslib):
A graph-based approximate nearest neighbor algorithm that strikes a balance between search speed and accuracy by navigating through hierarchical layers of small-world graphs.
	3.	FAISS + IVF:
Inverted File Index (IVF) improves search efficiency by clustering vectors and searching within relevant clusters.
	4.	FAISS + HNSW64:
A hybrid approach using both FAISS and HNSW to accelerate search speed while maintaining accuracy.
	5.	FAISS + PQ:
Product Quantization (PQ) allows for efficient search with reduced memory usage, though at the expense of slight accuracy loss.

## Results

	•	FAISS Flat: Achieved 98%+ accuracy on both datasets with brute-force search.
	•	HNSW: Balanced speed and accuracy, with tunable parameters like M, efConstruction, and ef.
	•	FAISS + IVF: Improved speed with minor loss in accuracy.
	•	FAISS + HNSW64: Faster search times but with a slight decrease in accuracy.
	•	FAISS + PQ: Faster searches but with reduced accuracy compared to other methods.

<img width="936" alt="image" src="https://github.com/user-attachments/assets/3a8ba5f5-b023-433e-a7e3-7c2824ae1b78">

## Evaluation Metrics

	•	Accuracy: Measured by Precision at K (P@K).
	•	Speed: Average query time recorded for each algorithm.

## Setup and Installation

To run this project locally, follow these steps:

Prerequisites

	•	Python 3.x
	•	FAISS (install via pip install faiss-cpu or faiss-gpu for GPU support)
	•	NMSLIB (install via pip install nmslib)
	•	NumPy and other dependencies (install via pip install -r requirements.txt)

Installation

	1.	Clone the repository:

git clone https://github.com/your-username/vector-db-retrieval.git


	2.	Install the required dependencies:

pip install -r requirements.txt


	3.	Download or generate datasets and place them in the data/ directory.

Usage

To run the similarity search using FAISS or HNSW, use the following command structure:

	1.	Build the index:

from search_algorithms import build_index
index = build_index(data_vectors)


	2.	Run the search for K nearest neighbors:

from search_algorithms import find_k_similar
results = find_k_similar(query_vectors, index, k=50)


	3.	For HNSW, configure the search parameters:

from hnsw_search import hnsw_search
results = hnsw_search(query_vectors, k=50, ef=4000)



## Optimization Strategies

	•	Parallelization: Utilized GPU parallelization for FAISS to reduce query times.
	•	Parameter Tuning: Fine-tuned parameters like M, efConstruction, and ef for HNSW to balance accuracy and speed.
	•	Hybrid Indexing: Combined multiple indexing techniques (e.g., FAISS + IVF, FAISS + HNSW) to further enhance performance.

## Future Work

	•	Explore dynamic updates to the vector database for real-time querying.
	•	Investigate automatic parameter tuning using machine learning approaches.
	•	Implement hybrid solutions combining multiple algorithms for both fast screening and accurate retrieval.

License

This project is licensed under the MIT License - see the LICENSE file for details.

This README.md file provides a detailed explanation of the project, including its purpose, setup, and usage instructions, along with key technical details for anyone interested in contributing or using the code.
