# Sentiment Analysis Project

This project focuses on sentiment analysis for customer messages using machine learning techniques. The goal is to classify customer messages as positive or negative based on their sentiment.

## Overview

- Perform data collection and preparation using the IMDB dataset available on Kaggle (https://www.kaggle.com/code/iamrahulthorat/sentiment-analysis-textblob-data-preprocessing).
- Preprocess the customer messages by removing noise, stopwords, and applying tokenization.
- Implement different models for sentiment analysis:
    - Support Vector Machine (SVM)
    - Random Forest (RF)
    - GRU-DNN (Gated Recurrent Unit - Deep Neural Network)
- Train the models on the training set and evaluate their performance on the test set.
    - Loss&Acc plot. 
- Load checkpoints for all type of models. 
- Data Pipeline for model usage on real time data. (Will be added)
- Containerize the final model using Docker for easy deployment and reproducibility. (Will be added)

## Project Structure

The project is structured as follows:

- `data/`: Contains the IMDb dataset and any other necessary data files.
- `src/data/`: Handles data loading, preprocessing, tokenization and other staff related to data preparationç
- `src/model/`: Contains model files and a factory code to model environment setup.
- `tools/train.py`: Trains the selected model on the training set.

## Usage

1. Clone the repository: \
git clone https://github.com/anilyerlikaya/Sentiment.git

2. (Optional) Create a new conda environment \
conda create -n sentiment python=3.10 \
conda activate sentiment

3. Install the required dependencies \
pip install -r requirements.txt

4. Train the Model \
python tools/train.py --trainset-path "path to the train set" --valset-path "path to the validation set" --model-name "svm" --train-size 5000 --test_size 1000 --epochs 300

## License

This project is licensed under the [MIT License](LICENSE).
