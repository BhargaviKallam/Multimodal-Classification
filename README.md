# Introduction

Multimodal-Classification

One machine learning model may not be able to handle the complexity of the binary classification challenge required for real-world false post detection. One of the potential strategies to achive more accurate outcomes is in the employing of the multiple models to process several modalities (e.g. picture modality and textual modality) of the same datasource. Built hybrid text+image models and conducted extensive tests for numerous classification variations, highlighting the significance of the unique multimodality and fine-grained categorization capabilities.

The project's objective is to demonstrate how well the multimodal classification approach performs using a common real-world dataset. 
The three neural networks used in the newly presented multimodal classification approach are as follows: 

1. From the visual modality, the first neural network extracts image features. 
2. From the textual modality, the second neural network extracts text features.
3. The third neural network determines which of two modalities is somewhat more insightful and then performs classification with more emphasis on the more insightful modality and less emphasis on the less insightful modality.

# Dataset
 A sizable multimodal fake news dataset [(r/Fakeddit)](https://github.com/entitize/Fakeddit) with data from a wide range of sources, including text, images, metadata, and comments, has about 800,000 samples.
 
Four datasets are created for experimentation on different modality of data as follows,

1. Textual modality
    - Text dataset + comments dataset to combine post's title and its comments to give more meaningful insights to data.
    - Text dataset alone with "clean-title" as main text feature.
2. Visual modality 
    - image data for posts that have images as attachement from metadata dataset.
3. Mutimodalality
    - text + image data merged using text dataset and image dataset based on the "submission id" of each post if it has image.

# Folder Structure Conventions

```bash
.
|___ ...
|___ Data
|    |____df_test.pkl
|    |____df_train.pkl
|    |____df_validation.pkl
|    |____multimodal.pkl
|    |____test_featuremap.pkl
|    |____train_featuremap.pkl
|
|___ Text Classification(Comments+title)
|    |____Baseline_models(title+comments).ipynb        # Machine Learning Algorithms such as MLP Classifier, SGD, linear SVC etc.
|    |____CNN_model(title+comments).ipynb              # Convolutional neural networks with different word embeddings
|    |____Cleaning_data(title+comments).ipynb          # Text data preprocessing such as cleaning and removing unnecessary features and noise.
|    |____Data_analysis(title+comments).ipynb          # text analytics to find patterns across hundreds of texts and produces graphs.
|    |____RNN_models(title+comments).ipynb             # Recurrent neural networks namely LSTM and Bidirectional LSTM.
|
|___ classification(images).ipynb            # Image classificztion using basic ML models and applying oversampling, also used neural networks such as CNN, VGG16 and VGG19 for Transfer learning with fine-tuning.
|
|___ text_classification.ipynb              # Text classification using all the methods and models from above mentioned "Text classification" file without comments data.
|
|___ README.md
|
|___ requirements.txt
```

# 


