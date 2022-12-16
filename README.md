# Introduction

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

#  Best Models
         
   The experimentation is done on different modality features using various models for comparision such as Machine Leaning models ('Logestic Regression','Stocatic Gradient Descent', 'LinearSVC', 'Random Forest', 'KNearestNeighbours', 'MLPClassifier'), convolutional Neural Networks with different embeddings ('GloVe','Keras'), Recurrent Neural Networks( 'LSTM', 'BI-LSTM') and Transferlearning ('VGG16', 'VGG19').
         
   For ML models on textual features techniques such as Oversampling (SMOTE) is used to balance the class districution. And also applied Hyper-tuning using GridserachCV which did not show any improvement in model performance.
         
   Evaluation is done using RepeatedKFold cross-validation and then tested on the validation dataset which is unseen to the train data.


| Modality                                | Model         | Accuracy  | F1_score  |
| --------------------------------------- |:-------------:|:---------:| ---------:|
| Textual features   (Comments+title)     | BI-LSTM       |       0.86|       0.79|
| Textual features                        | BI-LSTM       |       0.81|       0.82|
| Visual  features                        | are neat      |       0.74|       0.70|


# Multimodal-Classification

# Multimodal-Classification

  Although we obtained good results, the rest of misclassification is because of the inconsistent data source. Furthermore, due to a lack of computational resources, more advanced models such as ResNet could not be used for image processing. However, it is clear that image features degrade the performance of the better performing text models.

  But it is clearly seen that the multimodal classification is performing far better than image or text classification models.

| Modality                                | Model                                         | Accuracy  | F1_score  |
| --------------------------------------- |:---------------------------------------------:|:---------:| ---------:|
| Textual features   (title)              | BI-LSTM + 2 Dense layers                      |       0.74|       0.62|
| Visual features                         | VGG16 + 2 Dense layers                        |       0.61|       0.47|
**| Visual + Texual features                | Concate (BI-LSTM + VGG16) + 2 Dense layers    |       0.77|       0.73|**__





