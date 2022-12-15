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

1. Textual modality - 
...(1.text dataset + comments dataset to combine post's title and its comments to give more meaningful insights to data.)
...2.text dataset alone with "clean-title" as main text feature.
2. Visual modality -  
...(image data for posts that have images as attachement from metadata dataset.)
3. Mutimodalality - 
...(text + image data merged using text dataset and image dataset based on the "submission id" of each post if it has image.)
