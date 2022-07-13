# Project: Cassava Leaf Disease Classification
Identify the type of disease present on a Cassava Leaf image

### Use Case
Cassava is the second-largest source of carbohydrates in Africa and commonly grown by smallholder farmers because it can overcome harsh conditions. It is estimated that around 80% of smallhold farms in Sub-Saharan Africa grow cassava, however; the crop often yields poorly due to viral diseases. With the use of data science techniques, it may be possible to automate the process of identifying common cassava diseases, so they can be quickly treated.

Generally, farmers have to solicit the help of government-funded agricultural experts to visually inspect the plants. This is time-consuming, labor-intensive, and there may not be an expert available at all times. Additionally, the pictures that are supplied by African farmers may be of poor quality, since many only have access to poor quality cameras and low-bandwitch.

This classification problem contains 21,367 labeled images collected during a regular survey in Uganda. Most images were crowdsourced from farmers taking photos of their gardens and annotated by experts at the National Crops Resources Research Institute (NaCCRI) in collaboration with the AI lab at Makerere University, Kampala.

The task is to classify each cassava image into four disease categories or a fifth category indicating a healthy leaf. This will help farmers to quickly identify diseased plants before any damage is done to their crop.


### Project Description

To find a solution to the cassava classification problem we performed the following steps:

1) Fine-tuned ResNet18 on imbalanced cassava leaf data
2) Fine-tuned ResNet18 on imbalanced cassava leaf data using minimum/10 learning rate
4) Used data augmentation to balance classification dataset
 - Transformations used: horizontal flip, vertical flip, center crop, rotation, perspective, and color jitter.
3) Fine-tuned ResNet18 on balanced cassava leaf data using minimum/10 learning rate
4) Created baseline model that uses average tensor of each class to generate predictions
 - Note: Only 500 images were used in each class due to limited resouurces
5) Compared performance of all models
 - Calculated ROC curve and AUC
 - Calculated recall, precision, and f1-score

ResNet18 is described in this [paper](https://arxiv.org/pdf/1512.03385.pdf).

### Running Code

The entire project was developed on a `g4dn.xlarge` instance on Amazon using Ubuntu 20.04.
For in-depth instructions on how to properly install all libraries please refer to step 6 [here](https://course.fast.ai/start_aws).

Files:
- `fastai_computer_vision.ipynb`: Jupyter notebook containing entire project exploratory data analysis and modeling results

### Conclusion

Our recommendation is that the NaCCRI and Makerere University implement a ResNet18 model fine-tuned on augmented data to help automate detection of cassava leaf diseases. Although the augmentated data did not significantly change the performance of the fine-tuned model on test data, we suspect this model is more likely to better generalize to future datasets. It has been [shown](https://arxiv.org/pdf/1904.12848.pdf) that models trained on augmented data tend to generalize better to different datasets. 

The ResNet18 model fine-tuned on augmented data was able to achieve an accuracy of 87%, which is only 4% lower than the state-of-the-art on Kaggle. Of course, this does not consider the precision and recall of each class in our test set; however, our model was able to perform almost as well as the best Kaggle model with very little data preperation. Our model most struggled to identify healthy leaves and leaves with cassava bacterial blight (CBB), and it generally performed well across the rest of the classes. This is especially concerning because our model reported a precision of 0.65 for predicting healthy leaves; this metric means that out of all of the images we predicted as healthy, we only got 65% of them right. In this case, we may want our model to return a message that our prediction is not confident enough if we are predicting a leaf is healthy and it is below a certain discrimination threshold (a lower descrimination threshold with a lower false positive rate). In other words, we want to avoid predicting that a plant is healthy when it actually has a disease; the consequences of such errors could be devastating to small-scale farmers.

If NaCCRI and Makerere University wishes further improve the performance of the model we are proposing, we would need to conduct a second exploratory data analysis that is more focused on leveraging biological data. One thing that was missing in this analysis was that we created these models without much knowledge of the biology behind the disease. Perhaps after gaining a better biological understanding of cassava diseases, we may be able to more effectively employ data augmentation and other techniques. For example, when applying transformations to images in the data set, we may have been choosing transformations obscure important features that are characteristic of each disease.
