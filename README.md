# Classification of Fashion-MNIST images
## Description:
The aim of this project is to train a ResNet model using the Fashion-MNIST dataset, which contains images of 10 different types of clothing.
- **Link to Fashion-MNIST dataset: https://github.com/zalandoresearch/fashion-mnist**


## ResNet-8 and ResNet-18:
In this project, two neural network architectures were implemented. The pretrained ResNet-18 model from torchvision and a ResNet-8 architecture, built from scratch, and not using any pretrained weights.

- **ResNet-18:**

 ![ResNet-18](https://github.com/MAJozwiak/Fashion-MNIST/blob/train/screenshots/Structure-of-the-Resnet-18-Model.png)
## File structure and description:
```
.
└── best_model/
    │   ├── model.pth
└── data/
    └── ....
└──data_analysis/
    │   ├── fashion-MNIST.ipynb
└──scores/
    │   ├── scores.txt
└──sreenshots/
    └── ....
└──src/
    │   ├── main.py
    └──dataset_dataloader/
        │   ├── dataset_dataloader.py
    └──model/
        │   ├── early_stopping.py
        │   ├── network.py
        │   ├── network.pytraining.py
    └──pca/
        │   ├── pca.py
    └──test/
        │   ├── test.py
│   ├── config.yaml
```

- **model.pth -** 
 contains the best model, saved during training,
- **Fashion-MNIST.ipynb -**
 contains data analysis,
- **scores.txt -**
 contains saving accuracy scores,
- **config.yaml -**
 contains paths to the files,
- **dataset_dataloader.py -**
 contains dataset and dataloader,
- **early_stopping.py -**
 contains the EarlyStopping class, which implements an early stopping mechanism during model training,
- **network.py -**
 contains implementation of ResNet-8,
- **training.py -**
 contains function responsible for model training and evaluation,
- **pca.py -**
 contains feature extraction and visualization,
- **test.py -**
 saves final accuracy.

## Added elements to enhancing the training process:
 - **early stopping -**
 is a technique used to halt training when the model's performance on the validation set no longer improves, helping to prevent overfitting,
 - **sheduler -**
 is a technique that monitore the validation loss and reduce the learning rate if there is not improvment.

## Visualization of the learned features

After training the model, the final layer was removed to extract feature. Principal Component Analysis (PCA) was then applied to reduce the dimensionality of these extracted features and visualize them. This approach enables a visual representation of the separation between different clothing categories, demonstrating the model’s ability to tell them apart.
![myplot.png](screenshots%2Fmyplot.png)

## Score

Achieved the following accuracy results:
 - **ResNet-8 -**
 90.38%
 - **ResNet-18 -**
 89.52%