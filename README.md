# Classification of Fashion-MNIST images
## Description:
The aim of this project is to train a ResNet model using the Fashion-MNIST dataset, which contains images of 10 different types of clothing.
- **Link to Fashion-MNIST dataset: https://github.com/zalandoresearch/fashion-mnist**

After training the model, the final layer was removed, and Principal Component Analysis (PCA) was applied to visualize the learned features. 
This approach allows for a visual representation of the separation between the different clothing categories, showing the model’s ability to tell them apart.
## ResNet-8 and ResNet-18:
In this project, two neural network architectures were implemented. The pretrained ResNet-18 model from torchvision and a ResNet-8 architecture, built from scratch, and not using any pretrained weights.
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
 contains data analysis
- **scores.txt -**
 contains saving accuracy scores
- **config.yaml -**
 contains paths to the files
- **dataset_dataloader.py -**
 contains dataset and dataloader
- **early_stopping.py -**
 contains the EarlyStopping class, which implements an early stopping mechanism during model training
- **network.py -**
 contains implementation of ResNet-8
- **training.py -**
 contains function responsible for model training and evaluation
- **pca.py -**
 contains feature extraction and visualization
- **test.py -**
 saves final accuracy
