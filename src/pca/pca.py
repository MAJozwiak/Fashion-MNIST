import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

class ResNet8FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(ResNet8FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        return self.features(x).view(x.size(0), -1)


def extractor(original_model,device):
    feature_extractor = ResNet8FeatureExtractor(original_model).to(device)
    feature_extractor.eval()
    return feature_extractor


def extract_features(test_loader, model, device):
    features_list = []
    labels_list = []
    images_list = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            features = model(inputs)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            images_list.append(inputs.cpu().numpy())


    features_array = np.vstack(features_list)
    labels_array = np.hstack(labels_list)
    images_array = np.vstack(images_list)
    return features_array, labels_array, images_array


def apply_pca(features: np.ndarray, labels: np.ndarray, images: np.ndarray, num_components=2):
    pca = PCA(n_components=num_components)
    reduced_features = pca.fit_transform(features)
    plt.figure(figsize=(15, 10))
    ax = plt.gca()
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='tab10', alpha=0.5)

    for i in range(len(reduced_features)):
        x, y = reduced_features[i, :]
        img = images[i]
        img = np.moveaxis(img, 0, -1)
        img = np.clip(img, 0, 1)

        imagebox = OffsetImage(img, zoom=0.5, cmap='gray')
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0.1)
        ax.add_artist(ab)

    plt.colorbar(scatter, label='Classes')
    plt.xlabel('1')
    plt.ylabel('2')
    plt.title('PCA of Fashion MNIST Features')
    plt.show()

