import click
import yaml
from dataset_dataloader import dataset_dataloader
from model import training
from test import test
from pca import pca
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

@click.command()
@click.option('--config-path', default='../config.yaml', help='Path to the .yaml file')
def main(config_path) -> None:
    config = load_config(config_path)
    data_root = config['paths']['data']
    model_path = config['paths']['model_path']
    scores_path= config['paths']['scores_path']
    print(model_path)
    train_loader, test_loader,val_loader=dataset_dataloader.dataset_dataloader(data_root)
    model,device=training.train(train_loader, val_loader,model_path,False)
    test.test(test_loader,model,model_path,device,scores_path)

    feature_extractor = pca.extractor(model,device)
    features, labels,images = pca.extract_features(test_loader, feature_extractor, device)
    pca.apply_pca(features,labels,images, num_components=2)

if __name__ == "__main__":
    main()