from typing import List

from keras.models import Model
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from models.extract_features import Vgg16Features
from sklearn.decomposition import PCA
import pandas as pd
import pickle
import umap

import argparse

PATH = r'C:\Users\a829748\Studia\cell-painting-dashboard\new'


def extract_features(file: str, model: Model):
    img = load_img(file, target_size=(224, 224))
    img = np.array(img)
    tensor_img = img.reshape(1, 224, 224, 3)
    preprocessed_img = preprocess_input(tensor_img)
    features = model.predict(preprocessed_img, use_multiprocessing=True)
    return features


def load_data(path: str) -> List[str]:
    images: List[str] = []
    os.chdir(path)
    with os.scandir(path) as files:
        for file in files:
            images.append(file.name)
    return images


def generate_base_features(path: str) -> dict:
    features_dict = {}
    images = load_data(path)
    for cell in images:
        feat = extract_features(cell, Vgg16Features)
        features_dict[cell] = feat
    if args.save_base_features:
        a_file = open("features_dict.pkl", "wb")
        pickle.dump(features_dict, a_file)
        a_file.close()
    return features_dict


def features_preprocess(features_dict: dict):
    filenames = np.array(list(features_dict.keys()))
    feat = np.array(list(features_dict.values()))
    feat = feat.reshape(-1, 4096)
    pca = PCA(n_components=100, random_state=22)
    pca.fit(feat)
    data_after_pca = pca.transform(feat)
    standard_embedding = umap.UMAP(random_state=42).fit_transform(data_after_pca)

    return pd.DataFrame({'name': filenames, 'vector1': standard_embedding[:, 0], 'vector2': standard_embedding[:, 1]})


parser = argparse.ArgumentParser(
    description='Generating vectors of features.')
parser.add_argument('-f', '--features', action='store_true', default=False,
                    help='Load base feature vectors.')
parser.add_argument('-s', '--save-base-features', action='store_true', default=False,
                    help='Save based features ot file.')
parser.add_argument('-o', '--out', default='features',
                    help='Name of output file.')
parser.add_argument('-fp', '--features-path',
                    help='path to base features.')
parser.add_argument('-p', '--path',
                    help='path to directory of images.')

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.features:
        features_dict = generate_base_features(args.path)
    else:
        file = open(args.features_path, "rb")
        features_dict = pickle.load(file)
    result: pd.DataFrame = features_preprocess(features_dict)
    if args.out:
        result.to_csv(f'{args.out}.csv', index=False)
    else:
        result.to_csv('features.csv', index=False)
