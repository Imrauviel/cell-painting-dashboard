from typing import List
from tqdm import tqdm
import cv2
from keras.models import Model
from keras.applications.vgg19 import preprocess_input
import numpy as np
import os
from models.extract_features import create_model
from sklearn.decomposition import PCA
import pandas as pd
import pickle
import umap

import argparse


def extract_features(file: str, model: Model):
    # print(file)
    path_df = []
    for ch in range(1, 5):
        path_df.append(file.replace('ch1', f'ch{ch}'))
    c,m,y,k = [cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (224, 224), cv2.INTER_LANCZOS4) for path in path_df]
    image = np.array([np.stack((c, m, y, k), axis=2)])
    image = preprocess_input(image)

    # img = load_img(file, target_size=(224, 224))
    # img = np.array(img)
    # tensor_img = img.reshape(1, 224, 224, 3)
    preprocessed_img = preprocess_input(image)
    features = model.predict(preprocessed_img, use_multiprocessing=True)
    return features


def load_data(path: str) -> List[str]:
    images: List[str] = []
    os.chdir(path)
    with os.scandir(path) as files:
        for file in files:
            if file.name.endswith('tiff'):
                images.append(file.name)
    return images


def generate_base_features(path: str, model) -> dict:
    features_dict = {}
    images = load_data(path)
    images = [i for i in images if 'ch1' in i]

    for cell in tqdm(images):
        feat = extract_features(cell, model)
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

    return pd.DataFrame({'Name': filenames, 'Vector1': standard_embedding[:, 0], 'Vector2': standard_embedding[:, 1]})


def prepare_info_df(df):
    df['Row'] = df['Name'].apply(lambda x: x[1:3]).apply(lambda x: chr(int(x) + 64))
    df['Column'] = df['Name'].apply(lambda x: x[4:6])
    df['F'] = df['Name'].apply(lambda x: x[7:9])
    df['Well'] = df['Row'] + df['Column']
    for channel_str in ['ch1', 'ch2', 'ch3', 'ch4']:  # here comes the problem
        df['Name'] = df['Name'].apply(lambda x: x.replace(channel_str, ''))
    df = df.drop_duplicates()
    well_df = pd.read_csv(args.path_info,
                          usecols=[0, 1, 2])
    df = df.merge(well_df)
    df.rename(columns={'Concentration [uM]': 'Concentration'}, inplace=True)
    return df


parser = argparse.ArgumentParser(
    description='Generating vectors of features.')
parser.add_argument('-f', '--features', default=None,
                    help='Load base feature vectors.')
parser.add_argument('-s', '--save-base-features', action='store_true', default=False,
                    help='Save based features ot file.')
parser.add_argument('-o', '--out', default='features',
                    help='Name of output file.')
parser.add_argument('-p', '--path',
                    help='Path to directory of images.')
parser.add_argument('-pinfo', '--path-info',
                    help='Path to additional csv.')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args.path_info)
    model = create_model()
    if not args.features:
        features_dict = generate_base_features(args.path, model)
    else:
        file = open(args.features, "rb")
        features_dict = pickle.load(file)
    result: pd.DataFrame = features_preprocess(features_dict)

    result = prepare_info_df(result)
    print(result)
    if args.out:
        result.to_csv(f'{args.out}.csv', index=False)
    else:
        result.to_csv('features.csv', index=False)
