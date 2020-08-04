import json
import os
import p_tqdm
import numpy as np
from tqdm import tqdm
import torch
from collections import defaultdict

def random_sampler(tensor, args):
    indices = torch.LongTensor(np.random.choice(tensor.size(0), args.num_images_per_mol)).to(args.device)
    return torch.index_select(tensor, 0, indices)

def mean_sampler(tensor, args):
    assert args.num_images_per_mol == 1
    return tensor.mean(dim=0).unsqueeze(0)

def max_sampler(tensor, args):
    assert args.num_images_per_mol == 1
    return tensor.max(dim=0).unsqueeze(0)

SAMPLER_REGISTRY = {
    'random': random_sampler,
    'mean': mean_sampler,
    'max': max_sampler,
}

class Images:
    def __init__(self, smiles2images, args):
        self.smiles2images = smiles2images
        for smiles, tensor in self.smiles2images.items():
            self.smiles2images[smiles] = tensor.to(args.device)
        self.args = args
        self.feature_size = list(smiles2images.values())[0].size(-1)

    @staticmethod
    def from_pickle(args):
        smiles2images = torch.load(args.smiles_to_images_pickle_path)
        return Images(smiles2images, args)

    @staticmethod
    def from_image_directory(args):
        with open(args.image_metadata_json_path) as f:
            metadata = json.load(f)

        smiles2paths = defaultdict(list)

        for smiles, val in tqdm(metadata.items()):
            for image in val['images']:
                smiles2paths[smiles].append(os.path.join(args.image_directory, image['sample_key'] + ".npy"))

        def read_images(tup):
            key, paths = tup
            return key, [np.load(path) for path in paths]

        smiles_images = map(read_images, smiles2paths.items())

        smiles2images = {k : torch.Tensor(v) for (k, v) in tqdm(smiles_images)}

        return Images(smiles2images, args)

    def get_item(self, smiles_list):
        return torch.cat([SAMPLER_REGISTRY[self.args.image_sampler_name](self.smiles2images[smiles], self.args).unsqueeze(0) for smiles in smiles_list], dim=0)

    def get_feature_size(self):
        return self.feature_size
