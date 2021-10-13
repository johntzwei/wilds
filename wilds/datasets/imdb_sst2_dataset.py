import ipdb
from tqdm import tqdm
import os
import torch
import pandas as pd
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

class IMDBandSST2Dataset(WILDSDataset):
    _dataset_name = 'mnli'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/',
            'compressed_size': 90_644_480}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)

        self._metadata_array = []
        self._text_array = []
        self._split_array = []

        self._y_array = []
        self._y_size = 1
        self._n_classes = 3

        split_to_idx = { 'train' : 0, 'validation' : 2, 'test' : 2}

        from datasets import load_dataset
        sst2 = load_dataset('glue', 'sst2')
        for split in ['train', 'validation']:
            print('Importing %s...' % split)
            for i, example in tqdm(enumerate(sst2[split])):
                assert(example['label'] >= 0)

                text = example['sentence']
                self._text_array.append(text)
                self._y_array.append(example['label'])

                if split == 'train' and i < 5000:
                    self._split_array.append(1)
                else:
                    self._split_array.append(split_to_idx[split])

                self._metadata_array.append((1, example['label']))

        imdb = load_dataset('imdb')
        for split in ['train', 'test']:
            print('Importing %s...' % split)
            for i, example in tqdm(enumerate(imdb[split])):
                assert(example['label'] >= 0)

                text = example['text']
                self._text_array.append(text)
                self._y_array.append(example['label'])

                if split == 'train' and i < 5000:
                    self._split_array.append(1)
                else:
                    self._split_array.append(split_to_idx[split])

                self._metadata_array.append((0, example['label']))
 
        self._y_array = torch.LongTensor(self._y_array)
        self._metadata_array = torch.LongTensor(self._metadata_array)
        self._split_array = torch.LongTensor(self._split_array)

        self._split_scheme = 'official'
        self._identity_vars = ['sst2_not_imdb']
        self._metadata_fields = ['sst2_not_imdb', 'y']

        self._eval_groupers = [
            CombinatorialGrouper(
                dataset=self,
                groupby_fields=[identity_var, 'y'])
            for identity_var in self._identity_vars]

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        return self._text_array[idx]

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels 
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric = Accuracy(prediction_fn=prediction_fn)
        results = {
            **metric.compute(y_pred, y_true),
        }
        results_str = f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"
        # Each eval_grouper is over label + a single identity
        # We only want to keep the groups where the identity is positive
        # The groups are:
        #   Group 0: identity = 0, y = 0
        #   Group 1: identity = 1, y = 0
        #   Group 2: identity = 0, y = 1
        #   Group 3: identity = 1, y = 1
        # so this means we want only groups 1 and 3.
        worst_group_metric = None
        for identity_var, eval_grouper in zip(self._identity_vars, self._eval_groupers):
            g = eval_grouper.metadata_to_group(metadata)
            group_results = {
                **metric.compute_group_wise(y_pred, y_true, g, eval_grouper.n_groups)
            }
            results_str += f"  {identity_var:20s}"
            for group_idx in range(eval_grouper.n_groups):
                group_str = eval_grouper.group_field_str(group_idx)
                if f'{identity_var}:1' in group_str:
                    group_metric = group_results[metric.group_metric_field(group_idx)]
                    group_counts = group_results[metric.group_count_field(group_idx)]
                    results[f'{metric.name}_{group_str}'] = group_metric
                    results[f'count_{group_str}'] = group_counts
                    if f'y:0' in group_str:
                        label_str = 'non_toxic'
                    else:
                        label_str = 'toxic'
                    results_str += (
                        f"   {metric.name} on {label_str}: {group_metric:.3f}"
                        f" (n = {results[f'count_{group_str}']:6.0f}) "
                    )
                    if worst_group_metric is None:
                        worst_group_metric = group_metric
                    else:
                        worst_group_metric = metric.worst(
                            [worst_group_metric, group_metric])
            results_str += f"\n"
        results[f'{metric.worst_group_metric_field}'] = worst_group_metric
        results_str += f"Worst-group {metric.name}: {worst_group_metric:.3f}\n"

        return results, results_str

if __name__ == '__main__':
    IMDBandSST2Dataset(download=True, root_dir='/home/johnny/al_datasets/data/')
