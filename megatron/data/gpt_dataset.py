# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""GPT style dataset."""

import hashlib
import os
import time

import numpy as np
import torch

from megatron import print_rank_0, get_args, get_tokenizer
from megatron.core import mpu
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples
from megatron.data.dataset_utils import get_train_valid_test_split_, get_split_by_range_
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.tokenizer.tokenizer import FIM_MIDDLE, FIM_PAD, FIM_PREFIX, FIM_SUFFIX


def build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                    train_valid_test_num_samples,
                                    seq_length, seed, skip_warmup,
                                    train_data_prefix=None,
                                    valid_data_prefix=None,
                                    test_data_prefix=None,
                                    return_doc_ids=False, *,
                                    data_cache_path=None):
    """Build train, valid, and test datasets."""

    # Single dataset.
    if data_prefix and len(data_prefix) == 1:
        print_rank_0("Single data path provided for train, valid & test")
        all_train_datasets, all_valid_datasets, all_test_datasets = _build_train_valid_test_datasets(data_prefix[0],
                                                data_impl, splits_string,
                                                train_valid_test_num_samples,
                                                seq_length, seed, skip_warmup,
                                                data_cache_path=data_cache_path)
    # Blending dataset.
    elif data_prefix:
        print_rank_0("Blending dataset for train, valid & test")
        output = get_datasets_weights_and_num_samples(data_prefix,
                                                    train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output
        
        train_num_samples, valid_num_samples, test_num_samples = map(
            sum,
            zip(*datasets_train_valid_test_num_samples)
        )

        # Build individual datasets.
        train_datasets = []
        valid_datasets = []
        test_datasets = []
        for i in range(len(prefixes)):
            train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
                prefixes[i], data_impl, splits_string,
                datasets_train_valid_test_num_samples[i],
                seq_length, seed, skip_warmup,
                return_doc_ids,
                data_cache_path=data_cache_path)
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)
        all_train_datasets = BlendableDataset(train_datasets, weights, train_num_samples, data_cache_path=data_cache_path) \
                            if train_datasets else None
        all_valid_datasets = BlendableDataset(valid_datasets, weights, valid_num_samples, data_cache_path=data_cache_path) \
                            if valid_datasets else None
        all_test_datasets = BlendableDataset(test_datasets, weights, test_num_samples, data_cache_path=data_cache_path) \
                            if test_datasets else None
    else:
        print_rank_0("Separate data paths provided for train, valid & test. Split string will be ignored.")

        train_dataset, valid_dataset, test_dataset = None, None, None
        # Single dataset.
        if train_data_prefix is not None:
            train_dataset = build_dataset("train", train_data_prefix, data_impl,
                                          splits_string,
                                          train_valid_test_num_samples[0],
                                          seq_length, seed, skip_warmup,
                                          data_cache_path=data_cache_path)

        if valid_data_prefix is not None:
            valid_dataset = build_dataset("valid", valid_data_prefix, data_impl,
                                          splits_string,
                                          train_valid_test_num_samples[1],
                                          seq_length, seed, False,
                                          data_cache_path=data_cache_path)


        if test_data_prefix is not None:
            test_dataset = build_dataset("test", test_data_prefix, data_impl,
                                         splits_string,
                                         train_valid_test_num_samples[2],
                                         seq_length, seed, False,
                                         data_cache_path=data_cache_path)

        return (train_dataset, valid_dataset, test_dataset)

    return all_train_datasets, all_valid_datasets, all_test_datasets



def build_dataset_group(dataset_group_name, paths, weights, splits, data_impl,
                        train_valid_test_num_samples,
                        seq_length, seed, skip_warmup, train_valid_test,
                        data_cache_path=None):
    '''
    Build a single dataset group corresponding to Option 2 of data loading see arguments.py
    a dataset group is passed on the following form
    GIVEN_NAME WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2, WEIGHT2 START:END PATH2
    or alternatively
    GIVEN_NAME PATH1    # for a single dataset to be used fully
    '''

    assert train_valid_test in ["train","valid","test"]
    index = ["train","valid","test"].index(train_valid_test)

    # Single dataset.
    if len(paths) == 1:
        dataset =  _build_single_datasets(paths[0],
                                          splits[0],
                                          data_impl,
                                          train_valid_test_num_samples,
                                          seq_length, seed, skip_warmup,
                                          dataset_group_name, train_valid_test,
                                          data_cache_path=data_cache_path)
        return dataset
    # Blending dataset.
    else:

        data_prefix = []
        # data_prefix is on the shape:
        # ["WEIGHT1", "PATH1", "WEIGHT2", "PATH2", "WEIGHT3", "PATH3"]
        for w,p in zip(weights, paths):
            data_prefix += [w,p]

        output = get_datasets_weights_and_num_samples(data_prefix,
                                                      train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        datasets = []
        for i in range(len(prefixes)):
            ds = _build_single_datasets(prefixes[i],
                                        splits[i],
                                        data_impl,
                                        datasets_train_valid_test_num_samples[i],
                                        seq_length,
                                        seed, skip_warmup,
                                        dataset_group_name, train_valid_test,
                                        data_cache_path=data_cache_path)

            # ds can be none if the dataset is so small that not a single document
            # is present in the split.
            assert ds is not None, \
                f"Got an empty split when trying to create dataset: {prefixes[i], splits[i]}"
            datasets.append(ds)
        all_datasets = BlendableDataset(datasets,
                                        weights,
                                        train_valid_test_num_samples[index],
                                        data_cache_path=data_cache_path)

        return all_datasets

def _build_single_datasets(data_prefix, range_string, data_impl,
                           train_valid_test_num_samples,
                           seq_length, seed, skip_warmup, dataset_group_name, train_valid_test,
                           data_cache_path=None):
    """Build a single dataset"""

    assert train_valid_test in ["train","valid","test"]
    index = ["train","valid","test"].index(train_valid_test)

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    # this corresponds to option2 for data loading on the form
    # WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2, WEIGHT3 START:END PATH3
    # splits here is an array of size 2  [start_index, end_index]
    splits = get_split_by_range_(range_string=range_string, size=total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    print_rank_0('    {}:'.format(dataset_group_name))
    print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[0], splits[1],
                                        splits[1] - splits[0]))

    def build_dataset(name):
        dataset = None
        if splits[1] > splits[0]:
            documents = np.arange(start=splits[0], stop=splits[1],
                                  step=1, dtype=np.int32)
            dataset = GPTDataset(name, data_prefix,
                                  documents, indexed_dataset,
                                  range_string,
                                  train_valid_test_num_samples[index],
                                  seq_length, seed,
                                  data_cache_path=data_cache_path)
        return dataset

    dataset = build_dataset(dataset_group_name)

    return dataset


def _build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                     train_valid_test_num_samples,
                                     seq_length, seed, skip_warmup,
                                     return_doc_ids=False, *,
                                     data_cache_path=None):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    # splits here is an array of size 4  [train_start_index, valid_start_index, test_start_index, test_end_index]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1],
                                  step=1, dtype=np.int32)
            dataset = GPTDataset(name, data_prefix, documents, indexed_dataset,
                                 splits_string,
                                 train_valid_test_num_samples[index],
                                 seq_length, seed,
                                 return_doc_ids,
                                 data_cache_path=data_cache_path)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


def build_dataset(dataset_name, data_prefix, data_impl,
                  splits_string, num_samples,
                  seq_length, seed, skip_warmup,
                  *,
                  data_cache_path=None):
    dataset = None
    if len(data_prefix) == 1:
        dataset = _build_dataset(dataset_name, data_prefix[0], data_impl,
                                 splits_string, num_samples, seq_length,
                                 seed, skip_warmup,
                                 data_cache_path=data_cache_path)
    else:
        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix, num_samples)
        prefixes, weights, dataset_num_samples = output
        num_samples = sum(dataset_num_samples)

        # Build individual datasets.
        datasets = []
        for i in range(len(prefixes)):
            ds = _build_dataset(dataset_name, prefixes[i], data_impl,
                                splits_string, dataset_num_samples[i],
                                seq_length, seed, skip_warmup,
                                data_cache_path=data_cache_path)
            if ds:
                datasets.append(ds)

        if datasets:
            dataset = BlendableDataset(datasets, weights, num_samples,
                                       data_cache_path=data_cache_path)

    return dataset


def _build_dataset(dataset_name, data_prefix, data_impl, splits_string,
                   num_samples, seq_length, seed, skip_warmup,
                   *,
                   data_cache_path=None):
    """
    Build dataset. This method is called when individual
    train, valid, test datasets are provided
    """

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]

    print_rank_0('    {}:'.format(dataset_name))
    print_rank_0('     document indices in [0, {}) total of {} '
                 'documents'.format(total_num_of_documents, total_num_of_documents))

    documents = np.arange(start=0, stop=total_num_of_documents,
                        step=1, dtype=np.int32)

    dataset = GPTDataset(dataset_name, data_prefix, documents, indexed_dataset,
                         splits_string, num_samples, seq_length, seed,
                         data_cache_path=data_cache_path)

    return dataset


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):
    """Build indexed dataset."""
    print_rank_0(' > building dataset index ...')

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix,
                                           data_impl,
                                           skip_warmup)
    print_rank_0(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))
    print_rank_0('    number of documents: {}'.format(
        indexed_dataset.sizes.shape[0]))

    return indexed_dataset


class GPTDataset(torch.utils.data.Dataset):

    def __init__(self, name, data_prefix, documents, indexed_dataset,
                 splits_string, num_samples, seq_length, seed,
                 return_doc_ids=False, *,
                 data_cache_path=None):

        self.name = name
        self.indexed_dataset = indexed_dataset
        self.return_doc_ids = return_doc_ids

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx, self.desc, self.desc_hash = \
            _build_index_mappings(self.name, data_prefix,
                                  documents, self.indexed_dataset.sizes,
                                  splits_string, num_samples, seq_length, seed,
                                  data_cache_path=data_cache_path)
        
        self.args = get_args()
        self.tokenizer = get_tokenizer()
        self.np_rng = np.random.RandomState(seed=seed) # rng state for FIM

        self.fim_rate = self.args.fim_rate
        self.fim_spm_rate = self.args.fim_spm_rate
        self.fragment_fim_rate = self.args.fragment_fim_rate
        self.fim_split_sample = self.tokenizer.vocab[self.args.fim_split_sample] if self.args.fim_split_sample is not None else None

        try:
            self.suffix_tok_id, self.prefix_tok_id, self.middle_tok_id, self.pad_tok_id = (self.tokenizer.special_tokens[tok] for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD])
        except KeyError:
            self.suffix_tok_id, self.prefix_tok_id, self.middle_tok_id, self.pad_tok_id = (self.tokenizer.vocab[tok] for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD])

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        doc_ids = []
        if doc_index_f == doc_index_l:
            doc_ids.append(self.doc_idx[doc_index_f])
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            doc_ids.append(self.doc_idx[doc_index_f])
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                                    offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                doc_ids.append(self.doc_idx[i])
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            doc_ids.append(self.doc_idx[doc_index_l])
            sample_list.append(self.indexed_dataset.get(
                self.doc_idx[doc_index_l],
                length=offset_l + 1))
            sample = np.concatenate(sample_list)
        
        # Code from: https://github.com/EleutherAI/gpt-neox/blob/FIM-clean/megatron/data/gpt2_dataset.py#L109
        # TODO(Hailey): can merge the code below this line with code above this line.
        # TODO(Hailey), cont: above already iterates through loop, so just add the permuting in there?
        sample = np.array(sample, dtype=np.int64)
        sample_len = sample.shape[0]
        # # print(sample, sample.shape)
        # # do FIM here, if enabled
        # TODO: Do we handle the following point from FIM paper?
        # To transform data in the character space for context-level FIM, the tokenized documents have to be decoded back into strings before FIM augmentation. Depending on the vocabulary, some care has to be given to ensure decoding does not introduce any spurious characters into training. For example, utf-8 characters are encoded as multiple tokens with a BPE vocabulary; they can result in fragments from chunking and fail to decode. To prevent unforeseen errors midway through training, we encourage checking for these fragments at the beginning or end of a context and removing them.
        eod = self.tokenizer.eod
        segment_breaks = np.argwhere(sample == eod) # split sample by document

        if self.fim_rate == 0:
            return sample.astype(np.int64)
    
        def fim_permute_sequence(sequence, rate):
            return permute(
                sequence,
                self.np_rng,
                rate,
                self.fim_spm_rate,
                self.tokenizer,
                truncate_or_pad=False,
                suffix_tok_id=self.suffix_tok_id,
                prefix_tok_id=self.prefix_tok_id,
                middle_tok_id=self.middle_tok_id,
                pad_tok_id=self.pad_tok_id,
            )

        def fim_split_and_permute_sequence(sequence):
            """
            If self.fim_split_sample is not None, split the sequence.
            Then apply FIM on the fragments, or the whole sequence if self.fim_split_sample is None.
            """
            if self.fim_split_sample is None:
                return fim_permute_sequence(sequence, self.fim_rate)
            # fim_split_sample is set: split the sample on this token and permute each fragment separately.
            # Typically, if each sample is a repository, then we split again on the file level.
            # Each fragment is a file, and we permute the files.
            fragment_breaks = np.argwhere(sequence == self.fim_split_sample)
            if fragment_breaks.shape == (0, 1):
                # no split token in this sample
                return fim_permute_sequence(sequence, self.fim_rate)
            if not self.np_rng.binomial(1, self.fim_rate):
                # don't do FIM preproc
                return sequence
            # Do FIM on each fragment
            curr_start_position = 0
            new_samples = []
            for loc in np.nditer(fragment_breaks):
                if loc - curr_start_position > 0:
                    permuted = fim_permute_sequence(sequence[curr_start_position:loc], self.fragment_fim_rate)
                    new_samples += [permuted, [self.fim_split_sample]]
                curr_start_position = loc + 1  # Jump over the split token
            # Permute the segment after the last split token
            permuted = fim_permute_sequence(sequence[curr_start_position:], self.fragment_fim_rate)
            new_samples.append(permuted)
            return np.concatenate(new_samples)

        if segment_breaks.shape != (0, 1):  # then there is an EOD token in this example
            curr_start_position = 0
            new_samples = []
            for loc in np.nditer(segment_breaks):
                # Only permute non-empty segments.
                if loc - curr_start_position > 0:
                    # permute {prefix, suffix, middle} or {suffix, prefix, middle}
                    permuted = fim_split_and_permute_sequence(sample[curr_start_position:loc])
                    new_samples += [permuted, [eod]]

                curr_start_position = loc + 1  # jump over the EOD token
            # Permute the segment after the last EOD
            permuted = fim_split_and_permute_sequence(sample[curr_start_position:])
            new_samples.append(permuted)

            sample = np.concatenate(new_samples)
        else:
            sample = fim_split_and_permute_sequence(sample)
            
        # Truncate or pad sequence to max-length
        diff = sample.shape[0] - sample_len
        if diff > 0: # too long
            sample = sample[:sample_len]
        elif diff < 0: # too short
            sample = np.concatenate([sample, np.full((-1 * diff), self.pad_tok_id)])

        assert sample.shape[0] == sample_len
        # end FIM-specific code
        if self.return_doc_ids: # for retro preprocessing
            return {'text': sample,
                    'doc_ids': np.array(doc_ids, dtype=np.int64)}
        else:
            return {'text': sample}


def _build_index_mappings(name, data_prefix, documents, sizes,
                          splits_string, num_samples, seq_length, seed,
                          *,
                          data_cache_path):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    print_rank_0(f' > Tokens per epoch: {tokens_per_epoch}')
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)

    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    desc = "GPT Dataset\n\n"
    desc += f"Data prefix {data_prefix}\n"
    desc += f"Dataset name {name}\n"
    desc += f"Number of samples {num_samples}\n"
    desc += f"Sequence length {seq_length}\n"
    desc += f"Random seed {seed}\n"
    desc += f"Split {splits_string}\n"
    desc_hash = hashlib.md5(desc.encode('utf-8')).hexdigest()
    desc_filename = desc_hash + ".dsc"
    doc_idx_filename = desc_hash + '_doc_idx.npy'
    sample_idx_filename = desc_hash + '_sample_idx.npy'
    shuffle_idx_filename = desc_hash + '_shuffle_idx.npy'

    # Look for cache in main data dir first to avoid unnecessary
    # duplication, then look in data-cache-path if specified,
    # If nothing is found, use the last path looked in
    build_indices = True
    prefixes = [os.path.join(os.path.dirname(data_prefix), 'index-cache')]
    if data_cache_path is not None:
        prefixes.append(data_cache_path)
    for prefix in prefixes:
        idx_path = {
            'desc': os.path.join(prefix, desc_filename),
            'doc': os.path.join(prefix, doc_idx_filename),
            'sample': os.path.join(prefix, sample_idx_filename),
            'shuffle': os.path.join(prefix, shuffle_idx_filename)
        }
        for f in idx_path.values():
            if not os.path.isfile(f):
                break
        else:
            # Found our files!
            build_indices = False
            break
    data_cache_dir = os.path.dirname(idx_path['desc'])
    data_cache_success = True

    # Build the indexed mapping if not exist.
    if build_indices and torch.distributed.get_rank() == 0:
        print_rank_0(' > WARNING: could not find index map files, building '
                     'the indices on rank 0 ...')

        # For the last epoch, decide whether include the entire epoch
        # in the global shuffle or not.

        # If we need only one epoch, then separating last epoch  does
        # not mean anything.
        if num_epochs == 1:
            separate_last_epoch = False
            print(' > only one epoch required, setting '
                  'separate_last_epoch to False', flush=True)

        else:
            # Get the number of samples for the last epoch
            num_samples_from_epochs_minus_one = (
                (num_epochs - 1) * tokens_per_epoch - 1) // seq_length
            last_epoch_num_samples = num_samples - \
                                     num_samples_from_epochs_minus_one
            assert last_epoch_num_samples >= 0, \
                'last epoch number of samples should be non-negative.'
            num_samples_per_epoch = (tokens_per_epoch - 1) // seq_length
            # For very small datasets, `last_epoch_num_samples` can be equal to
            # (num_samples_per_epoch + 1).
            assert last_epoch_num_samples <= (num_samples_per_epoch + 1), \
                'last epoch number of samples exceeded max value.'
            # If we have less than 80% of the samples for the last epoch,
            # seperate out the epoch and treat it differently.
            # Note: the 80% number is just based on common sense and can
            # be adjusted if needed.
            separate_last_epoch = (last_epoch_num_samples <
                                   int(0.80 * num_samples_per_epoch))
            if separate_last_epoch:
                string = ' > last epoch number of samples ({}) is smaller '\
                         'than 80% of number of samples per epoch ({}), '\
                         'setting separate_last_epoch to True'
            else:
                string = ' > last epoch number of samples ({}) is larger '\
                         'than 80% of number of samples per epoch ({}), '\
                         'setting separate_last_epoch to False'
            print(string.format(last_epoch_num_samples,
                                num_samples_per_epoch), flush=True)


        try:
            os.makedirs(data_cache_dir, exist_ok=True)

            # description
            with open(idx_path['desc'], 'wt') as fd:
                fd.write(desc)

            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng,
                                     separate_last_epoch)
            np.save(idx_path['doc'], doc_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save doc-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            # First compile and then import.
            from megatron.data import helpers
            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32
            sample_idx = helpers.build_sample_idx(sizes, doc_idx, seq_length,
                                                  num_epochs, tokens_per_epoch)
            np.save(idx_path['sample'], sample_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save sample-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1
            shuffle_idx = _build_shuffle_idx(num_samples_,
                                             sample_idx.shape[0] - 1, np_rng)
            np.save(idx_path['shuffle'], shuffle_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save shuffle-idx mapping'
                         ' (seconds): {:4f}'.format(time.time() - start_time))
        except OSError:
            print(f'There was an error trying to create the data cache directory ({data_cache_dir})')
            print('or a file in it. This defaults to a directory "index-cache" within the directory')
            print('the data files are in and can be set with the --data-cache-path argument. Please')
            print('ensure you have write access to this directory or specify one that you do have')
            print('write access to.')
            data_cache_success = False

    counts = torch.cuda.LongTensor([data_cache_success])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
    if counts[0].item() != (
        torch.distributed.get_world_size() //
        torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group())):
        print_rank_0("Data index creation unsuccessful, exiting.")
        exit()

    # Load mappings.
    start_time = time.time()
    print_rank_0(f" > loading doc-idx mapping from {idx_path['doc']}")
    doc_idx = np.load(idx_path['doc'], allow_pickle=True, mmap_mode='r')

    print_rank_0(f" > loading sample-idx mapping from {idx_path['sample']}")
    sample_idx = np.load(idx_path['sample'], allow_pickle=True, mmap_mode='r')

    print_rank_0(f" > loading shuffle-idx mapping from {idx_path['shuffle']}")
    shuffle_idx = np.load(idx_path['shuffle'], allow_pickle=True, mmap_mode='r')

    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(
        sample_idx.shape[0]))
    print_rank_0('    total number of epochs: {}'.format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx, desc, desc_hash


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence lenght, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0:len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs-1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))


def _build_sample_idx(sizes, doc_idx, seq_length,
                      num_epochs, tokens_per_epoch):
    """Sample index mapping is a 2D array with sizes
    [number-of-samples + 1, 2] where [..., 0] contains
    the index into `doc_idx` and [..., 1] is the
    starting offset in that document."""

    # Total number of samples. For -1 see comments in `_num_epochs`.
    num_samples = (num_epochs * tokens_per_epoch - 1) // seq_length
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)

    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Begining offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + 1
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += (remaining_seq_length + doc_length - 1)
                remaining_seq_length = 0
            else:
                # Otherwise, start from the begining of the next document.
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx


def _build_shuffle_idx(num_samples, total_size, np_rng):
    """Build the range [0, size) and shuffle."""
    print(' > building shuffle index with split [0, {}) and [{}, {}) '
          '...'.format(num_samples, num_samples, total_size), flush=True)

    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples,
                                  step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size,
                                 step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))


# From https://github.com/EleutherAI/gpt-neox/blob/FIM-clean/megatron/data/gpt2_dataset.py#L339
def permute(sample, np_rng, fim_rate, fim_spm_rate, tokenizer, truncate_or_pad=True,
            suffix_tok_id=None, prefix_tok_id=None, middle_tok_id=None, pad_tok_id=None):
    """
    Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation on it. 
    Maintain the same sample length (if transform creates a few extra tokens, drop them).
    """
    if np_rng.binomial(1, fim_rate): # sample bernoulli dist

        contents = tokenizer.detokenize(sample)
        
        try:
            # A boundary can be =0 (prefix will be empty)
            # a boundary can be =len(contents) (suffix will be empty)
            # The two boundaries can be equal (middle will be empty)
            boundaries = list(np_rng.randint(low=0, high=len(contents) + 1, size=2))
            boundaries.sort()
        except ValueError as e:
            print(len(contents), contents)
            print(e)
            raise e

        prefix = contents[:boundaries[0]]
        middle = contents[boundaries[0]:boundaries[1]]
        suffix = contents[boundaries[1]:]

        prefix = np.array([*tokenizer.tokenize(prefix)], dtype=np.int64)
        middle = np.array([*tokenizer.tokenize(middle)], dtype=np.int64)
        suffix = np.array([*tokenizer.tokenize(suffix)], dtype=np.int64)

        # here we truncate each given segment to fit the same length as it was before
        # A consequence is that we never reach the end of a file?
        # we should rather truncate at the context-level
        if truncate_or_pad:
            # need to make same length as the input. Take the 3 sentinel tokens into account
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - sample.shape[0]
            if diff > 0: # too long
                if suffix.shape[0] <= diff: # if there's no space to truncate the suffix: stop and report it. atm i should have stopped this from happening
                    return sample, np_rng
                suffix = suffix[:suffix.shape[0] - diff]
            elif diff < 0: # too short
                suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])
        
        if np_rng.binomial(1, fim_spm_rate):
            # SPM (variant 2 from FIM paper)
            new_sample = np.concatenate([
                [prefix_tok_id, suffix_tok_id], suffix,
                [middle_tok_id], prefix, middle
            ])
        else:
            # PSM
            new_sample = np.concatenate([
                [prefix_tok_id], prefix,
                [suffix_tok_id], suffix,
                [middle_tok_id], middle
            ])
        
    else:
        # don't do FIM preproc
        new_sample = sample

    return new_sample
