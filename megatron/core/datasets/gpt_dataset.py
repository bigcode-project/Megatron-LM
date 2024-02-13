# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy
import torch

from megatron import get_args, get_tokenizer
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
from megatron.core.datasets.megatron_dataset import MegatronDataset, MockDataset
from megatron.core.datasets.utils import Split, log_single_rank
from megatron.tokenizer.tokenizer import FIM_MIDDLE, FIM_PAD, FIM_PREFIX, FIM_SUFFIX

logger = logging.getLogger(__name__)


@dataclass
class GPTDatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration object for Megatron Core GPT datasets

    Attributes:          
        reset_position_ids (bool): Option to reset the position IDs in the dataset at an interval

        reset_attention_mask (bool): Option to reset the attention mask from the dataset

        eod_mask_loss (bool): Option to enable the EOD mask loss

        vocab_size (int): Size of vocabulary
      
    """

    reset_position_ids: bool = None

    reset_attention_mask: bool = None

    eod_mask_loss: bool = None

    vocab_size: int = sys.maxsize

    def __post_init__(self) -> None:
        """Do asserts and set fields post init
        """
        super().__post_init__()

        assert self.tokenizer is not None

        assert self.reset_position_ids is not None
        assert self.reset_attention_mask is not None
        assert self.eod_mask_loss is not None


class MockGPTDataset(MockDataset):
    """The mock GPT dataset
    """

    def __getitem__(self, idx: int) -> Dict[str, numpy.ndarray]:
        """Return a sequence_length + 1 token sequence consisting of the following:
            - (1) S, the RNG length-sentinel in the range [0, sequence_length)
            - (S) tokens
            - (1) end of document token
            - (sequence_length - S - 1) padding tokens

        Args:
            idx (int): The integer seed for mock data generation

        Returns:
            Dict[str, numpy.ndarray]: The mock data
        """
        tok = 1
        pad = 2
        eod = 0

        rng = numpy.random.default_rng(seed=[self.split.value, idx])
        length = rng.integers(low=0, high=self.config.sequence_length)
        sample_toks = numpy.zeros(length) + tok
        sample_pads = numpy.zeros(self.config.sequence_length - length - 1) + pad
        sample = numpy.int64(numpy.concatenate([[length], sample_toks, [eod], sample_pads]))

        text = torch.from_numpy(sample).long()
        labels = text[1:].contiguous()
        tokens = text[:-1].contiguous()

        attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
            tokens,
            eod,
            self.config.reset_position_ids,
            self.config.reset_attention_mask,
            self.config.eod_mask_loss,
        )

        return {
            "tokens": tokens,
            "labels": labels,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }


class GPTDataset(MegatronDataset):
    """The base GPT dataset

    Args:
        indexed_dataset (MMapIndexedDataset): The MMapIndexedDataset around which to build the
        MegatronDataset

        dataset_path (str): The real path on disk to the dataset, for bookkeeping

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (int): The number of samples to draw from the indexed dataset

        index_split (Split): The indexed_indices Split

        config (GPTDatasetConfig): The config
    """

    def __init__(
        self,
        indexed_dataset: MMapIndexedDataset,
        dataset_path: str,
        indexed_indices: numpy.ndarray,
        num_samples: int,
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )
        self.args = get_args()
        self.tokenizer = get_tokenizer()
        self.np_rng = numpy.random.RandomState(seed=self.config.random_seed) # rng state for FIM

        self.use_fim = self.args.fim_rate!=0
        if self.use_fim:
            self.fim_rate = self.args.fim_rate
            self.fim_spm_rate = self.args.fim_spm_rate
            self.fragment_fim_rate = self.args.fragment_fim_rate
            self.fim_split_sample = self.tokenizer.vocab[self.args.fim_split_sample] if self.args.fim_split_sample is not None else None

            try:
                self.suffix_tok_id, self.prefix_tok_id, self.middle_tok_id, self.pad_tok_id = (self.tokenizer.special_tokens[tok] for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD])
            except KeyError:
                self.suffix_tok_id, self.prefix_tok_id, self.middle_tok_id, self.pad_tok_id = (self.tokenizer.vocab[tok] for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD])

        self.vocab_size = config.vocab_size

    def _finalize(self) -> None:
        """Abstract method implementation
        
        Load or build/cache the document, sample, and shuffle indices
        """
        (
            self.document_index,
            self.sample_index,
            self.shuffle_index,
        ) = self._build_document_sample_shuffle_indices()

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: MMapIndexedDataset) -> int:
        """Abstract method implementation

        For GPT, the underlying MMapIndexedDataset should be split by sequence, as opposed to, say,
        BERT, which should be split by document

        Args:
            low_level_dataset (MMapIndexedDataset): The underlying MMapIndexedDataset

        Returns:
            int: The number of unique elements in the underlying MMapIndexedDataset
        """
        return low_level_dataset.sequence_lengths.shape[0]

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> MMapIndexedDataset:
        """Abstract method implementation

        Args:
            dataset_path (str): The real path prefix to the MMapIndexedDataset .bin and .idx files

            config (BlendedMegatronDatasetConfig): The dataset config

        Returns:
            MMapIndexedDataset: The underlying MMapIndexedDataset
        """
        return MMapIndexedDataset(dataset_path, False)

    def __len__(self) -> int:
        """Abstract method implementation

        Returns:
            int: The length of the dataset
        """
        return self.sample_index.shape[0] - 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The text ids wrapped in a dictionary
        """
        text, _ = self._query_document_sample_shuffle_indices(idx)

        text = torch.from_numpy(text).long()
        labels = text[1:].contiguous()
        tokens = text[:-1].contiguous()

        assert not torch.any(
            tokens >= self.vocab_size
        ), "An input token is out of bounds of the tokenizer vocabulary"

        attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
            tokens,
            self.config.tokenizer.eod,
            self.config.reset_position_ids,
            self.config.reset_attention_mask,
            self.config.eod_mask_loss,
        )

        return {
            "tokens": tokens,
            "labels": labels,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }

    def _query_document_sample_shuffle_indices(
        self, idx: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get the text (token ids) and document ids for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The text ids and document ids
        """
        # Do the shuffle mapping
        idx = self.shuffle_index[idx]

        # Get the beginning and end documents and offsets
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        document_ids = []
        sample_parts = []

        # Sample spans a single document
        if doc_index_beg == doc_index_end:
            # Add the document id
            document_ids.append(self.document_index[doc_index_beg])

            # Add the entire sample
            sample_parts.append(
                self.dataset.get(
                    self.document_index[doc_index_beg],
                    offset=doc_index_beg_offset,
                    length=doc_index_end_offset - doc_index_beg_offset + 1,
                )
            )

        # Sample spans multiple documents
        else:
            for i in range(doc_index_beg, doc_index_end + 1):
                # Add the document id
                document_ids.append(self.document_index[i])

                # Add the sample part
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = None if i < doc_index_end else doc_index_end_offset + 1
                sample_parts.append(
                    self.dataset.get(self.document_index[i], offset=offset, length=length)
                )

        sample=numpy.concatenate(sample_parts)

        # Code from: https://github.com/EleutherAI/gpt-neox/blob/FIM-clean/megatron/data/gpt2_dataset.py#L109
        # TODO(Hailey): can merge the code below this line with code above this line.
        # TODO(Hailey), cont: above already iterates through loop, so just add the permuting in there?
        sample = numpy.array(sample, dtype=numpy.int64)
        sample_len = sample.shape[0]
        # # print(sample, sample.shape)
        # # do FIM here, if enabled
        # TODO: Do we handle the following point from FIM paper?
        # To transform data in the character space for context-level FIM, the tokenized documents have to be decoded back into strings before FIM augmentation. Depending on the vocabulary, some care has to be given to ensure decoding does not introduce any spurious characters into training. For example, utf-8 characters are encoded as multiple tokens with a BPE vocabulary; they can result in fragments from chunking and fail to decode. To prevent unforeseen errors midway through training, we encourage checking for these fragments at the beginning or end of a context and removing them.
        eod = self.tokenizer.eod
        segment_breaks = numpy.argwhere(sample == eod) # split sample by document

        if not self.use_fim:
            return (
                numpy.array(sample, dtype=numpy.int64),
                numpy.array(document_ids, dtype=numpy.int64),
            )

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
            fragment_breaks = numpy.argwhere(sequence == self.fim_split_sample)
            if fragment_breaks.shape == (0, 1):
                # no split token in this sample
                return fim_permute_sequence(sequence, self.fim_rate)
            if not self.np_rng.binomial(1, self.fim_rate):
                # don't do FIM preproc
                return sequence
            # Do FIM on each fragment
            curr_start_position = 0
            new_samples = []
            for loc in numpy.nditer(fragment_breaks):
                if loc - curr_start_position > 0:
                    permuted = fim_permute_sequence(sequence[curr_start_position:loc], self.fragment_fim_rate)
                    new_samples += [permuted, [self.fim_split_sample]]
                curr_start_position = loc + 1  # Jump over the split token
            # Permute the segment after the last split token
            permuted = fim_permute_sequence(sequence[curr_start_position:], self.fragment_fim_rate)
            new_samples.append(permuted)
            return numpy.concatenate(new_samples)

        if segment_breaks.shape != (0, 1):  # then there is an EOD token in this example
            curr_start_position = 0
            new_samples = []
            for loc in numpy.nditer(segment_breaks):
                # Only permute non-empty segments.
                if loc - curr_start_position > 0:
                    # permute {prefix, suffix, middle} or {suffix, prefix, middle}
                    permuted = fim_split_and_permute_sequence(sample[curr_start_position:loc])
                    new_samples += [permuted, [eod]]

                curr_start_position = loc + 1  # jump over the EOD token
            # Permute the segment after the last EOD
            permuted = fim_split_and_permute_sequence(sample[curr_start_position:])
            new_samples.append(permuted)

            sample = numpy.concatenate(new_samples)
        else:
            sample = fim_split_and_permute_sequence(sample)

        # Truncate or pad sequence to max-length
        diff = sample.shape[0] - sample_len
        if diff > 0: # too long
            sample = sample[:sample_len]
        elif diff < 0: # too short
            sample = numpy.concatenate([sample, numpy.full((-1 * diff), self.pad_tok_id)])

        assert sample.shape[0] == sample_len
        # end FIM-specific code

        return (
            numpy.array(sample, dtype=numpy.int64),
            numpy.array(document_ids, dtype=numpy.int64),
        )

    def _build_document_sample_shuffle_indices(
        self,
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Build the document index, the sample index, and the shuffle index
        
        The document index:
            -- 1-D
            -- An ordered array of document ids

        The sample index:
            -- 2-D
            -- The document indices and offsets which mark the start of every sample

        The shuffle index:
            -- 1-D
            -- A random permutation of index range of the sample index

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The document index, the sample index, and the
            shuffle index

        TODO: Explain the 80% threshold
        """
        path_to_cache = self.config.path_to_cache
        if path_to_cache is None:
            path_to_cache = os.path.join(
                self.dataset.path_prefix, "cache", f"{type(self).__name__}_indices"
            )

        get_path_to = lambda suffix: os.path.join(
            path_to_cache, f"{self.unique_description_hash}-{type(self).__name__}-{suffix}"
        )
        path_to_description = get_path_to("description.txt")
        path_to_document_index = get_path_to("document_index.npy")
        path_to_sample_index = get_path_to("sample_index.npy")
        path_to_shuffle_index = get_path_to("shuffle_index.npy")
        cache_hit = all(
            map(
                os.path.isfile,
                [
                    path_to_description,
                    path_to_document_index,
                    path_to_sample_index,
                    path_to_shuffle_index,
                ],
            )
        )

        num_tokens_per_epoch = self._get_num_tokens_per_epoch()
        num_epochs = self._get_num_epochs(num_tokens_per_epoch)

        if not cache_hit and torch.distributed.get_rank() == 0:
            log_single_rank(
                logger,
                logging.INFO,
                f"Build and save the {type(self).__name__} {self.index_split.name} indices",
            )

            sequence_length = self.config.sequence_length

            if num_epochs == 1:
                separate_final_epoch = False
            else:
                # Get the number of samples for the last epoch
                num_samples_sans_final_epoch = (
                    (num_epochs - 1) * num_tokens_per_epoch - 1
                ) // sequence_length
                num_samples_from_final_epoch = self.num_samples - num_samples_sans_final_epoch
                num_samples_per_epoch = (num_tokens_per_epoch - 1) // sequence_length

                # num_samples_from_final_epoch should be non-negative
                assert num_samples_from_final_epoch >= 0

                # num_samples_from_final_epoch should not exceed max value
                assert num_samples_from_final_epoch <= num_samples_per_epoch + 1

                # Separate the final epoch if it falls below the threshold
                threshold = 0.80
                separate_final_epoch = num_samples_from_final_epoch < int(
                    threshold * num_samples_per_epoch
                )

                log_single_rank(
                    logger,
                    logging.DEBUG,
                    f"> num_samples_from_final_epoch: {num_samples_from_final_epoch}",
                )
                log_single_rank(logger, logging.DEBUG, f"> threshold: {threshold}")
                log_single_rank(
                    logger, logging.DEBUG, f"> num_samples_per_epoch: {num_samples_per_epoch}"
                )

            log_single_rank(
                logger, logging.DEBUG, f"> separate_final_epoch: {separate_final_epoch}"
            )

            numpy_random_state = numpy.random.RandomState(self.config.random_seed)

            os.makedirs(path_to_cache, exist_ok=True)

            # Write the description
            with open(path_to_description, "wt") as writer:
                writer.write(self.unique_description)

            # Build the document index
            log_single_rank(
                logger,
                logging.INFO,
                f"\tBuild and save the document index to {os.path.basename(path_to_document_index)}",
            )
            t_beg = time.time()
            document_index = _build_document_index(
                self.indices, num_epochs, numpy_random_state, separate_final_epoch
            )
            numpy.save(path_to_document_index, document_index, allow_pickle=True)
            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            # Build the sample index
            log_single_rank(
                logger,
                logging.INFO,
                f"\tBuild and save the sample index to {os.path.basename(path_to_sample_index)}",
            )
            t_beg = time.time()
            from megatron.core.datasets import helpers

            assert document_index.dtype == numpy.int32
            assert self.dataset.sequence_lengths.dtype == numpy.int32
            sample_index = helpers.build_sample_idx(
                self.dataset.sequence_lengths,
                document_index,
                sequence_length,
                num_epochs,
                num_tokens_per_epoch,
            )
            numpy.save(path_to_sample_index, sample_index, allow_pickle=True)
            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            # Build the shuffle index
            log_single_rank(
                logger,
                logging.INFO,
                f"\tBuild and save the shuffle index to {os.path.basename(path_to_shuffle_index)}",
            )
            t_beg = time.time()
            if separate_final_epoch:
                shuffle_index = _build_shuffle_index(
                    num_samples_sans_final_epoch, sample_index.shape[0] - 1, numpy_random_state
                )
            else:
                shuffle_index = _build_shuffle_index(
                    sample_index.shape[0] - 1, sample_index.shape[0] - 1, numpy_random_state
                )
            numpy.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)
            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            log_single_rank(
                logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
            )
            log_single_rank(logger, logging.INFO, f"> total number of epochs: {num_epochs}")

            return document_index, sample_index, shuffle_index

        log_single_rank(
            logger, logging.INFO, f"Load the {type(self).__name__} {self.index_split.name} indices"
        )

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the document index from {os.path.basename(path_to_document_index)}",
        )
        t_beg = time.time()
        document_index = numpy.load(path_to_document_index, allow_pickle=True, mmap_mode='r')
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the sample index from {os.path.basename(path_to_sample_index)}",
        )
        t_beg = time.time()
        sample_index = numpy.load(path_to_sample_index, allow_pickle=True, mmap_mode='r')
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the shuffle index from {os.path.basename(path_to_shuffle_index)}",
        )
        t_beg = time.time()
        shuffle_index = numpy.load(path_to_shuffle_index, allow_pickle=True, mmap_mode='r')
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
        )
        log_single_rank(logger, logging.INFO, f"> total number of epochs: {num_epochs}")

        return document_index, sample_index, shuffle_index

    def _get_num_tokens_per_epoch(self) -> int:
        """Calculate the number of tokens in a single epoch

        Returns:
            int: The number of tokens in a single epoch
        """
        return int(numpy.sum(self.dataset.sequence_lengths[self.indices]))

    def _get_num_epochs(self, num_tokens_per_epoch: int) -> int:
        """Calculate the number of epochs

        Args:
            num_tokens_per_epoch (int): The number of tokens in a single epoch

        Returns:
            int: The number of epochs
        """
        num_epochs = 0
        num_tokens = 0
        num_tokens_requested = (self.num_samples * self.config.sequence_length) + 1
        while True:
            num_epochs += 1
            num_tokens += num_tokens_per_epoch
            if num_tokens >= num_tokens_requested:
                return num_epochs


def _build_document_index(
    documents: numpy.ndarray,
    num_epochs: int,
    numpy_random_state: numpy.random.RandomState,
    separate_final_epoch: bool,
) -> numpy.ndarray:
    """Build an array with length = num epochs * num documents

    Args:
        documents (numpy.ndarray): the subset of exposed document indices

        num_epochs (int): The number of epochs

        numpy_random_state (numpy.random.RandomState): The NumPy random state

        separate_final_epoch (bool): Whether to exclude the last epoch from the global shuffle

    Returns:
        numpy.ndarray: The document index

    TODO: Explain separate_final_epoch
    """
    if not separate_final_epoch or num_epochs == 1:
        document_index = numpy.mgrid[0:num_epochs, 0 : len(documents)][1]
        document_index[:] = documents
        document_index = document_index.reshape(-1)
        document_index = document_index.astype(numpy.int32)
        numpy_random_state.shuffle(document_index)
        return document_index

    doc_idx_first = _build_document_index(documents, num_epochs - 1, numpy_random_state, False)
    doc_idx_last = _build_document_index(documents, 1, numpy_random_state, False)
    return numpy.concatenate((doc_idx_first, doc_idx_last))


def _build_shuffle_index(
    num_samples: int, total_size: int, numpy_random_state: numpy.random.RandomState
) -> numpy.ndarray:
    """Build the range [0, size) and shuffle

    Args:
        num_samples (int): The size of the first shuffle range [0, num_samples)

        total_size (int): The size of the entire index. If larger than 'num_samples', it defines

        the second shuffle range [num_samples, total_size)

        numpy_random_state (numpy.random.RandomState): The NumPy random state

    Returns:
        numpy.ndarray: The shuffle index

    TODO: Explain [0, num_samples) [num_samples, total_size) split
    """
    dtype_ = numpy.uint32
    if total_size >= (numpy.iinfo(numpy.uint32).max - 1):
        dtype_ = numpy.int64

    shuffle_idx_first = numpy.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = numpy.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_last)

    return numpy.concatenate((shuffle_idx_first, shuffle_idx_last))


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

        prefix = numpy.array([*tokenizer.tokenize(prefix)], dtype=numpy.int64)
        middle = numpy.array([*tokenizer.tokenize(middle)], dtype=numpy.int64)
        suffix = numpy.array([*tokenizer.tokenize(suffix)], dtype=numpy.int64)

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
                suffix = numpy.concatenate([suffix, numpy.full((-1 * diff), pad_tok_id)])

        if np_rng.binomial(1, fim_spm_rate):
            # SPM (variant 2 from FIM paper)
            new_sample = numpy.concatenate([
                [prefix_tok_id, suffix_tok_id], suffix,
                [middle_tok_id], prefix, middle
            ])
        else:
            # PSM
            new_sample = numpy.concatenate([
                [prefix_tok_id], prefix,
                [suffix_tok_id], suffix,
                [middle_tok_id], middle
            ])

    else:
        # don't do FIM preproc
        new_sample = sample

    return new_sample


def _get_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
):
    """Build masks and position id for left to right model.

    Args:
        data (torch.Tensor): The data tenor that holds the tokens from the dataset

        eod_token (int): ID of the token to that is considered the EOD

        reset_position_ids (bool): Switch to reset the document position ID's

        reset_attention_mask (bool): Switch to reset the attention mask

        eod_mask_loss (bool): Switch to enable the EOD mask loss

    Returns:
        torch.Tensor : Attention mask needed to be used for Attention

        torch.Tensor : The mask used for loss value during training

        torch.Tensor : The position ID's of the token
    """
    seq_length = data.numel()

    attention_mask = torch.tril(torch.ones((seq_length, seq_length), device=data.device)).unsqueeze(
        0
    )

    # Loss mask.
    loss_mask = torch.ones(seq_length, dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Find indices where EOD token is.
        eod_index = position_ids[data == eod_token]
        # Detach indices from positions if going to modify positions.
        if reset_position_ids:
            eod_index = eod_index.clone()

        # Loop through EOD indices:
        prev_index = 0
        for j in range(eod_index.numel()):
            i = eod_index[j]
            # Mask attention loss.
            if reset_attention_mask:
                attention_mask[0, (i + 1) :, : (i + 1)] = 0
            # Reset positions.
            if reset_position_ids:
                position_ids[(i + 1) :] -= i + 1 - prev_index
                prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids

