"""Multitask Finetuning"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer

from megatron import get_args, get_tokenizer, print_rank_0, mpu
from megatron.data.decoder_packed_mtf_dataset import build_train_valid_test_datasets, build_dataset_group
from megatron.model.enums import PositionEmbeddingType, AttnMaskType
from megatron.model import GPTModel, ModelType
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids, get_packed_attention_mask
from megatron.utils import average_losses_across_data_parallel_group

### Debugging Helpers ###

def visualize_model_inputs(tokens, attention_mask, labels, loss_mask, position_ids):
    tok = get_tokenizer()
    print("TOKENS:", ",".join([tok.detokenize(tokens[0, i]) for i in range(100)]))
    print("ATTN:", attention_mask[0, :, :100, :100])
    print("LABS:", labels[0, :100])
    print("LOSSMSK:", loss_mask[:100])
    print("POSIDS:", position_ids[0, :100])

def save_model_inputs(tokens, attention_mask, labels, loss_mask, position_ids, segment_ids, iteration):
    """Save as tensors for debugging"""
    torch.save(tokens, f"tokens_{iteration}.pt")
    torch.save(attention_mask, f"attention_mask_{iteration}.pt")
    torch.save(labels, f"labels_{iteration}.pt")
    torch.save(loss_mask, f"loss_mask_{iteration}.pt")
    torch.save(position_ids, f"position_ids_{iteration}.pt")
    torch.save(segment_ids, f"segment_ids_{iteration}.pt")
    # exit() # Optionaly exit right after

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
        attn_mask_type=AttnMaskType.custom,
    )
    return model

def fast_normalize(loss_mask: torch.Tensor):
    """
    Turn loss_mask from [0,0,0,1,1,0,0,1,0,0,1,1,1] > [0,0,0,0.5,0.5,0,0,1,0,0,0.3,0.3,0.3]
    """
    _, inverse_indices, counts = torch.unique_consecutive(loss_mask, return_inverse=True, return_counts=True)
    counts = torch.gather(dim=0, index=inverse_indices, input=counts)
    return loss_mask / counts

def get_batch(data):
    """
    Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator` & in packed fashion
    
    data:
    decoder_tokens = [[6, 7, 8, 3, 4, 5, 0]]
    decoder_segment_ids = [[1, 1, 1, 2, 2, 2, 0]]
    decoder_is_inputs = [[1, 1, 0, 1, 1, 0, 0]]
    """
    args = get_args()
    tokenizer = get_tokenizer()

    # Broadcast data.
    if data is not None:
        data = next(data)
    else:
        data = None

    data_b = mpu.broadcast_data(["decoder_token_ids", "decoder_segment_ids"], data, torch.int64)
    data_c = mpu.broadcast_data(["decoder_is_inputs"], data, torch.bool)

    # Unpack.
    tokens_ = data_b["decoder_token_ids"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    
    segment_ids = data_b["decoder_segment_ids"].long()[:, :-1]
    decoder_is_inputs = data_c["decoder_is_inputs"][:, :-1]

    # Get the masks and position ids.
    causal_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )
    # Only compute loss over causal target tokens, i.e. ignore input_tokens & padding
    loss_on_non_pad_only = (labels != tokenizer.pad)
    if args.loss_on_targets_only:
        loss_on_targets_only = ~data_c["decoder_is_inputs"][:, 1:]
        loss_mask *= loss_on_targets_only * loss_on_non_pad_only
    else:
        loss_mask *= loss_on_non_pad_only

    attention_mask = get_packed_attention_mask(
        is_causal=True, # Always make it causal for now; Could ablate this
        causal_mask=~(causal_mask.bool()), # Turn back into tril being ones
        decoder_is_inputs=decoder_is_inputs.bool(),
        segment_ids=segment_ids.long(),
    )

    if args.norm_target_loss:
        loss_mask = loss_mask.view(-1)
        loss_mask = fast_normalize(loss_mask)

    # For Alibi / Rotary, positions ids are not used so it does not matter
    if args.position_embedding_type == PositionEmbeddingType.absolute:
        # Create position ids from segment_ids
        # segment_ids = torch.tensor([[1, 1, 1, 2, 2, 2, 2, 0]]) (Shape: (batch_size, seq_len))
        # position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 0]]) (Shape: (batch_size, seq_len))
        # I.e. they should restart for each new segment from 0
        position_ids = []
        for b in segment_ids:
            counts = torch.unique_consecutive(b, return_counts=True, dim=-1)[1]
            p = torch.cat([torch.arange(c) for c in counts])
            position_ids.append(p)
        position_ids = torch.stack(position_ids).to(tokens.device)    

    #visualize_model_inputs(tokens, attention_mask, labels, loss_mask, position_ids)
    #if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
    #    save_model_inputs(tokens, attention_mask, labels, loss_mask, position_ids, segment_ids, args.curr_iteration)

    return tokens, labels, loss_mask, attention_mask, position_ids
    #return (tokens, position_ids, attention_mask), (labels, loss_mask)

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}

def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_ds, valid_ds, test_ds = None, None, None

    tokenizer = get_tokenizer()

    print_rank_0("> building train, validation, and test datasets for MTF ...")
    # Option 1 of data loading using --data-path
    if args.data_path:
        # TODO: Not yet compatible with dataset weights (Will break at prefixes, weights = analyze_data_prefix(args.data_path))
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            seq_length=args.seq_length + 1,
            pad_token=tokenizer.pad,
            eos_token=tokenizer.eod,
            train_valid_test_num_samples=train_val_test_num_samples,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup)
        )
    # Option 2 of data loading using --(train|valid|test)-weighted-split-paths
    elif args.train_weighted_split_paths:
        assigned_train_valid_test = []
        if args.train_weighted_split_paths is not None:
            train_ds = []
            assigned_train_valid_test.append("train")
        if args.valid_weighted_split_paths is not None:
            valid_ds = []
            assigned_train_valid_test.append("valid")
        if args.test_weighted_split_paths is not None:
            test_ds = []
            assigned_train_valid_test.append("test")

        for s in assigned_train_valid_test:
            data_groups = zip(eval(f"args.{s}_weighted_split_paths"),
                              eval(f"args.{s}_weighted_split_weights"),
                              eval(f"args.{s}_weighted_split_splits"),
                              eval(f"args.{s}_weighted_split_names"))
            for paths, weights, splits, name in data_groups:
                d = build_dataset_group(
                    dataset_group_name=name,
                    paths=paths,
                    weights=weights,
                    splits=splits,
                    data_impl=args.data_impl,
                    train_valid_test_num_samples=train_val_test_num_samples,
                    seq_length=args.seq_length + 1,
                    pad_token=tokenizer.pad,
                    eos_token=tokenizer.eod,
                    seed=args.seed,
                    skip_warmup=(not args.mmap_warmup),
                    train_valid_test=s
                )
                eval(f"{s}_ds").append(d)
    else:
        raise NotImplementedError("No dataloading argument passed")

    print_rank_0("> finished creating MTF datasets ...")
    return train_ds, valid_ds, test_ds

if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step, args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
