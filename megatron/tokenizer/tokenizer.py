# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""

from abc import ABC
from abc import abstractmethod

from transformers import PreTrainedTokenizerFast
from .bert_tokenization import FullTokenizer as FullBertTokenizer
from .gpt2_tokenization import GPT2Tokenizer


FIM_PREFIX = "<fim-prefix>"
FIM_MIDDLE = "<fim-middle>"
FIM_SUFFIX = "<fim-suffix>"
FIM_PAD = "<fim-pad>"
EOD = "<|endoftext|>"


def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.tokenizer_type),
              flush=True)

    if args.is_ul2:
        ul2_denoiser_tokens = [
            args.ul2_r_denoiser_token,
            args.ul2_s_denoiser_token,
            args.ul2_x_denoiser_token,
        ]
    else:
        ul2_denoiser_tokens = []

    # Select and instantiate the tokenizer.
    if args.tokenizer_type in ['BertWordPieceLowerCase', 'BertWordPieceCase', 'GPT2BPETokenizer', 'GPT2BPETokenizerWithFIM']:
        assert args.vocab_file is not None
    elif args.tokenizer_type == "SentencePieceTokenizer":
        assert args.tokenizer_model is not None
    else:
        assert args.tokenizer_file is not None
    if args.tokenizer_type == 'BertWordPieceLowerCase':
        tokenizer = _BertWordPieceTokenizer(
            vocab_file=args.vocab_file,
            lower_case=True,
            vocab_extra_ids=args.vocab_extra_ids,
            ul2_denoiser_tokens=ul2_denoiser_tokens,
        )
    elif args.tokenizer_type == 'BertWordPieceCase':
        tokenizer = _BertWordPieceTokenizer(
            vocab_file=args.vocab_file,
            lower_case=False,
            vocab_extra_ids=args.vocab_extra_ids,
            ul2_denoiser_tokens=ul2_denoiser_tokens,
        )
    elif args.tokenizer_type == 'GPT2BPETokenizer':
        assert args.merge_file is not None
        tokenizer = _GPT2BPETokenizer(
            args.vocab_file,
            args.merge_file,
            ul2_denoiser_tokens=ul2_denoiser_tokens,
        )
    # TODO: Should probably add a check that we are doing either FIM or UL2, not both.
    elif args.tokenizer_type == 'GPT2BPETokenizerWithFIM':
        assert args.merge_file is not None
        assert args.vocab_extra_ids == 0, "Are you sure you want to use the FIM tokenizer? it seems that vocab-extra-ids was set >0"
        tokenizer = _GPT2BPETokenizer(args.vocab_file, args.merge_file, special_tokens=[FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD])
    elif args.tokenizer_type == "TokenizerFromFile":
        assert args.tokenizer_file is not None
        tokenizer = _HFTokenizer(
            args.tokenizer_file,
            special_tokens=[EOD],
            ul2_denoiser_tokens=ul2_denoiser_tokens,
            vocab_extra_ids=args.vocab_extra_ids
        )
    elif args.tokenizer_type == "TokenizerFromFileWithFIM":
        assert args.tokenizer_file is not None
        assert args.vocab_extra_ids == 0, "Are you sure you want to use the FIM tokenizer? it seems that vocab-extra-ids was set >0"
        tokenizer = _HFTokenizer(args.tokenizer_file, special_tokens=[EOD, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD])
    elif args.tokenizer_type == 'SentencePieceTokenizer':
        assert args.tokenizer_model is not None
        tokenizer = _SentencePieceTokenizer(
            args.tokenizer_model,
            vocab_extra_ids=args.vocab_extra_ids,
            ul2_denoiser_tokens=ul2_denoiser_tokens,
        )
    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(args.tokenizer_type))

    # Add vocab size.
    # TODO: For most tokenizers, vocab_size does not take special_tokens into account. 
    # Might cause an issue if vocab_size + len(special_tokens) exceeds padded_vocab_size?
    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size,
                                                      args)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * \
        args.tensor_model_parallel_size
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(' > padded vocab (size: {}) with {} dummy tokens '
              '(new size: {})'.format(
                  orig_vocab_size, after - orig_vocab_size, after), flush=True)
    return after


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError('detokenizer is not implemented for {} '
                                  'tokenizer'.format(self.name))

    @property
    def cls(self):
        raise NotImplementedError('CLS is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def sep(self):
        raise NotImplementedError('SEP is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def pad(self):
        raise NotImplementedError('PAD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def eod(self):
        raise NotImplementedError('EOD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def mask(self):
        raise NotImplementedError('MASK is not provided for {} '
                                  'tokenizer'.format(self.name))


class _BertWordPieceTokenizer(AbstractTokenizer):
    """Original BERT wordpiece tokenizer."""

    def __init__(
            self,
            vocab_file,
            lower_case=True,
            vocab_extra_ids=0,
            ul2_denoiser_tokens=None,
    ):
        if lower_case:
            name = 'BERT Lower Case'
        else:
            name = 'BERT Upper Case'
        super().__init__(name)
        self.tokenizer = FullBertTokenizer(vocab_file, do_lower_case=lower_case)
        self.cls_id = self.tokenizer.vocab['[CLS]']
        self.sep_id = self.tokenizer.vocab['[SEP]']
        self.pad_id = self.tokenizer.vocab['[PAD]']
        self.mask_id = self.tokenizer.vocab['[MASK]']
        self._additional_special_tokens = []

        # (dsachan) Add BOS and EOS tokens
        SPECIAL_TOKENS = {'eos_token': '[EOS]',
                          'bos_token': '[BOS]'}
        self._bos_token = '[BOS]'
        self.add_token(self._bos_token)
        self._bos_token_id = self.vocab.get(self._bos_token)

        self._eos_token = '[EOS]'
        self.add_token(self._eos_token)
        self._eos_token_id = self.vocab.get(self._eos_token)

        # (dsachan) Add additional special tokens
        # These can be used as sentinel tokens in T5 model inputs
        additional_special_tokens = []
        additional_special_tokens.extend(
            ["<extra_id_{}>".format(i) for i in range(vocab_extra_ids)])

        if ul2_denoiser_tokens is None:
            ul2_denoiser_tokens = []
        self._ul2_tokens = ul2_denoiser_tokens
        for value in self._ul2_tokens:
            self.add_token(value)

        self.add_additional_special_tokens(additional_special_tokens)

    def add_token(self, token):
        if token not in self.vocab:
            self.inv_vocab[self.vocab_size] = token
            # self.vocab_size comes from len(vocab)
            # and it will increase as we add elements
            self.vocab[token] = self.vocab_size

    def add_additional_special_tokens(self, tokens_list):
        setattr(self, "additional_special_tokens", tokens_list)
        for value in tokens_list:
            self.add_token(value)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size()

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        return self.tokenizer.inv_vocab

    def tokenize(self, text):
        text_tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(text_tokens)

    def decode(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return self.tokenizer.convert_tokens_to_string(tokens)

    def decode_token_ids(self, token_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        exclude_list = ['[PAD]', '[CLS]']
        non_pads = [t for t in tokens if t not in exclude_list]

        result = ""
        for s in non_pads:
            if s.startswith("##"):
                result += s[2:]
            else:
                result += " " + s

        return result

    @property
    def cls(self):
        return self.cls_id

    @property
    def sep(self):
        return self.sep_id

    @property
    def pad(self):
        return self.pad_id

    @property
    def mask(self):
        return self.mask_id

    @property
    def bos_token(self):
        """ Beginning of sentence token id """
        return self._bos_token

    @property
    def eos_token(self):
        """ End of sentence token id """
        return self._eos_token

    @property
    def additional_special_tokens(self):
        """ All the additional special tokens you may want to use (list of strings)."""
        return self._additional_special_tokens

    @property
    def bos_token_id(self):
        """ Id of the beginning of sentence token in the vocabulary."""
        return self._bos_token_id

    @property
    def eos_token_id(self):
        """ Id of the end of sentence token in the vocabulary."""
        return self._eos_token_id

    @property
    def additional_special_tokens_ids(self):
        """ Ids of all the additional special tokens in the vocabulary (list of integers)."""
        return [self.vocab.get(token) for token in self._additional_special_tokens]

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value

    @property
    def ul2_token_ids(self):
        return [self.vocab[k] for k in self._ul2_tokens]


class _GPT2BPETokenizer(AbstractTokenizer):
    """Original GPT2 BPE tokenizer."""

    def __init__(self, vocab_file, merge_file, ul2_denoiser_tokens=None, special_tokens=None):
        name = 'GPT2 BPE'
        super().__init__(name)

        assert ul2_denoiser_tokens is None or special_tokens is None, "Cant use both ul2_denoiser_tokens and special_tokens"
        # TODO: refactor the special_tokens mess
        special_tokens = special_tokens if special_tokens is not None else []

        if ul2_denoiser_tokens is None:
            ul2_denoiser_tokens = []
        self._ul2_tokens = ul2_denoiser_tokens

        # Warning! `additional_special_token_ids` will also return the UL2
        # tokens here.
        special_tokens += self._ul2_tokens
        if self._ul2_tokens:
            special_tokens.append('<SEP>')

        self.tokenizer = GPT2Tokenizer(vocab_file, merge_file, errors='replace',
                                       special_tokens=special_tokens,
                                       max_len=None)
        if self._ul2_tokens:
            self.sep_id = self.tokenizer.encoder['<SEP>']
        else:
            self.sep_id = None
        self.eod_id = self.tokenizer.encoder['<|endoftext|>']
        self.special_tokens = self.tokenizer.special_tokens

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def sep(self):
        if self.sep_id is None:
            raise AttributeError(
                'GPT tokenizer does not have a SEP token by default; '
                'please add it to the `special_tokens`')
        return self.sep_id

    @property
    def eod(self):
        return self.eod_id

    @property
    def additional_special_tokens_ids(self):
        # Warning! This will also return the UL2 tokens.
        return [self.vocab[k] for k in self.tokenizer.special_tokens]

    # TODO: it seems this is not used and could be removed?
    @property
    def ul2_tokens_ids(self):
        return [self.vocab[k] for k in self._ul2_tokens]

class _HFTokenizer(AbstractTokenizer):
    """HF Tokenizer."""

    CLS = "<CLS>"
    SEP = "<SEP>"
    MASK = "<MASK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    PAD = "<PAD>"

    def __init__(self, tokenizer_file, ul2_denoiser_tokens=None, special_tokens=None, vocab_extra_ids=None):
        name = 'HF Tokenizer'
        super().__init__(name)

        special_tokens = special_tokens if special_tokens is not None else []
        assert EOD in special_tokens
        # For backward compatibility, other special tokens should come after EOD
        # Append at the end of the special tokens:
        special_tokens += [
            _HFTokenizer.CLS, _HFTokenizer.SEP, _HFTokenizer.MASK, _HFTokenizer.BOS, _HFTokenizer.EOS, _HFTokenizer.PAD
        ]
        # Add UL2 tokens
        special_tokens += ul2_denoiser_tokens if ul2_denoiser_tokens is not None else []
        # add extra-token-ids
        if vocab_extra_ids is not None:
            self._t5_tokens = ["<extra_id_{}>".format(i) for i in range(vocab_extra_ids)]
            special_tokens += self._t5_tokens
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, errors='replace', max_len=None)
        for tok in special_tokens:
            assert tok not in self.tokenizer.vocab, f"Special token {tok} was already in vocab"
        
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self._eod_id = self.tokenizer.vocab[EOD]
        # Token->id mapping for additional special-tokens
        self.special_tokens = {
            tok: self.tokenizer.vocab[tok] for tok in special_tokens
        }
        self._inv_vocab = {v: k for k, v in self.tokenizer.vocab.items()}

        self._cls_id = self.tokenizer.vocab[_HFTokenizer.CLS]
        self._sep_id = self.tokenizer.vocab[_HFTokenizer.SEP]
        self._mask_id = self.tokenizer.vocab[_HFTokenizer.MASK]
        self._bos_id = self.tokenizer.vocab[_HFTokenizer.BOS]
        self._eos_id = self.tokenizer.vocab[_HFTokenizer.EOS]
        self._pad_id = self.tokenizer.vocab[_HFTokenizer.PAD]

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def vocab(self):
        return self.tokenizer.vocab
    
    @property
    def inv_vocab(self):
        return self._inv_vocab
    
    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)
    
    @property
    def cls(self):
        return self._cls_id

    @property
    def sep(self):
        return self._sep_id

    @property
    def pad(self):
        return self._pad_id

    @property
    def bos_token_id(self):
        return self._bos_id

    @property
    def bos(self):
        return self._bos_id

    @property
    def eod(self):
        return self._eod_id

    @property
    def eos_token_id(self):
        return self._eos_id

    @property
    def eos(self):
        return self._eos_id

    @property
    def mask(self):
        return self._mask_id
    
    @property
    def additional_special_tokens_ids(self):
        """T5 extra token_ids"""
        return [self.vocab[k] for k in self._t5_tokens]


class _SentencePieceTokenizer(AbstractTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(
            self, model_file, vocab_extra_ids=0, ul2_denoiser_tokens=None):
        name = 'SentencePieceTokenizer'
        super().__init__(name)

        import sentencepiece
        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)

        if ul2_denoiser_tokens is None:
            ul2_denoiser_tokens = []
        self._initialize(vocab_extra_ids, ul2_denoiser_tokens)

    def _initialize(self, vocab_extra_ids, ul2_denoiser_tokens):
        self._vocab = {}
        self._inv_vocab = {}

        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []
        self._ul2_tokens = []

        for i in range(len(self._tokenizer)):
            t = self._tokenizer.id_to_piece(i)
            self._inv_vocab[i] = t
            self._vocab[t] = i

        def _add_special_token(t):
            if t not in self._vocab:
                next_id = len(self._vocab)
                self._vocab[t] = next_id
                self._inv_vocab[next_id] = t
            self._special_tokens[t] = self._vocab[t]
            self._inv_special_tokens[self._vocab[t]] = t

        _add_special_token('<CLS>')
        self._cls_id = self._vocab['<CLS>']
        _add_special_token('<SEP>')
        self._sep_id = self._vocab['<SEP>']
        _add_special_token('<EOD>')
        self._eod_id = self._vocab['<EOD>']
        _add_special_token('<MASK>')
        self._mask_id = self._vocab['<MASK>']

        pad_id = self._tokenizer.pad_id()
        try:
            pad_token = self._tokenizer.id_to_piece(pad_id)
        except IndexError:
            pad_token = '<PAD>'
        _add_special_token(pad_token)
        self._pad_id = self._vocab[pad_token]

        bos_id = self._tokenizer.bos_id()
        try:
            bos_token = self._tokenizer.id_to_piece(bos_id)
        except IndexError:
            bos_token = '<BOS>'
        _add_special_token(bos_token)
        self._bos_id = self._vocab[bos_token]

        eos_id = self._tokenizer.eos_id()
        try:
            eos_token = self._tokenizer.id_to_piece(eos_id)
        except IndexError:
            eos_token = '<EOS>'
        _add_special_token(eos_token)
        self._eos_id = self._vocab[eos_token]

        for i in range(vocab_extra_ids):
            t = "<extra_id_{}>".format(i)
            _add_special_token(t)
            self._t5_tokens += [t]

        for t in ul2_denoiser_tokens:
            _add_special_token(t)
            self._ul2_tokens.append(t)

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L89
    def tokenize(self, text):
        ids = []
        idx = 0

        while 1:
            indices = {}
            for token in self._special_tokens:
                try:
                    indices[token] = text[idx:].index(token)
                except ValueError:
                    continue
            if len(indices) == 0:
                break

            next_token = min(indices, key=indices.get)
            next_idx = idx + indices[next_token]

            ids.extend(self._tokenizer.encode_as_ids(text[idx:next_idx]))
            ids.append(self._special_tokens[next_token])
            idx = next_idx + len(next_token)

        ids.extend(self._tokenizer.encode_as_ids(text[idx:]))
        return ids

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L125
    def detokenize(self, ids):
        text = ""
        last_i = 0

        for i, id in enumerate(ids):
            if id in self._inv_special_tokens:
                text += self._tokenizer.decode_ids(ids[last_i:i]) + " "
                text += self._inv_special_tokens[id] + " "
                last_i = i + 1

        text += self._tokenizer.decode_ids(ids[last_i:])
        return text.strip()

    @property
    def cls(self):
        return self._cls_id

    @property
    def sep(self):
        return self._sep_id

    @property
    def pad(self):
        return self._pad_id

    @property
    def bos_token_id(self):
        return self._bos_id

    @property
    def bos(self):
        return self._bos_id

    @property
    def eod(self):
        return self._eod_id

    @property
    def eos_token_id(self):
        return self._eos_id

    @property
    def eos(self):
        return self._eos_id

    @property
    def mask(self):
        return self._mask_id

    @property
    def additional_special_tokens_ids(self):
        return [self.vocab[k] for k in self._t5_tokens]

    @property
    def ul2_token_ids(self):
        return [self.vocab[k] for k in self._ul2_tokens]
