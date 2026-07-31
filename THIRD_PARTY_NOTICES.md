# Third-party notices

AuroraLM's original source code and documentation are licensed under the
Apache License 2.0. Third-party software, source data, and generated artefacts
remain subject to their own licenses and terms. Nothing in AuroraLM's license
relicenses those materials.

## SmolLM Corpus

The tokenizer-development workflow can sample the `fineweb-edu-dedup` and
`cosmopedia-v2` subsets of Hugging Face's
[SmolLM Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/3ba9d605774198c5868892d7a8deda78031a781f),
pinned to revision `3ba9d605774198c5868892d7a8deda78031a781f`.

The dataset is published under the
[Open Data Commons Attribution License (ODC-By) 1.0](https://opendatacommons.org/licenses/by/1-0/).
Users of the sampling workflow are responsible for reviewing the dataset card,
source-specific notices, and applicable attribution requirements before
redistributing sampled data or derived artefacts. No SmolLM source text is
intended to be committed to this repository.

## PyTorch

[PyTorch](https://github.com/pytorch/pytorch) is a runtime dependency and is
distributed under its
[BSD-style license](https://github.com/pytorch/pytorch/blob/main/LICENSE).
PyTorch source code is not vendored in this repository.

## Hugging Face Tokenizers

[Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) is a
runtime and tokenizer-development dependency and is distributed under the
[Apache License 2.0](https://github.com/huggingface/tokenizers/blob/main/LICENSE).
Its source code is not vendored in this repository.

## Generated tokenizer artefact

`artifacts/aurora_bpe_8192_v1/` contains a generated tokenizer definition,
configuration, manifest, and validation receipt. It does not intentionally
contain the source training corpus or pretrained model weights. To the extent
that third-party rights apply to the generated artefact, the relevant
third-party terms remain in effect.
