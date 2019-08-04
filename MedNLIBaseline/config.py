from pathlib import Path
import enum

import dataclasses


@enum.unique
class WordEmbeddings(enum.Enum):
    GloVe = 'glove'
    MIMIC = 'mimic'
    BioAsq = 'bioasq'
    WikiEn = 'wikien'
    WikiEnMIMIC = 'wikien_mimic'
    GloVeBioAsq = 'glove_bioasq'
    GloVeBioAsqMIMIC = 'glove_bioasq_mimic'
    MimicPubMed = 'mimic_pubmed'


@enum.unique
class Models(enum.Enum):
    Simple = 'simple' # CBOW model
    InferSent = 'infersent'
    ESIM = 'esim'


@dataclasses.dataclass
class Config:
    data_dir: Path

    model: Models = Models.Simple
    word_embeddings: WordEmbeddings = WordEmbeddings.MimicPubMed

    #-word_embeddings: WordEmbeddings = WordEmbeddings.MIMIC # Original

    lowercase: bool = True
    max_len: int = 50
    hidden_size: int = 128
    dropout: float = 0.6
    trainable_embeddings: bool = False

    weight_decay: float = 0.0001
    learning_rate: float =  1e-3
    max_grad_norm: float = 5.0
    batch_size: int = 64
    nb_epochs: int = 1


    @property
    def mednli_dir(self) -> Path:
        return self.data_dir.joinpath('mednli/')

    @property
    def cache_dir(self) -> Path:
        return self.data_dir.joinpath('cache/')

    @property
    def word_embeddings_dir(self) -> Path:
        return self.data_dir.joinpath('word_embeddings/')

    @property
    def models_dir(self) -> Path:
        return self.data_dir.joinpath('ted_models/')
