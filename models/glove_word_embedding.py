import os
import pickle as pkl
import typing as t
import warnings

import sh
import torch
from torch import nn
from tqdm import tqdm


class GloVe(nn.Module):
    _file_path: str  # location to read world embeddings from
    _size: int  # dimension of an embedding
    _words: t.List[str]  # list of world possible words
    _vectors: torch.nn.Parameter  # 2D matrix; first dimension is the word id corresponding to the list 'words', second dimension is the embedding vector of length 'size'

    def __init__(self, file_path: str, size: int, load_precached: bool = True):
        super(GloVe, self).__init__()
        self._file_path = file_path
        self._size = size
        self._words = []

        if self._file_path.startswith("s3://"):
            new_path = self._file_path.replace("s3://", "/tmp/")
            logger.info(f"Syncing {self._file_path} to {new_path}")
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            sh.aws.s3.sync(os.path.dirname(self._file_path), os.path.dirname(new_path))
            self._file_path = new_path

        precached_file_name = os.path.splitext(os.path.basename(self._file_path))
        precached_file_name = "".join([precached_file_name[0], ".pkl"])
        precached_file_path = os.path.join(os.path.dirname(self._file_path), precached_file_name)
        if not os.path.exists(precached_file_path):
            load_precached = False

        if load_precached:
            with open(precached_file_path, "rb") as f:
                data = pkl.load(f)
            assert self._size == data["size"]
            assert os.path.basename(self._file_path) == os.path.basename(data["file_path"])
            self._words = data["words"]
            self._vectors = torch.nn.Parameter(data["vectors"])
        else:
            vectors = []
            with open(self._file_path, "r") as f:
                lines = f.readlines()
            for line in tqdm(lines, desc="Loading word embedding"):
                line = line.split(" ")
                word, vector = line[0], torch.Tensor([float(value) for value in line[1:]])
                assert word not in self._words
                assert len(vector) == self._size
                self._words.append(word)
                vectors.append(vector)
            self._vectors = torch.nn.Parameter(torch.stack(vectors, dim=0))
            try:
                print(f"Writing embedding results to {precached_file_path}")
                with open(precached_file_path, "wb") as f:
                    pkl.dump(
                        {
                            "vectors": self._vectors,
                            "words": self._words,
                            "size": self._size,
                            "file_path": self._file_path,
                        },
                        f,
                    )
            except:
                warnings.warn("Cannot write the embeddings data")

    def vectorize(self, word: str) -> torch.Tensor:
        return self._vectors[self._words.index(word.lower())]

    def encode_single(self, word: str) -> torch.Tensor:
        try:
            return self.vectorize(word)
        except ValueError:

            def encode_split(word_str: str, splitters: t.Collection[str]):
                word_str = [word_str]
                for splitter in splitters:
                    word_out = []
                    for w in word_str:
                        word_out.extend(w.split(splitter))
                    word_str = word_out
                representations = [self.vectorize(w) for w in word_str]
                return sum(representations) / len(representations)

            return encode_split(word, [" ", "-"])

    def encode_list(self, words: t.Sequence[str]) -> torch.Tensor:
        return torch.stack([self.encode_single(word) for word in words])

    def forward(self, inp: t.Union[t.Sequence[str], str]) -> torch.Tensor:
        if isinstance(inp, str):
            return self.encode_single(inp)
        else:
            return self.encode_list(inp)

    def closest_word(self, vector: torch.Tensor) -> t.Tuple[str, torch.Tensor]:
        """
        returns the word with closest embedding representation and its PMF
        """
        dot_products = self._vectors.matmul(vector)
        index = dot_products.argmax()
        values = torch.softmax(dot_products, dim=0)
        return self._words[index], values


if __name__ == "__main__":
    embeddings = GloVe(file_path="/home/krm/datasets/glove/glove.6B.100d.txt", size=100)
    test_words = ["the", "welcome", "hello"]
    test_vector = torch.rand([100])
    for w in test_words:
        print(embeddings(w))
    print(embeddings.closest_word(test_vector))
