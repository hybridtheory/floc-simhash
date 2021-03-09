from typing import Optional, Union, Any

import numpy as np
import scipy
from sklearn.base import TransformerMixin


class SimHashTransformer(TransformerMixin):
    def __init__(self, n_bits: int):
        """Initialize a transformer capable of computing the SimHash algorithm on vectorized data.

        :param n_bits: Precision to hash vectors to.
        :type n_bits: int
        """
        self.n_bits = n_bits
        self._vectors: Optional[np.array] = None

    def fit(
        self, X: Union[np.array, scipy.sparse.csr_matrix], y: Any = None
    ) -> "SimHashTransformer":
        """Fit the transformer to the given data.

        This amounts to choosing random unit vectors of dimension `n`, where `X` is of dimension
        `(number_of_documents, n)`.

        :param X: vectorized data to be hashed. Can either be a numpy array or a sparse matrix.
        :type X: Union[np.array, scipy.sparse.csr_matrix]
        :param y: not used, left for matching the scikit-learn transformer pattern. Defaults to
        None.
        :type y: Any, optional
        :return: the fitted instance itself.
        :rtype: "SimHashTransformer"
        """
        random_vectors = np.random.randn(X.shape[1], self.n_bits)
        self._vectors = random_vectors / np.linalg.norm(
            random_vectors, axis=0, keepdims=True
        )
        return self

    def transform(
        self, X: Union[np.array, scipy.sparse.csr_matrix], y: Any = None
    ) -> np.array:
        """Compute the SimHashes of the vectors in X as an array of hexadecimal strings.

        :param X: 2D array, where each row contains a vector to be hashed.
        :type X: Union[np.array, scipy.sparse.csr_matrix]
        :param y: not used, left for matching the scikit-learn transformer pattern. Defaults to
        None.
        :type y: Any, optional
        :return: an array of hexadecimal strings
        :rtype: np.array
        :raises ValueError: if the fit method has not been previously called.
        """
        if self._vectors is None:
            raise ValueError("The fit method has not been called")
        mul = X.dot(self._vectors)
        bits = np.where(mul > 0, 1, 0)
        powers = 2 ** np.arange(self.n_bits)
        int_values = np.dot(bits, powers)
        hex_converter = np.vectorize(lambda x: str(hex(x)))
        return hex_converter(int_values)
