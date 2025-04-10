from typing import List, Optional, Union, Any, NoReturn
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances


class BestShotsSelector:
    """
    A class for selecting diverse or query-similar examples from a collection.

    This class implements methods to select the most diverse subset of examples
    using embedding-based similarity metrics, or to select examples most similar
    to a query.
    """

    def __init__(self, examples: Optional[List[str]] = None):
        """
        Initialize a BestShotsSelector instance.

        Args:
            examples: Optional list of text examples to select from
        """
        self._examples = examples
        self._embeddings: Optional[np.ndarray] = None

    def _get_embeddings(self) -> None:
        """
        Generate embeddings for all examples using a sentence transformer model.

        This method lazily computes embeddings when needed and caches them
        for future use.
        """
        if not self._embeddings:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            self._embeddings = model.encode(
                self.examples, normalize_embeddings=True
            )  # using normalise to make cosine sim = dot product

    @property
    def examples(self) -> Optional[List[str]]:
        """
        Get the list of examples.

        Returns:
            The list of examples, or None if not set
        """
        return self._examples

    @property
    def embeddings(self) -> Optional[List[str]]:
        """
        Get the examples if embeddings have been computed.

        Returns:
            The list of examples if embeddings exist, otherwise None
        """
        if self._embeddings is not None:
            return self._examples
        return None

    def min_max_diverse_subset(self, k: int = 10, seed: int = 9425) -> List[str]:
        """
        Selects k diverse sentences from the input list using Min-Max Diversity.

        This method implements a greedy algorithm that maximizes the minimum distance
        between selected examples based on their embeddings.

        Args:
            k: Number of diverse sentences to select (k <= N)
            seed: Random seed for reproducibility

        Returns:
            List of selected diverse sentences

        Raises:
            AssertionError: If k is greater than the number of examples
        """
        assert k <= len(
            self._examples
        ), "k must be less than or equal to number of sentences"

        if self._embeddings is None:
            self._get_embeddings()

        # Step 2: Min-Max Diversity Selection
        np.random.seed(seed)
        selected_indices = [np.random.choice(len(self.examples))]

        while len(selected_indices) < k:
            dists = cosine_distances(
                self._embeddings, self._embeddings[selected_indices]
            )
            min_dists = dists.min(axis=1)
            min_dists[selected_indices] = -1  # mask already selected
            next_idx = np.argmax(min_dists)
            selected_indices.append(next_idx)

        # Step 3: Return selected sentences
        return [self.examples[i] for i in selected_indices]

    def query_similarity_select(
        self, k: int, query: Optional[str] = None
    ) -> Optional[List[str]]:
        """
        Selects k examples most similar to a given query.

        This method ranks examples by their similarity to the query
        and returns the top k most similar ones.

        Args:
            k: Number of examples to select (k <= N)
            query: The query text to compare examples against

        Returns:
            List of selected sentences similar to the query, or None if no query is provided

        Raises:
            AssertionError: If k is greater than the number of examples
        """
        if not query:
            return None

        assert k <= len(
            self._examples
        ), "k must be less than or equal to number of sentences"

        if self._embeddings is None:
            self._get_embeddings()

        selected_indices = []
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embeddings = model.encode(query, normalize_embeddings=True)
        dists = cosine_distances(self._embeddings, query_embeddings.reshape(1, -1))
        dists = [d[0] for d in dists]
        while len(selected_indices) < k:
            next_idx = np.argmin(dists)
            selected_indices.append(next_idx)
            del dists[next_idx]

        # Step 3: Return selected sentences
        return [self.examples[i] for i in selected_indices]
