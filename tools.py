import numpy as np

def sort_each_np_array_column(array: np.ndarray) -> np.ndarray:
    """
    Sort the elements of each column into numerical order.
    For example:
                  np.array([[1, 0, 3],
                            [0, 5, 2],
                            [3, 0, 8]])
    sorts to:     
                  np.array([[0, 0, 2],
                            [1, 0, 3],
                            [3, 5, 8]])
    """
    return np.sort(array.T).T

def sort_np_array_rows_lexicographically(array: np.ndarray) -> np.ndarray:
    """
    Permutes rows of a numpy array (individual rows are preserved) so that the rows end up in lexicographical order.
    E.g.:
                  np.array([[1, 0, 2],
                            [0, 5, 2],
                            [3, 0, 8]])
    sorts to:     
                  np.array([[0, 5, 2],
                            [1, 0, 2],
                            [3, 0, 8]])
    """
    return array[np.lexsort(array.T[::-1])]
