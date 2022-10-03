from nltk.metrics.distance import edit_distance
import faiss
from queue import Queue
from sortedcontainers import SortedList


def get_split_points(array, size):
    assert size > 1

    prev = array[0]
    split_points = [0]
    for i in range(1, size):
        if prev != array[i]:
            prev = array[i]
            split_points.append(i)

    split_points.append(size)
    return split_points


def move_item_to_end_(arr, items):
    for item in items:
        arr.insert(len(arr), arr.pop(arr.index(item)))


def move_item_to_start_(arr, items):
    for item in items[::-1]:
        arr.insert(0, arr.pop(arr.index(item)))


def scaled_edit_distance(a: str, b: str):
    return edit_distance(a, b) / max(len(a), len(b))


def custom_index_cpu_to_gpu_multiple(resources, index, co=None, gpu_nos=None):
    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    if gpu_nos is None:
        gpu_nos = range(len(resources))
    for i, res in zip(gpu_nos, resources):
        vdev.push_back(i)
        vres.push_back(res)
    index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
    index.referenced_objects = resources
    return index


class DroppingPriorityQueue:
    """
    Priority queue with maximum size. Tail will be automatically dropped when reaching max size
    """
    def __init__(self, maxsize=None, reverse=False):
        self.reverse = reverse
        self.maxsize = maxsize
        self._queue = SortedList()

    def put(self, item):
        self._queue.add(item)
        if self.maxsize is not None and len(self._queue) > self.maxsize:
            if self.reverse:
                self._queue.pop(0)
            else:
                self._queue.pop(-1)

    def get(self):
        if self.reverse:
            return self._queue.pop(-1)
        else:
            return self._queue.pop(0)

    def __len__(self):
        return len(self._queue)


def equal_split(n, k):
    if n % k == 0:
        return [n // k for _ in range(k)]
    else:
        return [n // k for _ in range(k - 1)] + [n % k]
