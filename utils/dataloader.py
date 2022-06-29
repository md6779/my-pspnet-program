from torch.utils.data import DataLoader


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iter = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iter)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iter = super().__iter__()
            batch = next(self.dataset_iter)
        return batch
