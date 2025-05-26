import os
import shutil
import torch
import random
import time
import multiprocessing as mp
from typing import Iterable
import matplotlib.pyplot as plt

from src.dummy_dataset import DummyDataset

class OverComsumptionError(Exception):
    pass

class ReplayBufferWrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_steps: int,
        buffer: list,
        buffer_samples: dict,
        buffer_insertion_times: list,
        buffer_size: int,
        write_queue: mp.Queue,
        sampling_strategy: str,  # or "age_based"
        rlock: mp.Lock,
        wlock: mp.Lock,
    ):
        self.dataset = dataset
        self.num_steps = num_steps
        self.buffer = buffer
        self.buffer_samples = buffer_samples
        self.buffer_insertion_times = buffer_insertion_times
        self.buffer_size = buffer_size
        self.write_queue = write_queue
        self.sampling_strategy = sampling_strategy
        self.wlock = wlock
        self.rlock = rlock
        
        if sampling_strategy not in ["uniform", "age_based"]:
            raise ValueError("sampling_strategy must be either 'uniform' or 'age_based'")
        if sampling_strategy == "age_based" and buffer_insertion_times is None:
            raise ValueError("buffer_insertion_times must be provided for age_based sampling")

    @classmethod
    def init_main_process(
        cls,
        dataset: torch.utils.data.Dataset,
        num_steps: int,
        buffer_size: int,
        cache_dir: str,
        max_index: int | None = None,
        sampling_strategy: str = "uniform"  # or "age_based"
    ):
        max_index = max_index if max_index is not None else len(dataset) - 1

        manager = mp.Manager()
        buffer = manager.list()
        buffer_samples = manager.dict()
        buffer_insertion_times = manager.list()
        write_queue = manager.Queue()
        wlock = manager.Lock()
        rlock = manager.Lock()

        os.makedirs(cache_dir, exist_ok=True)

        # initialise the buffer with random indexes
        indexes = random.sample(range(max_index + 1), buffer_size)
        for index in indexes:
            buffer.append(index)

        # prefill the buffer_samples with empty lists for each index
        for index in range(max_index + 1):
            buffer_samples[index] = manager.list()
            
        # prefill buffer_insertion_times 
        now = time.time()
        for _ in range(buffer_size):
            buffer_insertion_times.append(now)

        # start the writer process
        writer = mp.Process(
            target=ReplayBufferWrapper._writer_process,
            args=(write_queue, buffer_samples, cache_dir)
        )
        writer.start()        

        return writer, cls(
            dataset=dataset,
            num_steps=num_steps,
            buffer=buffer,
            buffer_samples=buffer_samples,
            buffer_insertion_times=buffer_insertion_times,
            buffer_size=buffer_size,
            write_queue=write_queue,
            sampling_strategy=sampling_strategy,
            rlock=rlock,
            wlock=wlock
        )

    @staticmethod
    def _writer_process(write_queue: mp.Queue, buffer_samples: dict, cache_dir: str):
        while True:
            item = write_queue.get()
            if item == "STOP":
                break
            index, sample = item
            file_path = os.path.join(
                cache_dir, f"{index}_{torch.randint(0, 100000, ()).item()}.pt"
            )
            torch.save(sample, file_path)
            buffer_samples[index].append(file_path)

    def add_to_buffer(self, index: int|torch.Tensor, sample: torch.Tensor, full_ok: bool=False, ignore_out_of_bounds: bool=False):
        # make sure index is an integer on CPU
        if isinstance(index, torch.Tensor):
            index = index.detach().cpu().item()
            
        # make sure sample is on CPU
        if isinstance(sample, torch.Tensor):
            sample = sample.detach().cpu()
        
        if index >= len(self.dataset):
            # If we unfortunately get an index that is out of bounds, we either sample a fresh sample or do 
            # nothing based on the ignore_out_of_bounds flag. This can happen because we do not control for 
            # a maximum forecast horizon, so the index can grow beyond the dataset size in principal.
            if ignore_out_of_bounds:
                return
            else:
                index = random.randint(0, len(self.dataset) - 1)
                sample = self.dataset[index]
            
        with self.wlock:
            if len(self.buffer) >= self.buffer_size:
                # If the buffer is full, we either remove a random sample or raise an exception based on full_ok.
                if not full_ok:
                    # full buffer strictly not allowed
                    raise Exception("Buffer is full")
                
                # remove a random sample to "make space for the new one"
                index_to_pop = random.randint(0, len(self.buffer) - 1)
                self.buffer.pop(index_to_pop)
                if self.buffer_insertion_times is not None:
                    self.buffer_insertion_times.pop(index_to_pop)
                # pop the sample from the buffer_samples§§
                if len(self.buffer_samples[index_to_pop]) == 1:
                    # only one => delete the entry
                    del self.buffer_samples[index_to_pop]
                elif len(self.buffer_samples[index_to_pop]) > 1:
                    # more than one => remove randomly
                    list_index_to_pop = random.randint(0, len(self.buffer_samples[index_to_pop]) - 1)
                    self.buffer_samples[index_to_pop].pop(list_index_to_pop)
                else:
                    # length is zero, nothing to do
                    pass
                
            # Add to buffer and write queue
            self.buffer.append(index)
            self.write_queue.put((index, sample))
            if self.buffer_insertion_times is not None:
                self.buffer_insertion_times.append(time.time())

    def __len__(self):
        return self.num_steps

    def __getitem__(self, *args, **kwargs):
        with self.rlock and self.wlock:
            if not self.buffer:
                # Thebuffer can end up being empty when the samples are consumed faster than they are added.
                # This can happen if the buffer is small, or when the number samples consumed per iteration
                # (i.e. batch_size * num_workers) is large compared to the buffer size.
                raise OverComsumptionError("Buffer is empty")

            # naive uniform sampling from the buffer
            if self.sampling_strategy == "uniform":
                rand_idx = random.randint(0, len(self.buffer) - 1)
            
            # age-based sampling (the older, the more likely)
            elif self.sampling_strategy == "age_based":
                now = time.time()
                ages = [now - t for t in self.buffer_insertion_times]
                probs = [1 / (age + 1e-6) for age in ages]  # add small constant to avoid division by zero
                probs = [p / sum(probs) for p in probs]
                rand_idx = random.choices(range(len(self.buffer)), weights=probs, k=1)[0]
                
            # pop the index from the buffer and the corresponding insertion time
            index = self.buffer.pop(rand_idx)
            if self.buffer_insertion_times is not None:
                    self.buffer_insertion_times.pop(rand_idx)
            
            # get the sample
            sample_paths = self.buffer_samples.get(index, [])
            if len(sample_paths) == 0:
                # If no samples are available in the buffer for that index (i.e. the index has been added
                # in the initialization phase), we read it from the dataset.
                sample = self.dataset[index]
            else:
                # Otherwise, we randomly sample a path from the available paths, read it, and remove from
                # the list of available paths and from disk.
                rand_sample_path = random.choice(sample_paths)
                sample = torch.load(rand_sample_path)
                os.remove(rand_sample_path)
                sample_paths.remove(rand_sample_path)

        return index, sample
    
    def plot_buffer(self, step: int, max_duration: int = 60):
        """
        make a histogram of the buffer contents. If sampling_strategy is "age_based",
        have a second plot showing the distribution of age (i.e. time since insertion).
        """
        if self.buffer_insertion_times is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.hist(self.buffer, bins=50, alpha=0.7, color='blue', density=True)
            ax.set_title("Buffer contents")
            ax.set(xlabel="Index", ylabel="Frequency")
            ax.set_xlim(0, len(self.dataset) - 1)
            ax.set_ylim(0, 10/len(self.dataset))  # adjust y-limits for better visibility
            
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            ax1.hist(self.buffer, bins=50, alpha=0.7, color='blue', density=True)
            ax1.set_title("Buffer contents")
            ax1.set(xlabel="Index", ylabel="Frequency")
            ax1.set_xlim(0, len(self.dataset) - 1)
            ax1.set_ylim(0, 10/len(self.dataset))
            ages = [time.time() - t for t in self.buffer_insertion_times]
            ax2.hist(ages, bins=50, alpha=0.7, color='orange', density=True)
            ax2.set_title("Buffer ages")
            ax2.set(xlabel="Age (seconds)", ylabel="Frequency")
            ax2.set_xlim(0, max_duration)
            ax2.set_ylim(0, 1)
        
        suptitle = f"Buffer at step {step} ({self.sampling_strategy} sampling)\n"
        suptitle += ", ".join([
            f"Buffer size: {len(self.buffer)}",
            f"Dataset size: {len(self.dataset)}"
        ])
        fig.suptitle(suptitle)
        fig.tight_layout()
        
        return fig
        

def close_main_process(writer: mp.Process, dataloader: torch.utils.data.DataLoader, cache_dir: str):
    dataloader.dataset.write_queue.put("STOP")
    writer.join()
    shutil.rmtree(cache_dir, ignore_errors=True)

if __name__ == "__main__":
    data = torch.zeros(5000, 3, 4, 8)
    cache_dir = os.path.join(os.getcwd(), "cache")
    buffer_size = 200
    num_steps = 1000
    batch_size = 5

    dataset = DummyDataset(data)
    writer, rb_wrapper = ReplayBufferWrapper.init_main_process(
        dataset=dataset,
        num_steps=num_steps * batch_size,
        buffer_size=buffer_size,
        cache_dir=cache_dir,
        sampling_strategy="uniform" #"age_based",
    )

    dataloader = torch.utils.data.DataLoader(
        rb_wrapper,
        batch_size=batch_size,
        num_workers=10,
        shuffle=False,
        multiprocessing_context="forkserver"
    )

    for i, (indexes, batch) in enumerate(dataloader):
        print(f"Batch {i}: indexes: {indexes} batch shape: {batch.shape} unique: {torch.unique(batch)}, buffer length: {len(dataloader.dataset.buffer)}")
        new_batch = i * torch.ones_like(batch)
        new_indexes = indexes + 1
        for index, sample in zip(new_indexes, new_batch):
            dataloader.dataset.add_to_buffer(index.item(), sample, full_ok=True)

    close_main_process(writer, dataloader, cache_dir)
    print("Done.")
