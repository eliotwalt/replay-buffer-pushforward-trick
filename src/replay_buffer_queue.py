import os
import shutil
import torch
import random
import multiprocessing as mp
from typing import Iterable

from src.dummy_dataset import DummyDataset

class OverComsumptionError(Exception):
    """Custom exception for overconsumption of the buffer."""

class ReplayBufferWrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_steps: int,
        buffer: mp.Queue,
        buffer_samples: dict,
        write_queue: mp.Queue,
        rlock: mp.Lock,
        wlock: mp.Lock,
        buffer_timeout: float=60 # i.e. 1 minutes
    ):
        self.dataset = dataset
        self.num_steps = num_steps
        self.buffer = buffer
        self.buffer_samples = buffer_samples
        self.write_queue = write_queue
        self.wlock = wlock
        self.rlock = rlock
        self.buffer_timeout = buffer_timeout
        
    @classmethod
    def init_main_process(
        cls,
        dataset: torch.utils.data.Dataset,
        num_steps: int,
        buffer_size: int,
        cache_dir: str,
        buffer_timeout: float=60, # i.e. 1 minutes
        max_index: int|None=None
    ) -> None:
        """Initialize the main process for the dataset."""
        max_index = max_index if max_index is not None else len(data) - 1
        
        # create multiprocessing objects
        manager = mp.Manager()
        buffer = manager.Queue(maxsize=buffer_size)
        buffer_samples = manager.dict()
        wlock = manager.Lock()
        rlock = manager.Lock()
        write_queue = manager.Queue()
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # prefill the buffer with random samples
        indexes = random.sample(range(max_index+1), buffer_size)
        for index in indexes: buffer.put(index)

        # fill the sample buffer with queues for all possible indexes
        for index in range(max_index+1): buffer_samples[index] = manager.Queue(maxsize=buffer_size)
        
        # start the writer process
        writer = mp.Process(
            target=ReplayBufferWrapper._writer_process,
            args=(write_queue, buffer_samples, cache_dir, buffer_size)
        )
        writer.start()
        
        return writer, cls(
            dataset=dataset,
            num_steps=num_steps,
            buffer=buffer,
            buffer_samples=buffer_samples,
            write_queue=write_queue,
            rlock=rlock,
            wlock=wlock,
            buffer_timeout=buffer_timeout
        )
        
    @staticmethod
    def _writer_process(write_queue: mp.Queue, buffer_samples: dict, cache_dir: str, maxsize: int):
        make_path = lambda idx: os.path.join(
            cache_dir, f"{idx}_{torch.randint(0, 100000, ()).item()}.pt"
        )
        while True:
            item = write_queue.get()
            if item == "STOP":
                break
            index, sample = item
            file_path = make_path(index)
            while os.path.exists(file_path):
                file_path = make_path(index)
            torch.save(sample, file_path)

            if index not in buffer_samples:
                buffer_samples[index] = mp.Queue(maxsize=maxsize)
            buffer_samples[index].put(file_path)
            
    def add_to_buffer(self, index: int|torch.Tensor, sample: torch.Tensor, full_ok: bool=False):
        if isinstance(index, torch.Tensor):
            index = index.detach().cpu().item()
        
        if index >= len(self.dataset):
            # If we unfortunately get an index that is out of bounds, we sample a fresh sample
            old_index = index
            index = random.randint(0, len(self.dataset) - 1)
            sample = self.dataset[index]
            print(f"Index {old_index} is out of bounds for data of size {len(self.dataset)}. Sampled a new index {index} instead.")
            
        sample = sample.detach().cpu()
        
        with self.wlock:
            if self.buffer.full():
                if not full_ok:
                    raise Exception("Buffer is full and full_ok is False. Cannot add new sample.")
                else:
                    # make space
                    _index = self.buffer.get_nowait()
                    _ = self.buffer_samples[_index].get()
            
            # add sample to buffer
            self.buffer.put(index)
            
            # write to cache
            self.write_queue.put((index, sample))
            
    def __len__(self):
        return self.num_steps
    
    def __getitem__(self, *args, **kwargs):
        with self.rlock:                        
            # get index from buffer
            # we use a timeout because the dataloading workers might consume the buffer faster than we can fill it
            # when num_workers > batch_size
            # Note that this is a blocking call. It will result in an error when the buffer is close to num_workers * batch_size
            try: index = self.buffer.get(timeout=self.buffer_timeout)
            except: 
                # This might still happen if the buffer is way too small compared to the number of workers and the batch size.
                # We cannot think of an efficient fix here, so we jsut inform the user.
                raise OverComsumptionError(
                    "Buffer is empty and no new data was added within timeout."\
                    "This usually happens when the buffer is too small compared to the product of the number of workers and the batch size."\
                    "Consider increasing the buffer size, reducing the number of workers and/or reducing the batch size."
                )

            # if index is not in buffer_samples, create a new queue for it
            if index not in self.buffer_samples:
                raise Exception(f"Index {index} (type: {type(index)}) not in buffer_samples")
                
            # if the buffer_samples[index] is empty, get from data, otherwise from buffer_samples
            if self.buffer_samples[index].empty():
                sample = self.dataset[index]
            else:
                sample_path = self.buffer_samples[index].get(timeout=self.buffer_timeout)
                sample = torch.load(sample_path)
                os.remove(sample_path)
            
            # print(f"PID: {mp.current_process().pid} loaded item: {index}, sample: {sample.shape}, buffer size: {self.buffer.qsize()}")
            
        return index, sample
    
def close_main_process(writer: mp.Process, dataloader: torch.utils.data.DataLoader, cache_dir: str):
    """Close the main process and do some housekeeping."""
    dataloader.dataset.write_queue.put("STOP")
    writer.join()
    shutil.rmtree(cache_dir, ignore_errors=True)
    
if __name__ == "__main__":
    
    print("Configuring ...")
    data = torch.zeros(5000, 3, 120, 61)
    cache_dir = os.path.join(os.getcwd(), "cache")
    buffer_size = 200
    num_steps = 1000
    batch_size = 2
    buffer_timeout = 10
        
    num_steps_per_worker = num_steps * batch_size
    
    print("Creating dataloader...")
    dataset = DummyDataset(data)
    writer, rb_wrapper = ReplayBufferWrapper.init_main_process(
        dataset=dataset,
        num_steps=num_steps_per_worker,
        buffer_size=buffer_size,
        cache_dir=cache_dir,
        buffer_timeout=buffer_timeout
    )
    dataloader = torch.utils.data.DataLoader(
        rb_wrapper,
        batch_size=batch_size,
        num_workers=12,
        shuffle=False, # shuffle has no effect because we are using a buffer with random samples
        multiprocessing_context="forkserver"
    )
    
    print("Starting dataloader...")
    for i, (indexes, batch) in enumerate(dataloader):
        print(f"Batch {i}: indexes: {indexes} batch: {batch.shape} unique: {torch.unique(batch)}")
        
        new_batch = i * torch.ones_like(batch)
        new_indexes = indexes + 1
        
        # add new samples to buffer
        for index, sample in zip(new_indexes, new_batch):
            dataloader.dataset.add_to_buffer(index.item(), sample, full_ok=True)    
        
    print("Closing dataloader...")
    close_main_process(writer, dataloader, cache_dir)
    print("Done.")