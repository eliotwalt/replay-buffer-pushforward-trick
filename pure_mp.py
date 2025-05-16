import torch
import random
import multiprocessing as mp
from multiprocessing import Manager

class OverComsumptionError(Exception):
    """Custom exception for overconsumption of the buffer."""

class RBDS(torch.utils.data.Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        num_steps: int,
        buffer: mp.Queue,
        buffer_samples: dict, # actually a manager.dict()
        lock: mp.Lock,
        buffer_timeout: float=60 # i.e. 1 minutes
    ):
        self.data = data
        self.num_steps = num_steps
        self.buffer = buffer
        self.buffer_samples = buffer_samples
        self.lock = lock
        self.buffer_timeout = buffer_timeout
            
    def add_to_buffer(self, index: int|torch.Tensor, sample: torch.Tensor, full_ok: bool=False):
        if isinstance(index, torch.Tensor):
            index = index.item()
        
        if index >= len(self.data):
            # If we unfortunately get an index that is out of bounds, we sample a fresh sample
            old_index = index
            index = random.randint(0, len(self.data) - 1)
            sample = self.data[index]
            print(f"Index {old_index} is out of bounds for data of size {len(self.data)}. Sampled a new index {index} instead.")
        
        with self.lock:
            if self.buffer.full():
                if not full_ok:
                    raise Exception("Buffer is full and full_ok is False. Cannot add new sample.")
                else:
                    _ = self.buffer.get_nowait()
            
            # add sample to buffer
            self.buffer.put(index)
            
            # add sample to shared queue
            self.buffer_samples[index].put(sample)
            
    def __len__(self):
        return self.num_steps
    
    def __getitem__(self, *args, **kwargs):
        with self.lock:                        
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
            sample = self.data[index] if self.buffer_samples[index].empty() else self.buffer_samples[index].get()
            
            print(f"PID: {mp.current_process().pid} loaded item: {index}, sample: {sample.shape}, buffer size: {self.buffer.qsize()}")
            
        return index, sample
    
def get_dataloader(data: torch.Tensor, buffer_size: int, num_steps: int, buffer_timeout: float=60):
    manager = mp.Manager()
    buffer = manager.Queue(maxsize=buffer_size)
    buffer_samples = manager.dict()
    lock = manager.Lock()
    
    # calculate largest possible index that can be supervised by the dataset
    largest_index = len(data) - 1 # TODO: in true setting this is set by the forecast horizon scheduler

    # Fill the buffer before creating the dataset
    indexes = random.sample(range(largest_index+1), buffer_size) # +1 to include the largest index
    for index in indexes:
        buffer.put(index)
        
    # fill the sample buffer with queues for all possible indexes 
    for index in range(largest_index+1): # +1 to include the largest index
        buffer_samples[index] = manager.Queue(maxsize=len(data)) 

    # torch 
    dataset = RBDS(data, num_steps, buffer, buffer_samples, lock, buffer_timeout)
    return torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=12, shuffle=False, multiprocessing_context="forkserver")
    
if __name__ == "__main__":
    
    print("Configuring ...")
    data = torch.zeros(5000, 3, 120, 61)
    buffer_size = 2000
    num_steps = 10000
    batch_size = 1
    buffer_timeout = 10
        
    num_steps_per_worker = num_steps * batch_size
    
    
    print("Creating dataloader...")
    dataloader = get_dataloader(data, buffer_size, num_steps, buffer_timeout)
    
    print("Starting dataloader...")
    for i, (indexes, batch) in enumerate(dataloader):
        print(f"Batch {i}: indexes: {indexes} batch: {batch.shape} unique: {torch.unique(batch)}")
        
        new_batch = i * torch.ones_like(batch)
        new_indexes = indexes + 1
        
        # add new samples to buffer
        for index, sample in zip(new_indexes, new_batch):
            dataloader.dataset.add_to_buffer(index.item(), sample, full_ok=True)    