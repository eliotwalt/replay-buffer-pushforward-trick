import os
import shutil
import torch
import random
import multiprocessing as mp

class OverComsumptionError(Exception):
    """Custom exception for overconsumption of the buffer."""

def writer_process(write_queue: mp.Queue, buffer_samples: dict, cache_dir: str, maxsize: int):
    """Process that writes samples to the buffer's cache"""
    os.makedirs(cache_dir, exist_ok=True)
    make_path = lambda index: os.path.join(cache_dir, f"{index}_{torch.randint(0, 100000, ()).item()}.pt")
    while True:
        # Get whatever is in the write queue
        item = write_queue.get()
        if item == "STOP":
            break
        index, sample = item
        
        # Create a unique file path for the sample
        file_path = make_path(index)
        while os.path.exists(file_path):
            file_path = make_path(index)
            
        # Save the sample to the file
        torch.save(sample, file_path)
        
        # add index to buffer_samples' queue
        if index not in buffer_samples:
            buffer_samples[index] = mp.Queue(maxsize=maxsize)
        buffer_samples[index].put(file_path)

class RBDS(torch.utils.data.Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        num_steps: int,
        buffer: mp.Queue,
        buffer_samples: dict,
        write_queue: mp.Queue,
        rlock: mp.Lock,
        wlock: mp.Lock,
        buffer_timeout: float=60 # i.e. 1 minutes
    ):
        self.data = data
        self.num_steps = num_steps
        self.buffer = buffer
        self.buffer_samples = buffer_samples
        self.write_queue = write_queue
        self.wlock = wlock
        self.rlock = rlock
        self.buffer_timeout = buffer_timeout
            
    def add_to_buffer(self, index: int|torch.Tensor, sample: torch.Tensor, full_ok: bool=False):
        if isinstance(index, torch.Tensor):
            index = index.detach().cpu().item()
        
        if index >= len(self.data):
            # If we unfortunately get an index that is out of bounds, we sample a fresh sample
            old_index = index
            index = random.randint(0, len(self.data) - 1)
            sample = self.data[index]
            print(f"Index {old_index} is out of bounds for data of size {len(self.data)}. Sampled a new index {index} instead.")
            
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
                sample = self.data[index]
            else:
                sample_path = self.buffer_samples[index].get(timeout=self.buffer_timeout)
                sample = torch.load(sample_path)
                os.remove(sample_path)
            
            print(f"PID: {mp.current_process().pid} loaded item: {index}, sample: {sample.shape}, buffer size: {self.buffer.qsize()}")
            
        return index, sample
    
def get_dataloader(data: torch.Tensor, cache_dir: str, buffer_size: int, num_steps: int, buffer_timeout: float=60):
    manager = mp.Manager()
    buffer = manager.Queue(maxsize=buffer_size)
    buffer_samples = manager.dict()
    wlock = manager.Lock()
    rlock = manager.Lock()
    
    # create a write queue for the writer process
    write_queue = manager.Queue()
    writer = mp.Process(target=writer_process, args=(write_queue, buffer_samples, cache_dir, len(data)))
    writer.start()
    
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
    dataset = RBDS(data=data, num_steps=num_steps, buffer=buffer, buffer_samples=buffer_samples, write_queue=write_queue, rlock=rlock, wlock=wlock, buffer_timeout=buffer_timeout)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, num_workers=12, shuffle=False, multiprocessing_context="forkserver")
    return dataloader, writer
    
if __name__ == "__main__":
    
    print("Configuring ...")
    data = torch.zeros(5000, 3, 120, 61)
    cache_dir = os.path.join(os.getcwd(), "cache")
    buffer_size = 2000
    num_steps = 10000
    batch_size = 1
    buffer_timeout = 10
        
    num_steps_per_worker = num_steps * batch_size
    
    
    print("Creating dataloader...")
    dataloader, writer = get_dataloader(data, cache_dir, buffer_size, num_steps, buffer_timeout)
    
    print("Starting dataloader...")
    for i, (indexes, batch) in enumerate(dataloader):
        print(f"Batch {i}: indexes: {indexes} batch: {batch.shape} unique: {torch.unique(batch)}")
        
        new_batch = i * torch.ones_like(batch)
        new_indexes = indexes + 1
        
        # add new samples to buffer
        for index, sample in zip(new_indexes, new_batch):
            dataloader.dataset.add_to_buffer(index.item(), sample, full_ok=True)    
        
    print("Stopping writer...")
    dataloader.dataset.write_queue.put("STOP")
    writer.join()
    
    shutil.rmtree(cache_dir)
    print("Done.")