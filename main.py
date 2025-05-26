import os
import torch
import argparse
import imageio
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import time

from src.replay_buffer_list import ReplayBufferWrapper, DummyDataset, close_main_process

def get_args():
    p = argparse.ArgumentParser(description="Replay Buffer Example")
    p.add_argument("--data_dims", type=int, nargs="+", default=[3], help="Dimensions of a data sample")
    p.add_argument("--cache_dir", type=str, default=os.path.join(os.getcwd(), "cache"), help="Directory to cache the data")
    p.add_argument("--buffer_size", type=int, default=100, help="Size of the replay buffer")
    p.add_argument("--num_steps", type=int, default=1000, help="Number of steps to run")
    p.add_argument("--batch_size", type=int, default=2, help="Batch size for the dataloader")
    p.add_argument("--num_workers", type=int, default=10, help="Number of workers for the dataloader")
    p.add_argument("--sampling_strategy", type=str, default="age_based", choices=["uniform", "age_based"], help="Sampling strategy for the replay buffer")
    p.add_argument("--ignore_out_of_bounds", action="store_true", help="Ignore out of bounds samples when adding to the buffer")
    p.add_argument("--max_duration", type=int, default=120, help="Maximum duration for the buffer plot in seconds")
    return p.parse_args()

def main():
    args = get_args()
    
    os.makedirs(args.cache_dir, exist_ok=True)
    img_cache_dir = os.path.join(args.cache_dir, "imgs")
    os.makedirs(img_cache_dir, exist_ok=True)
    sample_cache_dir = os.path.join(args.cache_dir, "samples")
    os.makedirs(sample_cache_dir, exist_ok=True)
    
    print("Configuring ...")
    data = torch.zeros(5000, *args.data_dims)
    
    print("Creating dataloader...")
    dataset = DummyDataset(data)
    rb_writer, rb_wrapper = ReplayBufferWrapper.init_main_process(
        dataset=dataset,
        num_steps=args.num_steps * args.batch_size,
        buffer_size=args.buffer_size,
        cache_dir=sample_cache_dir,
        sampling_strategy=args.sampling_strategy,
    )
    
    dataloader = torch.utils.data.DataLoader(
        rb_wrapper,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,  # shuffle has no effect because we are using a buffer with random samples
        multiprocessing_context="forkserver"
    )
    
    print("Starting dataloader...")
    imgs = []
    t0 = time.time()
    for i, (indexes, batch) in enumerate(dataloader):
        print(f"Batch {i}: indexes: {indexes} batch: {batch.shape} unique: {torch.unique(batch)}")
        
        new_batch = i * torch.ones_like(batch)
        new_indexes = indexes + 1
        
        # add new samples to buffer
        for index, sample in zip(new_indexes, new_batch):
            dataloader.dataset.add_to_buffer(index.item(), 
                                             sample, 
                                             full_ok=True, 
                                             ignore_out_of_bounds=args.ignore_out_of_bounds)
        
        # break if we have reached the maximum duration
        if time.time() - t0 > args.max_duration:
            print(f"Stopping after {args.max_duration} seconds.")
            break
        
        # plot the buffer
        fig = dataloader.dataset.plot_buffer(step=i+1, max_duration=args.max_duration)
        img_path = os.path.join(img_cache_dir, f"buffer_{i:04d}.png")
        fig.savefig(img_path)
        imgs.append(img_path)
        plt.close(fig)
    
    # make a gif of the buffer    
    print("Creating GIF of the buffer...")
    with imageio.get_writer(
        f"gifs/{args.sampling_strategy}_ignoreOOB={args.ignore_out_of_bounds}.gif",
        mode='I',
        duration=0.2,  # duration in seconds between frames
    ) as gif_writer:
        for img in imgs:
            gif_writer.append_data(imageio.imread(img))
    shutil.rmtree(img_cache_dir, ignore_errors=True)
    
    # close the main process and clean up
    print("Closing main process...")
    close_main_process(rb_writer, dataloader, args.cache_dir)
    
    print("Done!")
    
if __name__ == "__main__":
    main()
    