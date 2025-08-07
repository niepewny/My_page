import h5py
import matplotlib.pyplot as plt
from SEVIR_data_loader import SEVIR_dataset, ConvLSTMSevirDataModule
# from SEVIR import SEVIRDataset
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import numpy as np

#todo:
    # parametryzacja resize
all_file_paths_2019 = [
    "../../data/2019/SEVIR_IR069_RANDOMEVENTS_2019_0101_0430.h5", #val
    "../../data/2019/SEVIR_IR069_RANDOMEVENTS_2019_0501_0831.h5", #test
    "../../data/2019/SEVIR_IR069_RANDOMEVENTS_2019_0901_1231.h5", #train
    "../../data/2019/SEVIR_IR069_STORMEVENTS_2019_0101_0630.h5", # val
    "../../data/2019/SEVIR_IR069_STORMEVENTS_2019_0701_1231.h5"  #test
]
all_file_paths_2018 = [ # train
    "../../data/2018/SEVIR_IR069_RANDOMEVENTS_2018_0101_0430.h5",
    "../../data/2018/SEVIR_IR069_RANDOMEVENTS_2018_0501_0831.h5",
    "../../data/2018/SEVIR_IR069_RANDOMEVENTS_2018_0901_1231.h5",
    "../../data/2018/SEVIR_IR069_STORMEVENTS_2018_0701_1231.h5",
    "../../data/2018/SEVIR_IR069_STORMEVENTS_2018_0101_0630.h5"
]
all_file_paths = all_file_paths_2018 + all_file_paths_2019

#test val train split(each includes at least one storm file)
train_files = [all_file_paths_2018,all_file_paths_2019[2]]

validate_files = [all_file_paths_2019[0],all_file_paths_2019[3]]

test_files = [ all_file_paths_2019[1],all_file_paths_2019[4]]

class SEVIRDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            self.data_length = f['ir069'].shape[0]
            self.ids = f['id'][:]

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            sample = f['ir069'][idx]
            sample = torch.tensor(sample, dtype=torch.float32)
        return sample




def print_data_info(file_path):
    with h5py.File(file_path, 'r') as f:
        # print(f.keys())
        # print(f['ir069'].shape)
        print(f['id'].shape,"\n")
        return f['id'].shape
        # print(f['id'][:])

def print_all_files_info(all_file_paths):
    samples_sum = 0
    for file in all_file_paths:
        print(f"FILE {file}")
        num_samples = print_data_info(file)
        samples_sum += num_samples[0]
    print(f"Total files: {len(all_file_paths)}")
    print(f"Total samples: {samples_sum}")
    print(f"size (X, 192, 192, 49)")


def compare_storm_to_randomevents():
    file_path_randomevents = "../../data/2018/SEVIR_IR069_RANDOMEVENTS_2018_0101_0430.h5"
    file_path_storm = "../../data/2018/SEVIR_IR069_STORMEVENTS_2018_0101_0630.h5"

    visualize_random_sample(file_path_storm)
    visualize_random_sample(file_path_randomevents)

def analyze_data_distribution(dataset, num_batches=100, batch_size=32):
    """
    Analizuje rozkład wartości w datasecie.

    Args:
        dataset: Dataset do przeanalizowania
        num_batches: Ile batchy przeanalizować
        batch_size: Wielkość batcha
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Inicjalizacja list na wartości
    all_mins = []
    all_maxs = []
    all_values = []

    print("Zbieranie statystyk...")

    # Zbieranie wartości z próbek
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        batch_min = batch.min().item()
        batch_max = batch.max().item()

        all_mins.append(batch_min)
        all_maxs.append(batch_max)

        # Zbieranie wszystkich wartości dla histogramu
        all_values.extend(batch.flatten().tolist())

        if i % 10 == 0:
            print(f"Przetworzono {i}/{num_batches} batchy")

    # Obliczanie globalnych statystyk
    global_min = min(all_mins)
    global_max = max(all_maxs)


    # Tworzenie histogramu
    plt.figure(figsize=(12, 6))
    plt.hist(all_values, bins=10, edgecolor='black')
    plt.title('Rozkład wartości w datasecie')
    plt.xlabel('Wartość')
    plt.ylabel('Liczba wystąpień')
    plt.grid(True, alpha=0.3)

    # Dodanie statystyk do wykresu
    stats_text = f'Min: {global_min:.2f}\nMax: {global_max:.2f}'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Obliczanie przedziałów
    bins = np.linspace(global_min, global_max, 11)
    print("\nPrzedziały wartości:")
    for i in range(len(bins)-1):
        print(f"Przedział {i+1}: [{bins[i]:.2f}, {bins[i+1]:.2f})")

    print(f"\nWartość minimalna: {global_min:.2f}")
    print(f"Wartość maksymalna: {global_max:.2f}")

    plt.show()

def calculate_batch_statistics(batch):
    """Calculate statistics for a batch"""
    avg = torch.mean(batch.float()).item()
    min_val = torch.min(batch.float()).item()
    max_val = torch.max(batch.float()).item()
    return avg, min_val, max_val

def print_dataset_statistics(dataloader, print_interval=50):
    """
    Loop through the dataset and print statistics every n batches
    """
    batch_count = 0
    running_avg = 0
    running_min = float('inf')
    running_max = float('-inf')

    print(f"Starting statistics calculation...")

    for batch in dataloader:
        batch_count += 1
        avg, min_val, max_val = calculate_batch_statistics(batch)

        # Update running statistics
        running_avg += avg
        running_min = min(running_min, min_val)
        running_max = max(running_max, max_val)

        if batch_count % print_interval == 0:
            current_avg = running_avg / print_interval
            print(f"\nBatch {batch_count}:")
            print(f"Last {print_interval} batches average: {current_avg:.4f}")
            print(f"Current batch min: {min_val:.4f}")
            print(f"Current batch max: {max_val:.4f}")
            print(f"Running min: {running_min:.4f}")
            print(f"Running max: {running_max:.4f}")
            running_avg = 0  # Reset running average

    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Total number of batches processed: {batch_count}")
    print(f"Overall min: {running_min:.4f}")
    print(f"Overall max: {running_max:.4f}")


if __name__ == "__main__":

    file_path_h5_dir = "../../data/"
    dm = ConvLSTMSevirDataModule(
        step=3,
        width=128,
        height=128,
        batch_size=4,
        num_workers=1,
        sequence_length=15,
        train_files_percent=0.7,
        val_files_percent=0.15,
        test_files_percent=0.15,
        files_dir=file_path_h5_dir
    )

    # Setup and get train loader
    dm.setup('fit')
    train_loader = dm.train_dataloader()

    # Calculate and print statistics
    print_dataset_statistics(train_loader, print_interval=50)
    # analyze_data_distribution(SEVIRDataset(all_file_paths[0]), num_batches=100, batch_size=32)

    # print_all_files_info(all_file_paths)
    # compare_storm_to_randomevents()

    # dane w formacie: shape=(553, 192, 192, 49)
    # 533 próbki, 192x192 pikseli, 49 klatek czasowych co 5 min
    # for file in all_file_paths_2018:
    #     print(file)
    #     sevirDataSet = SEVIRDataset(file)

    #     # przypadkoowy wybór próbki
    #     random = torch.randint(0, 49, (1,)).item()
    #     sample0 = sevirDataSet.__getitem__(random)


    #     print("data length:",evirDataSet.data_length,"\n")
    #     sample0 = SEVIRDataset.__getitem__(0)
    #     print("data shape:",sample0.shape, "\n")
    #     first_frame = sample0[:, :, 0]

    #     # visualize_frame(first_frame)
    #     visualize_tensor_interactive(sample0,f"pierwszy z {file}")
    #     # visualize_tensor_interactive(sample1,f"drugi z {file}")
    #     # visualize_tensor_interactive(sample2,f"trzeci z {file}")
