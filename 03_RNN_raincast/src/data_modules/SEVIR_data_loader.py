import h5py
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import pytorch_lightning as pl
from torch.nn import functional as F
import os
import sys
import math
from torchvision import transforms
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#from data_exploration import visualize_batch_tensor_interactive
from src.data_modules.vizualization import visualize_batch_tensor_interactive
import os
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'

TIME_STEPS = 49
DATA_MIN = -6500.0
DATA_MAX = -1000.0

class SEVIR_dataset(Dataset):
    """
    Dataset ładujący dane SEVIR z wielu plików HDF5.
    """
    def __init__(self, samples_dir_path, step, width, height, sequence_length):
        super().__init__()
        # crawl directory to retrive paths to h5 files absolute paths
        self.file_paths = self._get_h5_files(samples_dir_path)
        self.samples_per_file = []
        self._cumulative_indices = []
        self.step = step
        self.width = width
        self.height = height
        self.sequence_length = sequence_length


        current_cum = 0
        for path in self.file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            with h5py.File(path, 'r') as f:
                data_shape = f['ir069'].shape
                x_size = data_shape[0]
                self.samples_per_file.append(x_size)
                current_cum += x_size
                self._cumulative_indices.append(current_cum)

    def __len__(self):
        return self._cumulative_indices[-1]

    def __getitem__(self, index):
        # Znajdujemy indeks pliku, w którym znajduje się żądana próbka
        file_idx = self._find_file_index(index)

        # Obliczamy lokalny indeks w znalezionym pliku:
        # - Dla pierwszego pliku (idx=0) indeks lokalny = indeks globalny
        # - Dla kolejnych plików odejmujemy sumę próbek z poprzednich plików
        if file_idx == 0:
            local_index = index
        else:
            local_index = index - self._cumulative_indices[file_idx - 1]

        file_path = self.file_paths[file_idx]

        '''
        pipeline przetwarzania próbek:
        1. lazy loading na podstawie indeksu lokalnego
        2. zamiana z 192x192x49 na 49x192x192
        3. pobranie co ntej klatki czasowej - np przy kroku 2 zamiana na 25x192x192
        4. zmiana rozmiaru na np. 25 x height x width
        5. normalizacja z zakresu 0-255 na 0-1
        '''
        if self.sequence_length > math.ceil(TIME_STEPS/self.step):
                        raise ValueError(f"sequence_length {self.sequence_length} is greater than available frames {math.ceil(TIME_STEPS/self.step)} (TIME_STEPS/step)")

        try:
            # otwarcie sampla z pliku za pomocą indeksu lokalnego(własciwego dla danego pliku)
            with h5py.File(file_path, 'r') as f:
                sample = f['ir069'][local_index]
                sample = torch.tensor(sample, dtype=torch.float32)
                # zamienia z 192x192x49 na 49x192x192
                permuted_sample = sample.permute(2, 0, 1)
                # przy kroku 2 zamienia na 25x192x192
                permuted_sample_step = self._get_sample_with_step(permuted_sample, self.step)
                # check czy oczekiwana długość jest mniejsza niż wzięcie co ntej klatki, a torch robi ceil przy samplowaniu
                if self.sequence_length <= math.ceil(TIME_STEPS/self.step):
                    permuted_sample_step_len = permuted_sample_step[:self.sequence_length]
                # zmiana rozmiaru na na np. 25 x height x width
                if self.width != 192 or self.height != 192:
                    resize = transforms.Resize((self.height, self.width), antialias=True)
                    permuted_sample_step_resized = resize(permuted_sample_step_len)
                else:
                    permuted_sample_step_resized = permuted_sample_step
                # normalizacja z zakresu 0-255 na 0-1
                permuted_sample_step_resized_normalized = ((permuted_sample_step_resized - DATA_MIN)*2 / (DATA_MAX - DATA_MIN))-1
                permuted_sample_step_resized_normalized_channel = permuted_sample_step_resized_normalized.unsqueeze(1)

                return permuted_sample_step_resized_normalized_channel

        except Exception as e:
            print(f"Error loading file {file_path} at index {local_index}")
            raise e

    def _get_h5_files(self, directory):
        """
        Recursively finds all .h5 files in the given directory and its subdirectories.
        Returns list: A list of absolute paths to .h5 files.
        """
        h5_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.h5'):  # Check for .h5 extension
                    h5_files.append(os.path.abspath(os.path.join(root, file)))
        return h5_files

    def _find_file_index(self, index):
        for i, cum in enumerate(self._cumulative_indices):
            if index < cum:
                return i
        raise IndexError(f"Index {index} out of range {self.__len__()}")

    def _get_sample_with_step(self,tensor,step):
        """
        Zwraca klatki z tensora z krokiem step.
        """
        frames = []
        for i in range(0, tensor.shape[0], step):
            frame = tensor[i]
            frames.append(frame)
        return torch.stack(frames)

class ConvLSTMSevirDataModule(pl.LightningDataModule):
    """
    DataModule dla projektu z convLSTM na zbiorze SEVIR (kanał IR069).
    W metodzie setup 3 dataset-y (train, val, test)
    Domyślnie przypisujemy podział plików test,train,val jak na górze pliku.
    """

    def __init__(
        self,
        # parametry transformacji danych
        step,
        width,
        height,
        sequence_length,
        # przypisanie plików do zbiorów
        files_dir,
        # parametry DataLoadera
        batch_size=4,
        num_workers=2,
        train_files_percent=0.7,
        val_files_percent=0.15,
        test_files_percent=0.15
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.files_dir = files_dir
        self.step = step
        self.width = width
        self.height = height
        self.sequence_length = sequence_length
        self.train_files_percent = train_files_percent
        self.val_files_percent = val_files_percent
        self.test_files_percent = test_files_percent

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None


    def prepare_data(self):
        # zakładamy że pliki są już lokalnie.
        # pobrane komendą z README
        pass

    def setup(self, stage=None):
        # Tworzymy dataset-y.
        full_dataset = SEVIR_dataset(
            self.files_dir,
            self.step,
            self.width,
            self.height,
            self.sequence_length)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset,
            [self.train_files_percent, self.val_files_percent, self.test_files_percent])

        if stage == 'fit' or stage is None:
            self.train_dataset = train_dataset
            self.val_dataset   = val_dataset

        if stage == 'test' or stage is None:
            self.test_dataset  = test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


if __name__ == "__main__":
    ''' przykład użycia oba przykłady są równoważne '''
    # tylko jeden jest na pytorch a drugi na pytorch lighnting

    ''' pytorch dataset '''
    file_path_h5_dir = "../../data/"
    # przyjmuje step oraz szerokość i wysokość obrazka, oraz długość sekwencji(ucinamy 2 klatki tutaj)
    full_dataset = SEVIR_dataset(file_path_h5_dir, 3, 128, 128, 15)
    # podział na zbiory, procentowy
    train_dataset,  val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.7, 0.15, 0.15])
    # przykładowy dataloader dla train datasetu
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    fist_sample = next(iter(dataloader))
    print("data loader",fist_sample.shape) # zwraca torch.Size([10, 17, 128, 128])
    print("sample split data loader",len(train_dataset),len(val_dataset),len(test_dataset),"\n")


    ''' pytorch lightning datamodule '''
    # # przykład użycia
    dm = ConvLSTMSevirDataModule(
        step=2,
        width=192,
        height=192,
        batch_size=4,
        num_workers=1,
        sequence_length=9,
        train_files_percent=0.7,
        val_files_percent=0.15,
        test_files_percent=0.15,
        files_dir=file_path_h5_dir)

    dm.setup('fit')
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader
    dm.setup('test')
    test_loader = dm.test_dataloader
    batch = next(iter(train_loader))
    print("data module",batch.shape) # zwraca torch.Size([4, 17, 128, 128
    print("sample split data loader",len(dm.train_dataset),len(dm.val_dataset),len(dm.test_dataset))
    visualize_batch_tensor_interactive(batch, 0, "SEVIR dataset")
    # visualize_batch_tensor_interactive(batch, 0, "SEVIR dataset")
