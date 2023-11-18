import random
import numpy as np

class StockData:
    def __init__(self, data_path, id_path, sequence=20, shuffle=False) -> None:
        self.data_path = data_path
        self.id_path = id_path
        self.sequence = sequence
        self.data = np.loadtxt(self.data_path, dtype=np.float32)
        self.id = np.loadtxt(self.id_path, dtype=np.int32)
        
        self.size = len(self.data) - self.sequence
        self.seq_data = np.zeros((self.size, sequence), dtype=np.float32)
        self.gt_data = np.zeros(self.size, dtype=np.float32)
        
        for i in range(self.size):
            self.seq_data[i] = self.data[i : i + sequence]
            self.gt_data[i] = self.data[i + sequence]
            
        if shuffle:
            indice = list(range(self.size))
            random.shuffle(indice)
            self.seq_data = self.seq_data[indice]
            self.gt_data = self.gt_data[indice]
            # print(indice)

    def __len__(self):
        return self.size
    
    def __getitem__(self, key):
        return self.seq_data[key], self.gt_data[key]



if __name__ == '__main__':
    sd = StockData('test_stock.txt', 'test_label.txt', shuffle=True)
    for i, data in enumerate(sd):
        print(data)
        if i == 4:
            break
    