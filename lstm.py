import torch
import torch.nn
import numpy as np
from data import StockData
from draw import plot_predictions

class LSTM_Data:
    def __init__(self, data, gt, batch_size):
        data = torch.from_numpy(data)
        gt = torch.from_numpy(gt)
        seq_len = data.size(1)
        
        self.size = len(data) // batch_size
        self.batch_data = torch.zeros((self.size, batch_size, seq_len), dtype=torch.float32)
        self.batch_label = torch.zeros((self.size, batch_size), dtype=torch.float32)
        
        for i in range(self.size):
            self.batch_data[i] = data[i * batch_size : (i + 1) * batch_size]
            self.batch_label[i] = gt[i * batch_size : (i + 1) * batch_size]
        self.batch_data = self.batch_data.unsqueeze(-1)
        self.batch_label = self.batch_label.unsqueeze(-1)
        
    
    def __len__(self,):
        return self.size

    def __getitem__(self, key):
        return self.batch_data[key], self.batch_label[key]
        


class MyLSTM(torch.nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_layer_size, output_size)
        

    def forward(self, input_seq):
        b = input_seq.size(0)
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        # predictions = self.linear(lstm_out.view(len(input_seq), -1))
        x = self.linear(lstm_out.reshape(-1, self.hidden_layer_size))
        
        return x.view(b, -1)[:, -1 : ]
    
def eval(model, test_dataset):
    pred = torch.zeros(len(test_dataset), dtype=torch.float32)
    gt = torch.zeros(len(test_dataset), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for i, (seq, label) in enumerate(test_dataset):
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            pred[i] = model(seq)
            gt[i] = label
    
    plot_predictions(pred, gt, "LSTM")
    
    

if __name__ == "__main__":
    batch_size = 16
    sequence = 20
    epochs = 30
    
    
    train_data = StockData('train_stock.txt', 'train_label.txt', shuffle=True, sequence=sequence)
    test_data = StockData('test_stock.txt', 'test_label.txt', shuffle=False, sequence=sequence)
    
    
    test_dataset = LSTM_Data(test_data.seq_data, test_data.gt_data, 1)
    train_dataset = LSTM_Data(train_data.seq_data, train_data.gt_data, batch_size)
    
    model = MyLSTM()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
    for i in range(epochs):
        for j, (data, label) in enumerate(train_dataset):
            
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, batch_size, model.hidden_layer_size),
                            torch.zeros(1, batch_size, model.hidden_layer_size))

            y_pred = model(data)

            single_loss = loss_function(y_pred, label)
            single_loss.backward()
            optimizer.step()
            
        scheduler.step()
        
        print(f'epoch: [{i:2}] loss: {single_loss.item():10f}')
    
    eval(model, test_dataset)