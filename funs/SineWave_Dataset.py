from torch.utils.data import Dataset
import torch

class SineWave_DataSet(Dataset):
    def __init__(self, freq_max = 5):
        self.sample_generator = self.data_generating
        self.freq_max = freq_max
        self.sample_generating_given_freq = self.data_generating_given_freq

    def data_generating(self):
        with torch.no_grad():
            freq = torch.rand(1) * self.freq_max
            freq_int = torch.floor(freq)

            coord = torch.linspace(0, 2 * torch.pi,  200)
            y = torch.sin(coord * freq)

            if freq_int.item() %2 == 1:
                y_a = y[:100]
                y_a[y_a>0] = 1
                y_a[y_a<0] = -1

            else:
                y_a = y[100:]
                y_a[y_a>0] = 1
                y_a[y_a<0] = -1

        return freq, y
    
    def data_generating_given_freq(self, freq_inp):
        with torch.no_grad():
            
            freq_inp_int = torch.floor(freq_inp)

            coord = torch.linspace(0, 2 * torch.pi,  200)
            y = torch.sin(coord * freq_inp)

            if freq_inp_int.item() %2 == 1:
                y_a = y[:100]
                y_a[y_a>0] = 1
                y_a[y_a<0] = -1

            else:
                y_a = y[100:]
                y_a[y_a>0] = 1
                y_a[y_a<0] = -1

        return freq_inp, y.unsqueeze(0)

    def __len__(self):
        return 10000
    
    def __getitem__(self, index):
        return self.sample_generator()
