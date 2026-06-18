from torch.utils.data import Dataset

class SineWave_DataSet(Dataset):
    def __init__(self):
        self.sample_generator = data_generating

    def data_generating():
        freq = torch.rand(1) * 10 + 1
        freq = torch.floor(freq)

        coord = torch.linspace(0, 2 * torch.pi,  100)
        y = torch.sin(coord * freq)

        if freq %2 == 0:
            y_a = y[:50]
            y_a[y_a>0] = 1
            y_a[y_a<0] = -1

        else:
            y_a = y[50:]
            y_a[y_a>0] = 1
            y_a[y_a<0] = -1


        return freq, y
    

    def __len__(self):
        return 10000
    
    def __getitem__(self, index):
        return self.sample_generator()
