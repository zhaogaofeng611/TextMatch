# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 18:06:26 2020

@author: zhaog
"""
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from load_data import load_sentences

class LCQMC_Dataset(Dataset):
    def __init__(self, LCQMC_file):
        self.p, self.h, self.y = load_sentences(LCQMC_file, data_size=None)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.p[idx], self.h[idx], self.y[idx]
    
if __name__ == '__main__':
    ds = LCQMC_Dataset('data/LCQMC_dev.csv')
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    for i, (x,y,z) in enumerate(dl):
        print(z)
        #print(i, data)
        break