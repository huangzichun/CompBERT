import os
import pandas as pd
from datasets import Dataset


class BIRD(Dataset):
    def __init__(self, dir, arrow_table: Table):
        super().__init__(arrow_table)
        self.label, self.sentence1, self.sentence2 = self.get_data(dir)

    def get_data(self, dir):
        data = pd.read_csv(dir)
        return data["label"].tolist(), data["sentence1"].tolist(), data["sentence2"].tolist()

    def __len__(self):
        return len(self.sentence1)

    def __getitem__(self, idx):
        return self.label[idx], self.sentence1[idx], self.sentence2[idx]