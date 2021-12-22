from typing import Optional
from pandas.core.series import Series
import pandas as pd
import numpy as np


class ImageModel:
    def __init__(self, index: int, data: Series):
        self.file_name: str = data.values[0]
        self.index: int = index
        self.vector_1: float = data.Vector1
        self.vector_2: float = data.Vector2
        self.row: str = data.Row
        self.column: int = data.Column
        self.f: int = data.F
        self.well: str = data.Well
        self.compound: str = data.Compound
        self.concentration: str = str(int(data.Concentration)) if pd.notna(data.Concentration) else str(0)

    def get_channel_image(self, number_of_channel: int):
        channel = f'-ch{str(number_of_channel)}s'
        return self.file_name.replace('-s', channel)

    def __str__(self):
        return f"File name: {self.file_name}," \
               f" Index: {self.index}," \
               f" concentration: {self.concentration}"
