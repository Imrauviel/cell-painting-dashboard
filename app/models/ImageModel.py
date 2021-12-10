import numpy as np

class ImageModel:
    def __init__(self, file_name):
        self.file_name: str = file_name
        self.index: int = None
        self.image: np.array = None
        self.concentration: str = None
        self.well: str = None

    def get_channel_image(self, number_of_channel: int):
        channel = f'-ch{str(number_of_channel)}s'
        return self.file_name.replace('-s', channel)

    def __str__(self):
        return f"File name: {self.file_name}, index: {self.index}, concentration: {self.concentration}"
