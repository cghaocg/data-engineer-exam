import os
import pandas as pd

from data_processing import data_processing

if __name__ == '__main__':
    if not os.path.exists('result'):
        os.mkdir('result')

    data_processing()
