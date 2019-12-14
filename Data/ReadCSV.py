
import pandas as pd


def readcsv(filename):
    original_csv = pd.read_csv(filename)
    return original_csv

