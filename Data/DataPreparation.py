import numpy as np

def filterdata(dataset):
    rValue = dataset.copy(deep = True)
    rValue[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','Age']] = dataset[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','Age']].replace(0, np.NaN)
    return rValue.dropna()
