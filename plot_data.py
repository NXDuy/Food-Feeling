import matplotlib.pyplot as plt
from utils.utils import FoodFeeling

# 3 columns of food feeling is 
# start time, end time, number of ingredient

dataset = FoodFeeling()

plt.plot(dataset.input_data[:, 1], dataset.output_data, 'o')
plt.xlabel('Video start time')
plt.ylabel('Video feeling')
plt.show()