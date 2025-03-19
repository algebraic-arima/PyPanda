from matplotlib import pyplot as plt
import numpy as np

data = np.load(r'../../../DATA/Geant4/data_100w.npy')
hist_top=np.zeros((np.size(data,0),2))
hist_bot=np.zeros((np.size(data,0),2))
for i in range(np.size(data,0)):
    hist_top[i,0]=np.sum(data[i,:56**2])
    hist_top[i,1]=data[i,-1]
    hist_bot[i,0]=np.sum(data[i,56**2:56**2*2])
    hist_bot[i,1]=data[i,-1]
unique_values_top = np.unique(hist_top[:, 1])
unique_values_bot = np.unique(hist_bot[:, 1])
average_values_top = []
average_values_bot = []

for value in unique_values_top:
    mask = hist_top[:, 1] == value
    average = np.mean(hist_top[mask, 0])
    average_values_top.append(average)

for value in unique_values_bot:
    mask = hist_bot[:, 1] == value
    average = np.mean(hist_bot[mask, 0])
    average_values_bot.append(average)


processed_hist_top = np.array([average_values_top, unique_values_top]).T

coefficients = np.polyfit(processed_hist_top[:, 1], processed_hist_top[:, 0], 2)
print(coefficients)
p = np.poly1d(coefficients)

plt.scatter(processed_hist_top[:, 1], processed_hist_top[:, 0], label='Data')

x_fit = np.linspace(processed_hist_top[:, 1].min(), processed_hist_top[:, 1].max(), 100)
y_fit = p(x_fit)
plt.plot(x_fit, y_fit, color='red', label=f'Fit: ${coefficients[0]:.2f}x^2+{coefficients[1]:.2f}x+{coefficients[2]:.2f}$')


processed_hist_bot = np.array([average_values_bot, unique_values_bot]).T

coefficients = np.polyfit(processed_hist_bot[:, 1], processed_hist_bot[:, 0], 2)
print(coefficients)
p = np.poly1d(coefficients)

plt.scatter(processed_hist_bot[:, 1], processed_hist_bot[:, 0], label='Data')

x_fit = np.linspace(processed_hist_bot[:, 1].min(), processed_hist_bot[:, 1].max(), 100)
y_fit = p(x_fit)
plt.plot(x_fit, y_fit, color='red', label=f'Fit: ${coefficients[0]:.2f}x^2+{coefficients[1]:.2f}x+{coefficients[2]:.2f}$')
plt.legend()
plt.savefig('./fit.png')