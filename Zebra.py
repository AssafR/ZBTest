# Hello
import random

import numpy as np
import matplotlib.pyplot as plt

no_patients_1 = 500
no_patients_2 = 500
no_patients_3 = 500

random_ids = random.sample(range(100000,100000+3*1000*1000), no_patients_1+no_patients_2+no_patients_3)

def plot(stats1,range,no_bins=1000):
    count, bins, ignored = plt.hist(stats1, range, density=False)
    #stats2 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2))
    #plt.plot(bins, linewidth=2, color='r')
    #plt.plot(bins, color='r')
    plt.show()



#random_ages = np.round(np.absolute(np.round(np.random.normal(50,10,no_patients) + np.random.uniform(-40,50,no_patients))))

patients_1 =


sigma = 7
mu = 68
s = np.random.normal(mu,sigma,5000)
plot(s,range(20,110))

#s = random_ages
