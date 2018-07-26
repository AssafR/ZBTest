# Hello
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def random_numbers_by_median(low,high,median,no_elements):
    lower_half = np.random.random_integers(low,median, no_elements // 2);
    upper_half = np.random.random_integers(median,high,no_elements -(no_elements // 2));
    return np.hstack([lower_half,upper_half])

def get_sexes_by_distribution(percent_female,total):
    females = int(total * percent_female);
    males = total - females;
    sexes = (np.hstack([np.zeros(females), np.ones(males).astype(int)])).astype(int)
    np.random.shuffle(sexes)
    return sexes

def write_array_to_csv(np_array,filename):
    df = pd.DataFrame(np_array)
    df.to_csv(filename)

def plot(stats1,range,no_bins=1000):
    count, bins, ignored = plt.hist(stats1, range, density=False)
    #stats2 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2))
    #plt.plot(bins, linewidth=2, color='r')
    #plt.plot(bins, color='r')
    plt.show()


no_patients_1 = 500
no_patients_2 = 500
no_patients_3 = 500

random_ids = np.random.random_integers(100000,3000000, no_patients_1+no_patients_2+no_patients_3)
ages = random_numbers_by_median(10,100,68,no_patients_1)
sexes = get_sexes_by_distribution(2/3,no_patients_1)
random_ids_1 = random_ids[0:no_patients_1]

#plot(random_ids_1,100)






#random_ages = np.round(np.absolute(np.round(np.random.normal(50,10,no_patients) + np.random.uniform(-40,50,no_patients))))




sigma = 7
mu = 68
s = np.random.normal(mu,sigma,5000)
plot(s,range(20,110))

#s = random_ages
#plot(ages,range(0,110),100)


