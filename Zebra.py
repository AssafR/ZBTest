import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt

def random_ints_by_median(low, high, median, no_elements):
    """ Return an array of random integers with range and median """
    lower_half = np.random.random_integers(low,median,size=no_elements//2);
    upper_half = np.random.random_integers(median,high,size=(no_elements -(no_elements // 2)));
    res = np.concatenate((lower_half,upper_half))
    return res

def get_genders_by_distribution(percent_female,total):
    """ Return vector size total with first part size percent containing 'F' the rest 'M' """
    females = int(total * percent_female);
    males = total - females;
    genders = np.concatenate((np.full(females,'F',dtype=str), np.full(males,'M',dtype=str)))
    return genders

def plot(stats1,range,no_bins=1000):
    count, bins, ignored = plt.hist(stats1, range, density=False)
    #stats2 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2))
    #plt.plot(bins, linewidth=2, color='r')
    #plt.plot(bins, color='r')
    plt.show()

def test_median_for_field_value(records, val_field_name, value, numeric_field_name):
    """ Calculate median for numeric field only in rows filtered by value of another field """
    with_value = (records[val_field_name] == value)
    nums_in_field = records[with_value][numeric_field_name]
    return np.median(nums_in_field)

def truncated_normal_distribution(length,mu,sigma,left,right):
    """ Return normal distribution of certain length making sure not to have values outside [left,right] """
    result = np.zeros(length);
    outside_range = np.full(length,True,dtype=bool)
    while(np.sum(outside_range)>0): # Re-draw all elements which fell outside the range
        new_values = np.random.normal(mu,sigma,np.sum(outside_range));
        result[outside_range] = new_values;
        outside_range = [(result[:] < left) | (result[:] > right)];
    return result

def write_record_to_csv(np_array, filename):
    df = pd.DataFrame(np_array)
    df.to_csv(filename,index=False)

def read_record_from_csv(filename):
    df = pd.read_csv(filename,header=0,sep=',')
    return df.to_records(index=False)
    #return genfromtxt(filename, delimiter=',',dtype=None)

def split_male_female(total,female_ratio):
    females = int(female_ratio * total)
    males = total - females
    return females,males

def create_bone_cohort(ids):
    no_patients = ids.size
    females, males = split_male_female(no_patients,female_ratio)
    ages_female = random_ints_by_median(10, 100, 68, females)
    ages_male   = random_ints_by_median(10, 100, 62, males)
    ages = np.concatenate([ages_female, ages_male])
    genders = get_genders_by_distribution(female_ratio, no_patients)
    bone_density = truncated_normal_distribution(no_patients, 0, 1, -3, 3)
    cohort = np.rec.fromarrays([ids, ages, genders, bone_density],
                               names=['id', 'age', 'gender', 'bone_density'])
    return cohort

def create_liver_cohort(ids):
    no_patients = ids.size
    females, males = split_male_female(no_patients,0.5)
    genders = get_genders_by_distribution(0.5, no_patients)
    ages = truncated_normal_distribution(no_patients,40,20,20,90)
    liver =  random_ints_by_median(10, 150, 45, no_patients)
    print(np.median(liver))
    cohort = np.rec.fromarrays([ids,ages,genders,liver],names=['id', 'age', 'gender', 'liver'])
    return cohort

def create_calcium_cohort(no_patients,ids):
    females, males = split_male_female(no_patients,0.5)
    genders = get_genders_by_distribution(0.5, no_patients)
    ages = truncated_normal_distribution(no_patients,50,20,20,90)
    #calcium =


no_patients_1 = 500
no_patients_2 = 500
no_patients_3 = 500


female_ratio = 2/3

total_patients = no_patients_1 + no_patients_2 + no_patients_3;
random_ids = np.random.random_integers(100000,3000000, total_patients)
random_ids_1 = random_ids[0:no_patients_1]
random_ids_2 = random_ids[no_patients_1:no_patients_1+no_patients_2]




cohort_1 = create_bone_cohort(random_ids_1)
cohort_2 = create_liver_cohort(random_ids_2)


write_record_to_csv(cohort_1, "cohort_1.csv")
cohort_1_read = read_record_from_csv("cohort_1.csv")
print(test_median_for_field_value(cohort_1_read,'gender','M','age')) # Male
print(test_median_for_field_value(cohort_1_read,'gender','F','age')) # Female







#plot(random_ids_1,100)






#random_ages = np.round(np.absolute(np.round(np.random.normal(50,10,no_patients) + np.random.uniform(-40,50,no_patients))))




sigma = 7
mu = 68
s = np.random.normal(mu,sigma,5000)
#plot(s,range(20,110))

#s = random_ages
#plot(ages,range(0,110),100)


