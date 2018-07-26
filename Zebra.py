import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt

def random_ints_by_median(low, high, median, no_elements):
    """ Return an array of random integers with range and median """
    lower_half = np.random.random_integers(low,median,size=no_elements//2)
    upper_half = np.random.random_integers(median,high,size=(no_elements -(no_elements // 2)))
    res = np.concatenate((lower_half,upper_half))
    return res

def get_genders_by_distribution(percent_female,total):
    """ Return vector size total with first part size percent containing 'F' the rest 'M' """
    females = int(total * percent_female)
    males = total - females
    genders = np.concatenate((np.full(females,'F',dtype=str), np.full(males,'M',dtype=str)))
    return genders

def plot(series, range, no_bins=1000, title=None):
    stats_fix = series.dropna()
    count, bins, ignored = plt.hist(stats_fix, no_bins, density=False)
    plt.title(title)
    #stats2 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2))
    #plt.plot(bins, linewidth=2, color='r')
    #plt.plot(bins, color='r')
    plt.show()

def plot2(series, window_size, title=None):
    min = series.min()
    max = series.max()

    steps = math.ceil((max-min)/window_size)
    bins = pd.Series(range(0,steps));
    bins = window_size * bins + min
    series.hist(bins=bins)
    plt.title(title)
    plt.show()


def test_median_for_field_value(records, val_field_name, value, numeric_field_name):
    """ Calculate median for numeric field only in rows filtered by value of another field """
    with_value = (records[val_field_name] == value)
    nums_in_field = records[with_value][numeric_field_name]
    return np.median(nums_in_field)

def truncated_normal_distribution(length,mu,sigma,left,right):
    """ Return normal distribution of certain length making sure not to have values outside [left,right] """
    result = np.zeros(length)
    outside_range = np.full(length,True,dtype=bool)
    while(np.sum(outside_range)>0): # Re-draw all elements which fell outside the range
        new_values = np.random.normal(mu,sigma,np.sum(outside_range))
        result[outside_range] = new_values
        outside_range = [(result[:] < left) | (result[:] > right)]
    return result

def write_record_to_csv(np_array, filename):
    df = pd.DataFrame(np_array)
    df.to_csv(filename,index=False)

def read_record_from_csv(filename):
    df = pd.read_csv(filename,header=0,sep=',')
    return df # df.to_records(index=False)
    #return genfromtxt(filename, delimiter=',',dtype=None)

def split_male_female(total,female_ratio):
    females = int(female_ratio * total)
    males = total - females
    return females,males

def create_bone_cohort(ids):
    no_patients = ids.size
    female_ratio = 0.5
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
    female_ratio = 0.5
    females, males = split_male_female(no_patients,female_ratio)
    genders = get_genders_by_distribution(female_ratio, no_patients)
    ages = truncated_normal_distribution(no_patients,40,20,20,90)
    liver =  random_ints_by_median(10, 150, 45, no_patients)
    cohort = np.rec.fromarrays([ids,ages,genders,liver],names=['id', 'age', 'gender', 'liver'])
    return cohort

def create_calcium_cohort(ids):
    no_patients = ids.size
    females, males = split_male_female(no_patients,0.5)
    genders = get_genders_by_distribution(0.5, no_patients)
    ages = truncated_normal_distribution(no_patients,50,20,20,90)
    calcium_band_size_1 = no_patients // 2
    calcium_band_size_2 = no_patients // 4
    calcium_band_size_3 = no_patients - calcium_band_size_1 - calcium_band_size_2
    band_1 = truncated_normal_distribution(calcium_band_size_1,300,200,100, 400)
    band_2 = truncated_normal_distribution(calcium_band_size_2,150,200,  0, 100)
    band_3 = truncated_normal_distribution(calcium_band_size_3,600,200,400,1600)
    calcium = np.concatenate([band_1,band_2,band_3])
    cohort = np.rec.fromarrays([ids,ages,genders,calcium],names=['id', 'age', 'gender', 'calcium'])
    return cohort

def data_tests(cohort_1_read,cohort_2_read,cohort_3_read,joint_cohort):
    plot2(cohort_1_read['bone_density'], 0.05, 'Bone Density')
    plot2(cohort_2_read['liver'], 5, 'Liver')
    plot2(cohort_3_read['calcium'], 5, 'Coronary Calcium')
    print(test_median_for_field_value(cohort_1_read,'gender','M','age')) # Male
    print(test_median_for_field_value(cohort_1_read,'gender','F','age')) # Female
    print( np.median(cohort_2_read['liver']))
    print(joint_cohort.to_string())
    plot2(joint_cohort['calcium'], 5, 'Coronary Calcium')


def main():
    no_patients_1 = 500
    no_patients_2 = 400
    no_patients_3 = 500
    total_patients = no_patients_1 + no_patients_2 + no_patients_3

    random_ids = np.random.random_integers(100000,3000000, total_patients)
    random_ids_1 = random_ids[0:no_patients_1]
    random_ids_2 = random_ids[no_patients_1:no_patients_1+no_patients_2]
    random_ids_3 = random_ids[no_patients_1+no_patients_2:no_patients_1+no_patients_2+no_patients_3]

    cohort_1 = create_bone_cohort(random_ids_1)
    cohort_2 = create_liver_cohort(random_ids_2)
    cohort_3 = create_calcium_cohort(random_ids_3)

    write_record_to_csv(cohort_1, "cohort_1.csv")
    cohort_1_read = read_record_from_csv("cohort_1.csv")

    write_record_to_csv(cohort_2, "cohort_2.csv")
    cohort_2_read = read_record_from_csv("cohort_2.csv")

    write_record_to_csv(cohort_3, "cohort_3.csv")
    cohort_3_read = read_record_from_csv("cohort_3.csv")

    # Join the data
    joint_cohort = cohort_1_read.copy(deep=True)
    joint_cohort = joint_cohort.join(cohort_2_read.set_index('id'),lsuffix='_bone',rsuffix='_liver',on='id',how='outer')
    joint_cohort = joint_cohort.join(cohort_3_read.set_index('id'),lsuffix='_liver',rsuffix='_calcium',on='id',how='outer')
    joint_cohort = joint_cohort.rename(index=str, columns={"age": "age_calcium", "gender": "gender_calcium"}) # For compatibility


    series = joint_cohort['calcium']
    data_tests(cohort_1_read, cohort_2_read, cohort_3_read, joint_cohort)
    # window_size = 5
    # bins = range(math.floor(series.min() / window_size),math.ceil(series.max() / window_size),window_size)
    # series.hist(bins=bins)
    # plt.show()


if __name__ == "__main__":
    main()

#plot(random_ids_1,100)
#plot(s,range(20,110))

#s = random_ages
#plot(ages,range(0,110),100)


