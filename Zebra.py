import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

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
    np.random.shuffle(genders)
    return genders


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
    return df

def split_male_female(total,female_ratio):
    females = int(female_ratio * total)
    males = total - females
    return females,males

def create_calcium_distribution(no_patients):
    calcium_band_size_1 = no_patients // 2
    calcium_band_size_2 = no_patients // 4
    calcium_band_size_3 = no_patients - calcium_band_size_1 - calcium_band_size_2
    band_1 = truncated_normal_distribution(calcium_band_size_1, 300, 200, 100, 400)
    band_2 = truncated_normal_distribution(calcium_band_size_2, 150, 200, 0, 100)
    band_3 = truncated_normal_distribution(calcium_band_size_3, 600, 200, 400, 1600)
    calcium = np.concatenate([band_1, band_2, band_3])
    return calcium

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
    calcium = create_calcium_distribution(no_patients)
    cohort = np.rec.fromarrays([ids,ages,genders,calcium],names=['id', 'age', 'gender', 'calcium'])
    return cohort


def plot(df, window_size, title=None):
    series = df[df.notnull()]
    min = series.min()
    max = series.max()

    steps = math.ceil((max-min)/window_size)
    bins = pd.Series(range(0,steps));
    bins = window_size * bins + min
    series.hist(bins=bins)
    plt.title(title)
    plt.show()

def data_tests(cohort_1_read,cohort_2_read,cohort_3_read,joint_cohort):
    plot(cohort_1_read['bone_density'], 0.05, 'Bone Density')
    plot(cohort_2_read['liver'], 5, 'Liver')
    plot(cohort_3_read['calcium'], 20, 'Coronary Calcium')
    print(test_median_for_field_value(cohort_1_read,'gender','M','age')) # Male
    print(test_median_for_field_value(cohort_1_read,'gender','F','age')) # Female
    print(joint_cohort.shape)
    print( np.median(cohort_2_read['liver']))
    print(joint_cohort.to_string())
    plot(joint_cohort['calcium'], 20, 'Coronary Calcium')



def calculate_print_overlap(field1, field2, population):
    overlap, total, ratio = calculate_two_fields_overlap(population, field1, field2)
    print("Fields: [{3},{4}], Total:{0} Overlap:{1} Percent:{2}".format(total, overlap, round(ratio * 100), field1, field2))

def calculate_two_fields_overlap(df, field1, field2):
    overlap = df[~(df[field1].isnull() | df[field2].isnull())]
    return overlap.shape[0], df.shape[0], overlap.shape[0] / df.shape[0]

def main():
    no_patients_1 = 5000
    no_patients_2 = 5000
    no_patients_3 = 5000
    total_patients = no_patients_1 + no_patients_2 + no_patients_3

    random_ids =  pd.Series(random.sample(range(3000000), total_patients))
    random_ids_1 = random_ids[0:no_patients_1]
    random_ids_2 = random_ids[no_patients_1:no_patients_1+no_patients_2]
    random_ids_3 = random_ids[no_patients_1+no_patients_2:no_patients_1+no_patients_2+no_patients_3]

    cohort_1 = create_bone_cohort(random_ids_1)
    cohort_2 = create_liver_cohort(random_ids_2)
    cohort_3 = create_calcium_cohort(random_ids_3)

    write_record_to_csv(cohort_1, "cohort_1.csv")
    write_record_to_csv(cohort_2, "cohort_2.csv")
    write_record_to_csv(cohort_3, "cohort_3.csv")

    cohort_1_read = read_record_from_csv("cohort_1.csv")
    cohort_2_read = read_record_from_csv("cohort_2.csv")
    cohort_3_read = read_record_from_csv("cohort_3.csv")


    # Join the data
    joint_cohort = cohort_1_read.copy(deep=True)
    joint_cohort = joint_cohort.join(cohort_2_read.set_index('id'),lsuffix='_bone',rsuffix='_liver',on='id',how='outer')
    joint_cohort = joint_cohort.join(cohort_3_read.set_index('id'),lsuffix='_liver',rsuffix='_calcium',on='id',how='outer')
    joint_cohort = joint_cohort.rename(index=str, columns={"age": "age_calcium", "gender": "gender_calcium"}) # For compatibility
    write_record_to_csv(joint_cohort, "joint_cohort.csv")


    missing_bone = joint_cohort['bone_density'].isnull()
    to_fill_bone = missing_bone.sample(frac=0.6)
    missing_bone_values = pd.Series(truncated_normal_distribution(to_fill_bone.size, 0, 1, -3, 3))
    #missing_bone_values.reshape(missing_bone_values.size,1)
    #print(to_fill_bone.shape)
    #print(joint_cohort.loc[missing_bone].sample(frac=0.6).to_string())
    #joint_cohort['bone_density'] = joint_cohort['bone_density'].fillna(value=missing_bone_values)
    #write_record_to_csv(joint_cohort, "joint_cohort_.csv")




    added_noise_bone = pd.Series(truncated_normal_distribution(joint_cohort.shape[0], 0, 5, -3, 3))
    #plot(joint_cohort['bone_density'], 0.05, 'Bone Density Before')
    #plot(added_noise_bone, 0.05, 'Bone Density Noise')
    #plot((joint_cohort['liver']/25 -3), 0.05,'Liver Adjusted')
    #liv = (joint_cohort['liver']/25 -3);

    def impute_missing_values_artificially(roww):
        row = roww.copy(deep=True)
 #       print("\n-----\nRow Before=\n",row)
        new_value = None
        if np.isnan(row['bone_density']):
            if (np.random.random() < 0.8): # Fill 80% of the empty bone_density field
                new_value = np.random.random() * 6 -3;
                if row['gender_bone'] == 'F':
                    if not np.isnan(row['liver']):
                        new_value = row['liver']/25 -3
                if not np.isnan(row['calcium']):
                        new_value = (row['calcium'] / 400 - 3)
                row['bone_density'] = new_value

        if np.isnan(row['calcium']):
            if (np.random.random() < 0.8):
                new_value = np.random.random() * 1600;
                if row['age_calcium'] > 50:
                    new_value = row['age_calcium'] * 10
                elif row['age_calcium']> 30:
                    new_value = row['age_calcium'] * 5
                row['calcium'] = new_value
        if np.isnan(row['liver']):
            if (np.random.random() < 0.51):
                new_value = 150* np.random.random()
                row['liver'] = new_value
        return row


    joint_cohort.reset_index()

    #df.loc[df['foo'].isnull(), 'foo'] = df['bar']
    joint_cohort['gender_bone'].fillna(joint_cohort['gender_liver'],inplace=True)
    joint_cohort['gender_bone'].fillna(joint_cohort['gender_calcium'],inplace=True)
    joint_cohort['age_calcium'].fillna(joint_cohort['age_liver'],inplace=True)
    joint_cohort['age_calcium'].fillna(joint_cohort['age_bone'],inplace=True)

    #plot(joint_cohort['calcium'], 10, 'Calcium Before')
    joint_cohort = joint_cohort.apply(impute_missing_values_artificially,axis=1)


    #plot(joint_cohort['bone_density'], 0.1, 'Bone Density After')
    #plot(joint_cohort['calcium'], 10, 'Calcium After')
    print(joint_cohort.to_string())

    calculate_print_overlap('calcium', 'liver', joint_cohort)
    calculate_print_overlap('calcium', 'bone_density', joint_cohort)
    calculate_print_overlap('bone_density', 'liver', joint_cohort)



if __name__ == "__main__":
    main()
