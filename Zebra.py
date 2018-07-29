import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def test_median_for_field_value(records, val_field_name, value, numeric_field_name):
    """ Calculate median for numeric field only in rows filtered by value of another field """
    with_value = (records[val_field_name] == value)
    nums_in_field = records[with_value][numeric_field_name]
    return np.median(nums_in_field)

def truncated_normal_distribution(length,mu,sigma,left,right):
    """ Calculate set of normal distribution random numbers, making sure not to have values outside [left,right] """
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


def calc_split_male_female(total, female_ratio):
    females = int(female_ratio * total)
    males = total - females
    return females, males


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
    female_ratio = 2/3
    females, males = calc_split_male_female(no_patients, female_ratio)
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
    genders = get_genders_by_distribution(female_ratio, no_patients)
    ages = truncated_normal_distribution(no_patients,40,20,20,90)
    liver = random_ints_by_median(10, 150, 45, no_patients)
    cohort = np.rec.fromarrays([ids,ages,genders,liver],names=['id', 'age', 'gender', 'liver'])
    return cohort


def create_calcium_cohort(ids):
    no_patients = ids.size
    female_ratio = 0.5
    genders = get_genders_by_distribution(female_ratio, no_patients)
    ages = truncated_normal_distribution(no_patients,50,20,20,90)
    calcium = create_calcium_distribution(no_patients)
    cohort = np.rec.fromarrays([ids,ages,genders,calcium],names=['id', 'age', 'gender', 'calcium'])
    return cohort


def save_reload_cohorts(cohorts_files_list):
    """ List of pairs (data,filename) - save to filename and then re-read. Return all read values """
    read = []
    for cohort in cohorts_files_list:
        write_record_to_csv(cohort[0], cohort[1])
        read.append(read_record_from_csv(cohort[1]))
    return read


def create_cohorts(random_ids_1, random_ids_2, random_ids_3):
    cohort_1 = pd.DataFrame(create_bone_cohort(random_ids_1))
    cohort_2 = pd.DataFrame(create_liver_cohort(random_ids_2))
    cohort_3 = pd.DataFrame(create_calcium_cohort(random_ids_3))
    return cohort_1, cohort_2, cohort_3


def plot_hist(df, window_size, title=None):
    """ Given a DataFrame and a Window Size, plot a histogram using bins of width window_size """
    series = df[df.notnull()]
    s_min = series.min()
    s_max = series.max()

    steps = math.ceil((s_max-s_min)/window_size)
    bins = pd.Series(range(0, steps));
    bins = window_size * bins + s_min

    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(2000.0 / float(DPI), 960.0 / float(DPI))

    series.hist(bins=bins)
    plt.xticks(bins)
    plt.title(title)
    plt.show()


def data_sanity_tests(cohorts, joint_cohort):
    """ Check important features of the data and print the result """
    for idx, val in enumerate(cohorts):
        if val[0].to_string() == val[1].to_string():
            print("Cohort #",idx+1," read from disk is identical to original")
        else:
            print("ERROR: Cohort #", idx, " read from disk is not identical to original")

    cohort_1_read = cohorts[0][1];
    cohort_2_read = cohorts[1][1];
    cohort_3_read = cohorts[2][1];

    plot_hist(cohort_1_read['bone_density'], 0.1, 'Cohort 1: Bone Density')
    plot_hist(cohort_2_read['liver'], 5, 'Cohort 2: Liver')
    plot_hist(cohort_3_read['calcium'], 50, 'Cohort 3: Coronary Calcium')
    print("Median for age, females cohort 1:",test_median_for_field_value(cohort_1_read,'gender','F','age')) # Female
    print("Median for age,   males cohort 1:",test_median_for_field_value(cohort_1_read,'gender','M','age')) # Male
    print("Median of liver sample in cohort 2:",np.median(cohort_2_read['liver']))
    print("Size of joined cohort:",joint_cohort.shape)
    plot_hist(joint_cohort['calcium'], 50, 'Coronary Calcium After Inserting New Data')
    print_overlap_stats('calcium', 'liver', joint_cohort)
    print_overlap_stats('calcium', 'bone_density', joint_cohort)
    print_overlap_stats('bone_density', 'liver', joint_cohort)


def print_overlap_stats(field1, field2, population):
    overlap, total, ratio = calculate_two_fields_overlap(population, field1, field2)
    print("Fields: [{3},{4}], Total:{0} Overlap:{1} Percent:{2}".format(total, overlap, round(ratio * 100), field1, field2))


def calculate_two_fields_overlap(df, field1, field2):
    overlap = df[~(df[field1].isnull() | df[field2].isnull())]
    return overlap.shape[0], df.shape[0], overlap.shape[0] / df.shape[0]


def impute_missing_values_artificially(row):
    if np.isnan(row['bone_density']):
        if (np.random.random() < 0.8):  # Fill 80% of the empty bone_density field
            new_value = np.random.random() * 6 - 3;
            if row['gender_bone'] == 'F':
                if not np.isnan(row['liver']):
                    new_value = row['liver'] / 25 - 3
            if not np.isnan(row['calcium']):
                new_value = (row['calcium'] / 400 - 3)
            row['bone_density'] = new_value
    if np.isnan(row['calcium']):
        if np.random.random() < 0.8:
            new_value = np.random.random() * 1600;
            if row['age_calcium'] > 50:
                new_value = row['age_calcium'] * 10
            elif row['age_calcium'] > 30:
                new_value = row['age_calcium'] * 5
            row['calcium'] = new_value
    if np.isnan(row['liver']):
        if np.random.random() < 0.51:
            new_value = 150 * np.random.random()
            row['liver'] = new_value
    return row


def main():
    no_patients_1 = 1500
    no_patients_2 = 2000
    no_patients_3 = 1500
    total_patients = no_patients_1 + no_patients_2 + no_patients_3

    random_ids = 100000 + pd.Series(random.sample(range(3000000), total_patients))
    random_ids_1 = random_ids[0:no_patients_1]
    random_ids_2 = random_ids[no_patients_1:no_patients_1+no_patients_2]
    random_ids_3 = random_ids[no_patients_1+no_patients_2:no_patients_1+no_patients_2+no_patients_3]

    cohort_1, cohort_2, cohort_3 = create_cohorts(random_ids_1, random_ids_2, random_ids_3)

    read_cohorts = \
        save_reload_cohorts([(cohort_1, "cohort_1.csv"), (cohort_2, "cohort_2.csv"), (cohort_3, "cohort_3.csv")])
    cohort_1_read, cohort_2_read, cohort_3_read = read_cohorts[0], read_cohorts[1], read_cohorts[2]

    # Join the data
    joint_cohort = cohort_1_read.copy(deep=True)
    joint_cohort = joint_cohort.join(cohort_2_read.set_index('id'),lsuffix='_bone',rsuffix='_liver',on='id',how='outer')
    joint_cohort = joint_cohort.join(cohort_3_read.set_index('id'),lsuffix='_liver',rsuffix='_calcium',on='id',how='outer')
    joint_cohort = joint_cohort.rename(index=str, columns={"age": "age_calcium", "gender": "gender_calcium"}) # For compatibility
    print("Joined cohort fields are:", joint_cohort.keys())
    write_record_to_csv(joint_cohort, "joint_cohort.csv")

    imputed_joint_cohort = read_record_from_csv("joint_cohort.csv")

    # Copy None fields from corresponding column
    imputed_joint_cohort['gender_bone'].fillna(imputed_joint_cohort['gender_liver'],inplace=True)
    imputed_joint_cohort['gender_bone'].fillna(imputed_joint_cohort['gender_calcium'],inplace=True)
    imputed_joint_cohort['age_calcium'].fillna(imputed_joint_cohort['age_liver'],inplace=True)
    imputed_joint_cohort['age_calcium'].fillna(imputed_joint_cohort['age_bone'],inplace=True)

    imputed_joint_cohort = joint_cohort.apply(impute_missing_values_artificially, axis=1)
    write_record_to_csv(imputed_joint_cohort, "joint_cohort_imputed.csv")

    cohort_pairs = ((cohort_1,cohort_1_read),(cohort_2,cohort_2_read),(cohort_3,cohort_3_read),)
    data_sanity_tests(cohort_pairs, imputed_joint_cohort)

if __name__ == "__main__":
    main()

