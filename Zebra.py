import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def random_numbers_by_median(low,high,median,no_elements):
    lower_half = np.random.random_integers(low,median, size=(no_elements // 2,1));
    upper_half = np.random.random_integers(median,high,size=(no_elements -(no_elements // 2),1));
    res = np.vstack([lower_half,upper_half])
    return res

def get_genders_by_distribution(percent_female,total):
    """ Return vector size total with first part size percent containing 'F' the rest 'M' """
    females = int(total * percent_female);
    males = total - females;
    genders = (np.vstack([np.full((females,1),'F',dtype=str), np.full((males,1),'M',dtype=str)])) # First females, then males
    return genders

def write_array_to_csv(np_array,columns,filename):
    df = pd.DataFrame(np_array)
    df.columns = columns
    df.to_csv(filename,index=False)

def plot(stats1,range,no_bins=1000):
    count, bins, ignored = plt.hist(stats1, range, density=False)
    #stats2 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2))
    #plt.plot(bins, linewidth=2, color='r')
    #plt.plot(bins, color='r')
    plt.show()

def test_median_for_field_value(cohort,val_field_num,value,numeric_field_num):
    with_value = (cohort[:, val_field_num] == value)
    nums_in_field = cohort[with_value][:, numeric_field_num]
    return np.median(nums_in_field)

def truncated_normal_distribution(length,mu,sigma,left,right):
    ### Return normal distribution of certain length making sure not to have values outside [left,right] ###
    outside_range = np.full(length,True,dtype=bool)
    result = np.zeros(length);
    while(np.sum(outside_range)>0):
        new_values = np.random.normal(mu,sigma,np.sum(outside_range));
        result[outside_range] = new_values;
        outside_range = [(result[:] < left) | (result[:] > right)];
    result.shape = (length,1)
    return result

no_patients_1 = 500
no_patients_2 = 500
no_patients_3 = 500
female_ratio = 2/3

total_patients = no_patients_1 + no_patients_2 + no_patients_3;
random_ids = np.random.random_integers(100000,3000000, total_patients).reshape(total_patients,1)



females = int(female_ratio * no_patients_1)
males = no_patients_1 - females
ages_female = random_numbers_by_median(10,100,68,females)
ages_male   = random_numbers_by_median(10,100,62,males)
ages = np.vstack([ages_female,ages_male])

genders = get_genders_by_distribution(female_ratio,no_patients_1)
random_ids_1 = random_ids[0:no_patients_1]
bone_density = truncated_normal_distribution(no_patients_1,0,1,-3,3)
#cohort_1 = np.hstack([random_ids_1,ages,genders,bone_density])
cohort_1 = np.column_stack((np.transpose(random_ids_1),np.transpose(genders)))
write_array_to_csv(cohort_1,['id','age','gender','bone_density'],"cohort_1.csv")


print(test_median_for_field_value(cohort_1,2,'M',1)) # Male
print(test_median_for_field_value(cohort_1,2,'F',1)) # Female

#print(ages.size)





#plot(random_ids_1,100)






#random_ages = np.round(np.absolute(np.round(np.random.normal(50,10,no_patients) + np.random.uniform(-40,50,no_patients))))




sigma = 7
mu = 68
s = np.random.normal(mu,sigma,5000)
#plot(s,range(20,110))

#s = random_ages
#plot(ages,range(0,110),100)


