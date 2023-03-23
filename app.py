import uvicorn
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import random
from statsmodels.stats.power import TTestIndPower
from collections import Counter
from fastapi import FastAPI
from simulation import simulation
from clinical import clinical
#from power_analysis import power_analysis
from fastapi.responses import StreamingResponse,HTMLResponse
from fastapi import FastAPI, UploadFile,File
from pydantic import BaseModel
import io 
import requests
#from flask import Flask, request
app = FastAPI()


@app.get('/')
def index():
    return {'message': 'Hello, World'}

power_analysis=TTestIndPower()
def calculate_power1(x, y, df):
    power_analysis=TTestIndPower()
    treatment_arr=[]
    control_arr=[]
    treatment_locs=np.where((df['smoke_inter']==x))
    control_locs=np.where((df['smoke_inter']==y))
    for i in treatment_locs:
        treatment_arr.append(df['treatment_smoke'].iloc[i])
    for j in control_locs:
        control_arr.append(df['treatment_smoke'].iloc[j])
    l1 = len(treatment_arr[0])
    l2 = len(control_arr[0])
    index_treatment=np.arange(0,l1)
    index_control=np.arange(0,l2)
    treatment_df=pd.DataFrame({'idx':index_treatment,"Treatment":treatment_arr[0]})
    control_df=pd.DataFrame({'idx':index_control,"Control":control_arr[0]}) 
    mu1=treatment_df['Treatment'].mean()
    mu2=control_df['Control'].mean()
    std1=treatment_df['Treatment'].std()
    std2=control_df['Control'].std()
    s = np.sqrt(((l1 - 1) * std1 + (l2 - 1) * std2) / (l1 + l2 - 2))
    d = (mu1 - mu2) / s #cohen's effect size
    eff = round(d,2)
    p = power_analysis.power(effect_size=eff,alpha=0.05,nobs1=l1,ratio=(l1/l2),alternative='two-sided')
    return p
def calculate_power2(x, y, df):
    power_analysis=TTestIndPower()
    treatment_arr=[]
    control_arr=[]
    treatment_locs=np.where((df['alc_inter']==x))
    control_locs=np.where((df['alc_inter']==y))
    for i in treatment_locs:
        treatment_arr.append(df['treatment_alc'].iloc[i])
    for j in control_locs:
        control_arr.append(df['treatment_alc'].iloc[j])
    l1 = len(treatment_arr[0])
    l2 = len(control_arr[0])
    index_treatment=np.arange(0,l1)
    index_control=np.arange(0,l2)
    treatment_df=pd.DataFrame({'idx':index_treatment,"Treatment":treatment_arr[0]})
    control_df=pd.DataFrame({'idx':index_control,"Control":control_arr[0]}) 
    mu1=treatment_df['Treatment'].mean()
    mu2=control_df['Control'].mean()
    std1=treatment_df['Treatment'].std()
    std2=control_df['Control'].std()
    s = np.sqrt(((l1 - 1) * std1 + (l2 - 1) * std2) / (l1 + l2 - 2))
    d = (mu1 - mu2) / s #cohen's effect size
    eff = round(d,2)
    p = power_analysis.power(effect_size=eff,alpha=0.05,nobs1=l1,ratio=(l1/l2),alternative='two-sided')
    return p

def calculate_power3(x, y, df):
    power_analysis=TTestIndPower()
    treatment_arr=[]
    control_arr=[]
    treatment_locs=np.where((df['mh_inter']==x))
    control_locs=np.where((df['mh_inter']==y))
    for i in treatment_locs:
        treatment_arr.append(df['treatment_mh'].iloc[i])
    for j in control_locs:
        control_arr.append(df['treatment_mh'].iloc[j])
    l1 = len(treatment_arr[0])
    l2 = len(control_arr[0])
    index_treatment=np.arange(0,l1)
    index_control=np.arange(0,l2)
    treatment_df=pd.DataFrame({'idx':index_treatment,"Treatment":treatment_arr[0]})
    control_df=pd.DataFrame({'idx':index_control,"Control":control_arr[0]}) 
    mu1=treatment_df['Treatment'].mean()
    mu2=control_df['Control'].mean()
    std1=treatment_df['Treatment'].std()
    std2=control_df['Control'].std()
    s = np.sqrt(((l1 - 1) * std1 + (l2 - 1) * std2) / (l1 + l2 - 2))
    d = (mu1 - mu2) / s #cohen's effect size
    eff = round(d,2)
    p = power_analysis.power(effect_size=eff,alpha=0.05,nobs1=l1,ratio=(l1/l2),alternative='two-sided')
    return p

def try_sample_size(df,prob_2_2_and_1_1,prob_3_3,prob_2_1,prob_3_2,prob_3_1,prob_1_0,prob_2_0,prob_3_0,male,
    female,
    low_bmi,
    normal_bmi,
    high_bmi,
    l1_edu,
    l2_edu,
    l3_edu,
    l4_edu,
    min_age,
    max_age,
    mean_age,
    sd_age, 
    mean_ttd,std_ttd): 
    sample_sizes = {}
    sample_sizes['psmoke'] = calculate_power1(2, 1, df) #calculate power function
    sample_sizes['palc'] = calculate_power2(2, 1, df)
    sample_sizes['pmh'] = calculate_power3(2, 1, df)

    size1 = round(df.shape[0]/26)
    all_above_threshold = all(value >= 0.80 for value in sample_sizes.values())
    if not all_above_threshold:
        if sample_sizes['psmoke'] < 0.80:
            size1 += 10
        elif sample_sizes['palc'] < 0.80:
            size1 += 10
        elif sample_sizes['pmh'] < 0.80:
            size1 += 10

        df = create_clinical_dataset1(prob_2_2_and_1_1,prob_3_3,prob_2_1,prob_3_2,prob_3_1,prob_1_0,prob_2_0,prob_3_0,male,
                        female,
                        low_bmi,
                        normal_bmi,
                        high_bmi,
                        l1_edu,
                        l2_edu,
                        l3_edu,
                        l4_edu,
                        min_age,
                        max_age,
                        mean_age,
                        sd_age, 
                        mean_ttd,
                        std_ttd,size=size1)   #create dataset function
        print(sample_sizes.values()) #check sample sizes
        return try_sample_size(df,prob_2_2_and_1_1,prob_3_3,prob_2_1,prob_3_2,prob_3_1,prob_1_0,prob_2_0,prob_3_0,male,female,
    low_bmi,
    normal_bmi,
    high_bmi,
    l1_edu,
    l2_edu,
    l3_edu,
    l4_edu,
    min_age,
    max_age,
    mean_age,
    sd_age, 
    mean_ttd,std_ttd)

    return df

def create_clinical_dataset1(prob_2_2_and_1_1: float,
    prob_3_3: float,
    prob_2_1: float,
    prob_3_2: float,
    prob_3_1: float,
    prob_1_0: float,
    prob_2_0: float,
    prob_3_0: float,
    male: float,
    female : float,
    low_bmi:float,
    normal_bmi: float,
    high_bmi: float,
    l1_edu: float,
    l2_edu: float,
    l3_edu: float,
    l4_edu: float,
    min_age: int,
    max_age: int,
    mean_age: int,
    sd_age: int ,
    mean_ttd : float,
    std_ttd : float,
    size: int):
    #np.random.seed(200)
    # data= data.dict()
    # prob_2_2_and_1_1 = data['prob_2_2_and_1_1']
    # prob_3_3 = data['prob_3_3']
    # prob_2_1 = data['prob_2_1']
    # prob_3_2 = data['prob_3_2']
    # prob_3_1 = data['prob_3_1']
    # prob_1_0 = data['prob_1_0']
    # prob_2_0 = data['prob_2_0']
    # prob_3_0 = data['prob_3_0']
    # male= data['male']
    # female=data['female'] 
    # low_bmi=data['low_bmi']
    # normal_bmi=data['normal_bmi'] 
    # high_bmi=data['high_bmi'] 
    # l1_edu=data['l1_edu']
    # l2_edu=data['l2_edu']
    # l3_edu=data['l3_edu']
    # l4_edu=data['l4_edu']
    # min_age=data['min_age']
    # max_age=data['max_age']
    # mean_age=data['mean_age']
    # sd_age=data['sd_age']
    # mean_ttd = data['mean_ttd']
    # std_ttd = data['std_ttd']
    # size = data['size']
    gender=list()
    bmi=list()
    age=list()
    edu=list()
    ttd = list()
    gender.extend([male,female])
    bmi.extend([low_bmi,normal_bmi,high_bmi])
    edu.extend([l1_edu,l2_edu,l3_edu,l4_edu])
    age.extend([min_age,max_age,mean_age,sd_age])
    ttd.extend([mean_ttd,std_ttd])
    age_1 = np.random.normal(loc=age[2], scale=age[3], size=size*27)
    age_1 = np.clip(age_1, a_min=age[0], a_max=age[1])
    age_1 = np.round(age_1).astype(int)
    sex = np.random.choice(2,size*27,p=[gender[0],gender[1]])
    body_mass= np.random.choice(3,size*27,p=[bmi[0],bmi[1],bmi[2]])
    education= np.random.choice(4,size*27,p=[edu[0],edu[1],edu[2],edu[3]])
    cavitation = np.random.choice(2,size*27,p=[0.5,0.5])
    ttd1 = np.random.normal(loc=ttd[0], scale=ttd[1],size=size*27)
    alc_int = np.array([])
    mh_int = np.array([])
    smoking_int = np.array([])
    values = [0, 1, 2] #0 NA 1:NO INTER 2:Intervention
    combinations = np.array(np.meshgrid(*([values] * 3))).T.reshape(-1, 3)
    for value in values:
        if not any((combinations == value).all(axis=1)):
            # If a value doesn't appear, add it to a random combination
            i = np.random.randint(len(combinations))
            j = np.random.randint(3)
            combinations[i, j] = value

    #combinations = np.delete(combinations, 0, axis=0)
    for i in range(len(combinations)):
            alc_int = np.append(alc_int, np.tile(combinations[i, 0], size))
            mh_int = np.append(mh_int, np.tile(combinations[i, 1], size))
            smoking_int = np.append(smoking_int, np.tile(combinations[i, 2], size))
            alc_int = alc_int.astype(int)
            mh_int = mh_int.astype(int)
            smoking_int = smoking_int.astype(int)
    values_array = np.column_stack(((alc_int, mh_int, smoking_int)))
    # filter out rows where all values are zero
    values_array = values_array[~np.all(values_array == 0, axis=1)]
    tuples_list = list(map(tuple, values_array))
    print(tuples_list)
    combinations_dict = {    
                            (0, 1, 0):prob_1_0,
                            (0, 2, 0):prob_2_2_and_1_1,
                            (1, 0, 0):prob_1_0, 
                        (1, 1, 0):prob_2_0, 
                        (1, 2, 0):prob_2_1, 
                        (2, 0, 0):prob_2_2_and_1_1, 
                        (2, 1, 0):prob_2_1, 
                        (2, 2, 0):prob_2_2_and_1_1, 
                        (0, 0, 1):prob_1_0, 
                        (0, 1, 1):prob_2_0,
                        (0, 2, 1):prob_2_1,
                        (1, 0, 1):prob_2_0 ,
                        (1, 1, 1):prob_3_0,
                        (1, 2, 1):prob_3_1 ,
                        (2, 0, 1):prob_2_1 ,
                        (2, 1, 1):prob_3_1,
                        (2, 2, 1):prob_3_2 ,
                        (0, 0, 2):prob_2_2_and_1_1, 
                        (0, 1, 2):prob_2_1 ,
                        (0, 2, 2):prob_2_2_and_1_1, 
                        (1, 0, 2):prob_2_1, 
                        (1, 1, 2):prob_3_1,
                        (1, 2, 2):prob_3_2 ,
                        (2, 0, 2):prob_2_2_and_1_1, 
                        (2, 1, 2):prob_3_2 ,
                        (2, 2, 2):prob_3_3,
                        }
    prob = list()
    #prob_random = list()
    treatment_outcomes = np.empty(len(tuples_list))
    for i in range(len(tuples_list)):
        prob.append(combinations_dict.get(tuples_list[i]))
        #prob_random.append(prob[i]+round(np.random.uniform(-0.02,0.02),4))
        #print(Counter(prob_random))
        #probability = prob_random[i]
        probability = prob[i]
        if np.isnan(probability):
            treatment_outcomes[i] = np.nan
        # otherwise, generate a random outcome based on the probability using np.random.choice
        else:
            if np.random.rand() < probability:
                treatment_outcomes[i] = 1
            else:
                treatment_outcomes[i] = 0

    treatment_outcomes = treatment_outcomes.astype(int)
    df = pd.DataFrame({'alc_inter': alc_int, 'mh_inter': mh_int, 'smoke_inter': smoking_int,'age':age_1,'sex':sex,'bmi':body_mass,'education':education,'cavitation':cavitation,'ttd':ttd1})
    df = df[~((df['alc_inter'] == 0) & (df['mh_inter'] == 0) & (df['smoke_inter'] == 0))]
    df['treatment_outcomes'] = treatment_outcomes
    df['alcohol'] = df['alc_inter'].apply(lambda x: 1 if x == 1 or x == 2 else 0)
    df['mental_health'] = df['mh_inter'].apply(lambda x: 1 if x == 1 or x == 2 else 0)
    df['smoking'] = df['smoke_inter'].apply(lambda x: 1 if x == 1 or x == 2 else 0)
    df = df.sample(frac=1).reset_index(drop=True)
    case1_1 = df.loc[(df['smoke_inter'] == 2) & (df['alc_inter'] == 2) & (df['mh_inter'] == 2), 'treatment_outcomes'].value_counts(normalize=True)
    case1_1 = case1_1.get(1, 0)
    case1_2 = df.loc[(df['smoke_inter'] == 2) & (df['alc_inter'] == 2) & (df['mh_inter'] == 1), 'treatment_outcomes'].value_counts(normalize=True)
    case1_2 = case1_2.get(1, 0)
    case1_3 = df.loc[(df['smoke_inter'] == 2) & (df['alc_inter'] == 1) & (df['mh_inter'] == 2), 'treatment_outcomes'].value_counts(normalize=True)
    case1_3 = case1_3.get(1, 0)
    case1_4 = df.loc[(df['smoke_inter'] == 1) & (df['alc_inter'] == 2) & (df['mh_inter'] == 2), 'treatment_outcomes'].value_counts(normalize=True)
    case1_4 = case1_4.get(1, 0)
    case1_5 = df.loc[(df['smoke_inter'] == 2) & (df['alc_inter'] == 1) & (df['mh_inter'] == 1), 'treatment_outcomes'].value_counts(normalize=True)
    case1_5 = case1_5.get(1, 0)
    case1_6 = df.loc[(df['smoke_inter'] == 1) & (df['alc_inter'] == 2) & (df['mh_inter'] == 1), 'treatment_outcomes'].value_counts(normalize=True)
    case1_6 = case1_6.get(1, 0)
    case1_7 = df.loc[(df['smoke_inter'] == 1) & (df['alc_inter'] == 1) & (df['mh_inter'] == 2), 'treatment_outcomes'].value_counts(normalize=True)
    case1_7 = case1_7.get(1, 0)
    case1_8 = df.loc[(df['smoke_inter'] == 1) & (df['alc_inter'] == 1) & (df['mh_inter'] == 1), 'treatment_outcomes'].value_counts(normalize=True)
    case1_8 = case1_8.get(1, 0)

    case2_1 = df.loc[(df['smoke_inter'] == 2) & (df['alc_inter'] == 2) & (df['mh_inter'] == 0), 'treatment_outcomes'].value_counts(normalize=True)
    case2_1 = case2_1.get(1, 0)
    case2_2 = df.loc[(df['smoke_inter'] == 2) & (df['alc_inter'] == 1) & (df['mh_inter'] == 0), 'treatment_outcomes'].value_counts(normalize=True)
    case2_2 = case2_2.get(1, 0)
    case2_3 = df.loc[(df['smoke_inter'] == 1) & (df['alc_inter'] == 2) & (df['mh_inter'] == 0), 'treatment_outcomes'].value_counts(normalize=True)
    case2_3 = case2_3.get(1, 0)
    case2_4 = df.loc[(df['smoke_inter'] == 1) & (df['alc_inter'] == 1) & (df['mh_inter'] == 0), 'treatment_outcomes'].value_counts(normalize=True)
    case2_4 = case2_4.get(1, 0)

    case3_1 = df.loc[(df['smoke_inter'] == 2) & (df['alc_inter'] == 0) & (df['mh_inter'] == 2), 'treatment_outcomes'].value_counts(normalize=True)
    case3_1 = case3_1.get(1, 0)
    case3_2 = df.loc[(df['smoke_inter'] == 2) & (df['alc_inter'] == 0) & (df['mh_inter'] == 1), 'treatment_outcomes'].value_counts(normalize=True)
    case3_2 = case3_2.get(1, 0)
    case3_3 = df.loc[(df['smoke_inter'] == 1) & (df['alc_inter'] == 0) & (df['mh_inter'] == 2), 'treatment_outcomes'].value_counts(normalize=True)
    case3_3 = case3_3.get(1, 0)
    case3_4 = df.loc[(df['smoke_inter'] == 1) & (df['alc_inter'] == 0) & (df['mh_inter'] == 1), 'treatment_outcomes'].value_counts(normalize=True)
    case3_4 = case3_4.get(1, 0)

    case4_1 = df.loc[(df['smoke_inter'] == 0) & (df['alc_inter'] == 2) & (df['mh_inter'] == 2), 'treatment_outcomes'].value_counts(normalize=True)
    case4_1 = case4_1.get(1, 0)
    case4_2 = df.loc[(df['smoke_inter'] == 0) & (df['alc_inter'] == 2) & (df['mh_inter'] == 1), 'treatment_outcomes'].value_counts(normalize=True)
    case4_2 = case4_2.get(1, 0)
    case4_3 = df.loc[(df['smoke_inter'] == 0) & (df['alc_inter'] == 1) & (df['mh_inter'] == 2), 'treatment_outcomes'].value_counts(normalize=True)
    case4_3 = case4_3.get(1, 0)
    case4_4 = df.loc[(df['smoke_inter'] == 0) & (df['alc_inter'] == 1) & (df['mh_inter'] == 1), 'treatment_outcomes'].value_counts(normalize=True)
    case4_4 = case4_4.get(1, 0)

    case5_1 = df.loc[(df['smoke_inter'] == 2) & (df['alc_inter'] == 0) & (df['mh_inter'] == 0), 'treatment_outcomes'].value_counts(normalize=True)
    case5_1 = case5_1.get(1, 0)
    case5_2 = df.loc[(df['smoke_inter'] == 1) & (df['alc_inter'] == 0) & (df['mh_inter'] == 0), 'treatment_outcomes'].value_counts(normalize=True)
    case5_2 = case5_2.get(1, 0)

    case6_1 = df.loc[(df['smoke_inter'] == 0) & (df['alc_inter'] == 0) & (df['mh_inter'] == 2), 'treatment_outcomes'].value_counts(normalize=True)
    case6_1 = case6_1.get(1, 0)
    case6_2 = df.loc[(df['smoke_inter'] == 0) & (df['alc_inter'] == 0) & (df['mh_inter'] == 1), 'treatment_outcomes'].value_counts(normalize=True)
    case6_2 = case6_2.get(1, 0)

    case7_1 = df.loc[(df['smoke_inter'] == 0) & (df['alc_inter'] == 2) & (df['mh_inter'] == 0), 'treatment_outcomes'].value_counts(normalize=True)
    case7_1 = case7_1.get(1, 0)
    case7_2 = df.loc[(df['smoke_inter'] == 0) & (df['alc_inter'] == 1) & (df['mh_inter'] == 0), 'treatment_outcomes'].value_counts(normalize=True)
    case7_2 = case7_2.get(1, 0)

    power_intervention_smoke = case1_1*(1/8) + case1_2*(1/8) + case1_3*(1/8) + case1_5*(1/8) + case2_1*(1/4) + case2_2*(1/4) + case3_1*(1/4) + case3_2*(1/4) + case5_1*(1/2)
    power_intervention_alc = case1_1*(1/8) + case1_2*(1/8) + case1_4*(1/8) + case1_6*(1/8) + case2_1*(1/4) + case2_3*(1/4) + case4_1*(1/4) + case4_2*(1/4) + case7_1*(1/2)
    power_intervention_mh = case1_1*(1/8) + case1_3*(1/8) + case1_4*(1/8) + case1_7*(1/8) + case3_1*(1/4) + case3_3*(1/4) + case4_1*(1/4) + case4_3*(1/4) + case6_1*(1/2)


    power_control_smoke = case1_4*(1/8) + case1_6*(1/8) + case1_7*(1/8) + case1_8*(1/8) + case2_3*(1/4) + case2_4*(1/4) + case3_3*(1/4) + case3_4*(1/4) + case5_2*(1/2)
    power_control_alc = case1_3*(1/8) + case1_5*(1/8) + case1_7*(1/8) + case1_8*(1/8) + case2_2*(1/4) + case2_4*(1/4) + case4_3*(1/4) + case4_4*(1/4) + case7_2*(1/2)
    power_control_mh = case1_2*(1/8) + case1_5*(1/8) + case1_6*(1/8) + case1_8*(1/8) + case3_2*(1/4) + case3_4*(1/4) + case4_2*(1/4) + case4_4*(1/4) + case6_2*(1/2)
    
    my_list = [power_intervention_smoke, power_intervention_alc, power_intervention_mh, power_control_smoke, power_control_alc, power_control_mh]
    print(my_list)
    
    df['treatment_smoke'] = df['smoke_inter'].apply(lambda x: np.random.choice([2, 1], p=[power_intervention_smoke, 1- power_intervention_smoke]) if x == 2 
                                                    else np.random.choice([2, 1], p=[power_control_smoke, 1-power_control_smoke]) if x == 1 else 0)
    
    df['treatment_alc'] = df['alc_inter'].apply(lambda x: np.random.choice([2, 1], p=[power_intervention_alc, 1 - power_intervention_alc]) if x == 2 
                                                    else np.random.choice([2, 1], p=[power_control_alc, 1-power_control_alc]) if x == 1 else 0)
    
    df['treatment_mh'] = df['mh_inter'].apply(lambda x: np.random.choice([2, 1], p=[power_intervention_mh, 1-power_intervention_mh]) if x == 2 
                                                    else np.random.choice([2, 1], p=[power_control_mh, 1-power_control_mh]) if x == 1 else 0)
    #dfx = try_sample_size(df)
    return df

@app.post('/simulate_clinical_dataset')
def create_clinical_dataset(prob_2_2_and_1_1: float,
    prob_3_3: float,
    prob_2_1: float,
    prob_3_2: float,
    prob_3_1: float,
    prob_1_0: float,
    prob_2_0: float,
    prob_3_0: float,
    male: float,
    female : float,
    low_bmi:float,
    normal_bmi: float,
    high_bmi: float,
    l1_edu: float,
    l2_edu: float,
    l3_edu: float,
    l4_edu: float,
    min_age: int,
    max_age: int,
    mean_age: int,
    sd_age: int ,
    mean_ttd : float,
    std_ttd : float,
    size: int):
    #np.random.seed(200)
    # data= data.dict()
    # prob_2_2_and_1_1 = data['prob_2_2_and_1_1']
    # prob_3_3 = data['prob_3_3']
    # prob_2_1 = data['prob_2_1']
    # prob_3_2 = data['prob_3_2']
    # prob_3_1 = data['prob_3_1']
    # prob_1_0 = data['prob_1_0']
    # prob_2_0 = data['prob_2_0']
    # prob_3_0 = data['prob_3_0']
    # male= data['male']
    # female=data['female'] 
    # low_bmi=data['low_bmi']
    # normal_bmi=data['normal_bmi'] 
    # high_bmi=data['high_bmi'] 
    # l1_edu=data['l1_edu']
    # l2_edu=data['l2_edu']
    # l3_edu=data['l3_edu']
    # l4_edu=data['l4_edu']
    # min_age=data['min_age']
    # max_age=data['max_age']
    # mean_age=data['mean_age']
    # sd_age=data['sd_age']
    # mean_ttd = data['mean_ttd']
    # std_ttd = data['std_ttd']
    # size = data['size']
    gender=list()
    bmi=list()
    age=list()
    edu=list()
    ttd = list()
    gender.extend([male,female])
    bmi.extend([low_bmi,normal_bmi,high_bmi])
    edu.extend([l1_edu,l2_edu,l3_edu,l4_edu])
    age.extend([min_age,max_age,mean_age,sd_age])
    ttd.extend([mean_ttd,std_ttd])
    age_1 = np.random.normal(loc=age[2], scale=age[3], size=size*27)
    age_1 = np.clip(age_1, a_min=age[0], a_max=age[1])
    age_1 = np.round(age_1).astype(int)
    sex = np.random.choice(2,size*27,p=[gender[0],gender[1]])
    body_mass= np.random.choice(3,size*27,p=[bmi[0],bmi[1],bmi[2]])
    education= np.random.choice(4,size*27,p=[edu[0],edu[1],edu[2],edu[3]])
    cavitation = np.random.choice(2,size*27,p=[0.5,0.5])
    ttd1 = np.random.normal(loc=ttd[0], scale=ttd[1],size=size*27)
    alc_int = np.array([])
    mh_int = np.array([])
    smoking_int = np.array([])
    values = [0, 1, 2] #0 NA 1:NO INTER 2:Intervention
    combinations = np.array(np.meshgrid(*([values] * 3))).T.reshape(-1, 3)
    for value in values:
        if not any((combinations == value).all(axis=1)):
            # If a value doesn't appear, add it to a random combination
            i = np.random.randint(len(combinations))
            j = np.random.randint(3)
            combinations[i, j] = value

    #combinations = np.delete(combinations, 0, axis=0)
    for i in range(len(combinations)):
            alc_int = np.append(alc_int, np.tile(combinations[i, 0], size))
            mh_int = np.append(mh_int, np.tile(combinations[i, 1], size))
            smoking_int = np.append(smoking_int, np.tile(combinations[i, 2], size))
            alc_int = alc_int.astype(int)
            mh_int = mh_int.astype(int)
            smoking_int = smoking_int.astype(int)
    values_array = np.column_stack(((alc_int, mh_int, smoking_int)))
    # filter out rows where all values are zero
    values_array = values_array[~np.all(values_array == 0, axis=1)]
    tuples_list = list(map(tuple, values_array))
    print(tuples_list)
    combinations_dict = {    
                            (0, 1, 0):prob_1_0,
                            (0, 2, 0):prob_2_2_and_1_1,
                            (1, 0, 0):prob_1_0, 
                        (1, 1, 0):prob_2_0, 
                        (1, 2, 0):prob_2_1, 
                        (2, 0, 0):prob_2_2_and_1_1, 
                        (2, 1, 0):prob_2_1, 
                        (2, 2, 0):prob_2_2_and_1_1, 
                        (0, 0, 1):prob_1_0, 
                        (0, 1, 1):prob_2_0,
                        (0, 2, 1):prob_2_1,
                        (1, 0, 1):prob_2_0 ,
                        (1, 1, 1):prob_3_0,
                        (1, 2, 1):prob_3_1 ,
                        (2, 0, 1):prob_2_1 ,
                        (2, 1, 1):prob_3_1,
                        (2, 2, 1):prob_3_2 ,
                        (0, 0, 2):prob_2_2_and_1_1, 
                        (0, 1, 2):prob_2_1 ,
                        (0, 2, 2):prob_2_2_and_1_1, 
                        (1, 0, 2):prob_2_1, 
                        (1, 1, 2):prob_3_1,
                        (1, 2, 2):prob_3_2 ,
                        (2, 0, 2):prob_2_2_and_1_1, 
                        (2, 1, 2):prob_3_2 ,
                        (2, 2, 2):prob_3_3,
                        }
    prob = list()
    #prob_random = list()
    treatment_outcomes = np.empty(len(tuples_list))
    for i in range(len(tuples_list)):
        prob.append(combinations_dict.get(tuples_list[i]))
        #prob_random.append(prob[i]+round(np.random.uniform(-0.02,0.02),4))
        #print(Counter(prob_random))
        #probability = prob_random[i]
        probability = prob[i]
        if np.isnan(probability):
            treatment_outcomes[i] = np.nan
        # otherwise, generate a random outcome based on the probability using np.random.choice
        else:
            if np.random.rand() < probability:
                treatment_outcomes[i] = 1
            else:
                treatment_outcomes[i] = 0

    treatment_outcomes = treatment_outcomes.astype(int)
    df = pd.DataFrame({'alc_inter': alc_int, 'mh_inter': mh_int, 'smoke_inter': smoking_int,'age':age_1,'sex':sex,'bmi':body_mass,'education':education,'cavitation':cavitation,'ttd':ttd1})
    df = df[~((df['alc_inter'] == 0) & (df['mh_inter'] == 0) & (df['smoke_inter'] == 0))]
    df['treatment_outcomes'] = treatment_outcomes
    df['alcohol'] = df['alc_inter'].apply(lambda x: 1 if x == 1 or x == 2 else 0)
    df['mental_health'] = df['mh_inter'].apply(lambda x: 1 if x == 1 or x == 2 else 0)
    df['smoking'] = df['smoke_inter'].apply(lambda x: 1 if x == 1 or x == 2 else 0)
    df = df.sample(frac=1).reset_index(drop=True)
    case1_1 = df.loc[(df['smoke_inter'] == 2) & (df['alc_inter'] == 2) & (df['mh_inter'] == 2), 'treatment_outcomes'].value_counts(normalize=True)
    case1_1 = case1_1.get(1, 0)
    case1_2 = df.loc[(df['smoke_inter'] == 2) & (df['alc_inter'] == 2) & (df['mh_inter'] == 1), 'treatment_outcomes'].value_counts(normalize=True)
    case1_2 = case1_2.get(1, 0)
    case1_3 = df.loc[(df['smoke_inter'] == 2) & (df['alc_inter'] == 1) & (df['mh_inter'] == 2), 'treatment_outcomes'].value_counts(normalize=True)
    case1_3 = case1_3.get(1, 0)
    case1_4 = df.loc[(df['smoke_inter'] == 1) & (df['alc_inter'] == 2) & (df['mh_inter'] == 2), 'treatment_outcomes'].value_counts(normalize=True)
    case1_4 = case1_4.get(1, 0)
    case1_5 = df.loc[(df['smoke_inter'] == 2) & (df['alc_inter'] == 1) & (df['mh_inter'] == 1), 'treatment_outcomes'].value_counts(normalize=True)
    case1_5 = case1_5.get(1, 0)
    case1_6 = df.loc[(df['smoke_inter'] == 1) & (df['alc_inter'] == 2) & (df['mh_inter'] == 1), 'treatment_outcomes'].value_counts(normalize=True)
    case1_6 = case1_6.get(1, 0)
    case1_7 = df.loc[(df['smoke_inter'] == 1) & (df['alc_inter'] == 1) & (df['mh_inter'] == 2), 'treatment_outcomes'].value_counts(normalize=True)
    case1_7 = case1_7.get(1, 0)
    case1_8 = df.loc[(df['smoke_inter'] == 1) & (df['alc_inter'] == 1) & (df['mh_inter'] == 1), 'treatment_outcomes'].value_counts(normalize=True)
    case1_8 = case1_8.get(1, 0)

    case2_1 = df.loc[(df['smoke_inter'] == 2) & (df['alc_inter'] == 2) & (df['mh_inter'] == 0), 'treatment_outcomes'].value_counts(normalize=True)
    case2_1 = case2_1.get(1, 0)
    case2_2 = df.loc[(df['smoke_inter'] == 2) & (df['alc_inter'] == 1) & (df['mh_inter'] == 0), 'treatment_outcomes'].value_counts(normalize=True)
    case2_2 = case2_2.get(1, 0)
    case2_3 = df.loc[(df['smoke_inter'] == 1) & (df['alc_inter'] == 2) & (df['mh_inter'] == 0), 'treatment_outcomes'].value_counts(normalize=True)
    case2_3 = case2_3.get(1, 0)
    case2_4 = df.loc[(df['smoke_inter'] == 1) & (df['alc_inter'] == 1) & (df['mh_inter'] == 0), 'treatment_outcomes'].value_counts(normalize=True)
    case2_4 = case2_4.get(1, 0)

    case3_1 = df.loc[(df['smoke_inter'] == 2) & (df['alc_inter'] == 0) & (df['mh_inter'] == 2), 'treatment_outcomes'].value_counts(normalize=True)
    case3_1 = case3_1.get(1, 0)
    case3_2 = df.loc[(df['smoke_inter'] == 2) & (df['alc_inter'] == 0) & (df['mh_inter'] == 1), 'treatment_outcomes'].value_counts(normalize=True)
    case3_2 = case3_2.get(1, 0)
    case3_3 = df.loc[(df['smoke_inter'] == 1) & (df['alc_inter'] == 0) & (df['mh_inter'] == 2), 'treatment_outcomes'].value_counts(normalize=True)
    case3_3 = case3_3.get(1, 0)
    case3_4 = df.loc[(df['smoke_inter'] == 1) & (df['alc_inter'] == 0) & (df['mh_inter'] == 1), 'treatment_outcomes'].value_counts(normalize=True)
    case3_4 = case3_4.get(1, 0)

    case4_1 = df.loc[(df['smoke_inter'] == 0) & (df['alc_inter'] == 2) & (df['mh_inter'] == 2), 'treatment_outcomes'].value_counts(normalize=True)
    case4_1 = case4_1.get(1, 0)
    case4_2 = df.loc[(df['smoke_inter'] == 0) & (df['alc_inter'] == 2) & (df['mh_inter'] == 1), 'treatment_outcomes'].value_counts(normalize=True)
    case4_2 = case4_2.get(1, 0)
    case4_3 = df.loc[(df['smoke_inter'] == 0) & (df['alc_inter'] == 1) & (df['mh_inter'] == 2), 'treatment_outcomes'].value_counts(normalize=True)
    case4_3 = case4_3.get(1, 0)
    case4_4 = df.loc[(df['smoke_inter'] == 0) & (df['alc_inter'] == 1) & (df['mh_inter'] == 1), 'treatment_outcomes'].value_counts(normalize=True)
    case4_4 = case4_4.get(1, 0)

    case5_1 = df.loc[(df['smoke_inter'] == 2) & (df['alc_inter'] == 0) & (df['mh_inter'] == 0), 'treatment_outcomes'].value_counts(normalize=True)
    case5_1 = case5_1.get(1, 0)
    case5_2 = df.loc[(df['smoke_inter'] == 1) & (df['alc_inter'] == 0) & (df['mh_inter'] == 0), 'treatment_outcomes'].value_counts(normalize=True)
    case5_2 = case5_2.get(1, 0)

    case6_1 = df.loc[(df['smoke_inter'] == 0) & (df['alc_inter'] == 0) & (df['mh_inter'] == 2), 'treatment_outcomes'].value_counts(normalize=True)
    case6_1 = case6_1.get(1, 0)
    case6_2 = df.loc[(df['smoke_inter'] == 0) & (df['alc_inter'] == 0) & (df['mh_inter'] == 1), 'treatment_outcomes'].value_counts(normalize=True)
    case6_2 = case6_2.get(1, 0)

    case7_1 = df.loc[(df['smoke_inter'] == 0) & (df['alc_inter'] == 2) & (df['mh_inter'] == 0), 'treatment_outcomes'].value_counts(normalize=True)
    case7_1 = case7_1.get(1, 0)
    case7_2 = df.loc[(df['smoke_inter'] == 0) & (df['alc_inter'] == 1) & (df['mh_inter'] == 0), 'treatment_outcomes'].value_counts(normalize=True)
    case7_2 = case7_2.get(1, 0)

    power_intervention_smoke = case1_1*(1/8) + case1_2*(1/8) + case1_3*(1/8) + case1_5*(1/8) + case2_1*(1/4) + case2_2*(1/4) + case3_1*(1/4) + case3_2*(1/4) + case5_1*(1/2)
    power_intervention_alc = case1_1*(1/8) + case1_2*(1/8) + case1_4*(1/8) + case1_6*(1/8) + case2_1*(1/4) + case2_3*(1/4) + case4_1*(1/4) + case4_2*(1/4) + case7_1*(1/2)
    power_intervention_mh = case1_1*(1/8) + case1_3*(1/8) + case1_4*(1/8) + case1_7*(1/8) + case3_1*(1/4) + case3_3*(1/4) + case4_1*(1/4) + case4_3*(1/4) + case6_1*(1/2)


    power_control_smoke = case1_4*(1/8) + case1_6*(1/8) + case1_7*(1/8) + case1_8*(1/8) + case2_3*(1/4) + case2_4*(1/4) + case3_3*(1/4) + case3_4*(1/4) + case5_2*(1/2)
    power_control_alc = case1_3*(1/8) + case1_5*(1/8) + case1_7*(1/8) + case1_8*(1/8) + case2_2*(1/4) + case2_4*(1/4) + case4_3*(1/4) + case4_4*(1/4) + case7_2*(1/2)
    power_control_mh = case1_2*(1/8) + case1_5*(1/8) + case1_6*(1/8) + case1_8*(1/8) + case3_2*(1/4) + case3_4*(1/4) + case4_2*(1/4) + case4_4*(1/4) + case6_2*(1/2)
    
    my_list = [power_intervention_smoke, power_intervention_alc, power_intervention_mh, power_control_smoke, power_control_alc, power_control_mh]
    print(my_list)
    
    df['treatment_smoke'] = df['smoke_inter'].apply(lambda x: np.random.choice([2, 1], p=[power_intervention_smoke, 1- power_intervention_smoke]) if x == 2 
                                                    else np.random.choice([2, 1], p=[power_control_smoke, 1-power_control_smoke]) if x == 1 else 0)
    
    df['treatment_alc'] = df['alc_inter'].apply(lambda x: np.random.choice([2, 1], p=[power_intervention_alc, 1 - power_intervention_alc]) if x == 2 
                                                    else np.random.choice([2, 1], p=[power_control_alc, 1-power_control_alc]) if x == 1 else 0)
    
    df['treatment_mh'] = df['mh_inter'].apply(lambda x: np.random.choice([2, 1], p=[power_intervention_mh, 1-power_intervention_mh]) if x == 2 
                                                    else np.random.choice([2, 1], p=[power_control_mh, 1-power_control_mh]) if x == 1 else 0)
    dfx = try_sample_size(df,prob_2_2_and_1_1,prob_3_3,prob_2_1,prob_3_2,prob_3_1,prob_1_0,prob_2_0,prob_3_0,male,
                        female,
                        low_bmi,
                        normal_bmi,
                        high_bmi,
                        l1_edu,
                        l2_edu,
                        l3_edu,
                        l4_edu,
                        min_age,
                        max_age,
                        mean_age,
                        sd_age, 
                        mean_ttd,
                        std_ttd)
    print(calculate_power1(2,1,dfx))
    print(calculate_power2(2,1,dfx))
    print(calculate_power3(2,1,dfx))

    return StreamingResponse(iter([dfx.to_csv(index=False)]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=clinical_samples_dataset.csv"})
