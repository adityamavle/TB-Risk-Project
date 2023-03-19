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
#from power_analysis import power_analysis
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, UploadFile,File
from pydantic import BaseModel
import io 
#from flask import Flask, request
app = FastAPI()


@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post('/simulate_dataset')
def create_dataset2(data:simulation):
  
  """
  User Defined Inputs:
  1.size=Population Sample size
  2.percentage_alcoholism/depression/tobacco=Percentage of the the respective condition in the population
  3.percentage_alcoholism_depression(and other dual variable overlaps)= Percentage of the overlap of respective two condtions in the populations
  4.percentage_tobacco_alcoholism_depression= Percentage of the overlap of all 3 cases in the population
  5.age, gender, bmi, edu: A list consisting of the percentages of distribution in the various categorical buckets for the following variables:
  eg; age=[18,60,35,15]:min,max,mean,std dev 
  age:4 elements; gender: 2 buckets; bmi: 3 buckets; edu: 4 buckets
  6.Treatment_noth/treatment_1_conditions=The percentage of good treatment outcomes for the population with the following number of conditions
  """
  print(type(data.dict))
  try:
    data= data.dict()
    size = data['size']
    percentage_alc_only = data['percentage_alc_only']
    percentage_dep_only = data['percentage_dep_only'] 
    percentage_tobacco_only = data['percentage_tobacco_only']
    percentage_alc_dep = data['percentage_alc_dep']
    percentage_alc_tobacco = data['percentage_alc_tobacco']
    percentage_dep_tobacco = data['percentage_dep_tobacco'] 
    percentage_tobacco_alcoholism_depression = data['percentage_tobacco_alcoholism_depression']
    treatment_noth = data['treatment_noth']
    treatment_1_conditions = data['treatment_1_conditions']
    treatment_2_conditions = data['treatment_2_conditions']
    treatment_3_conditions = data['treatment_3_conditions']
    gender = data['gender']
    bmi = data['bmi']
    edu = data['edu']
    seed = data['seed']
    age = data['age']

    zeros=1-(percentage_alc_only + percentage_dep_only  + percentage_tobacco_only + percentage_alc_dep + percentage_alc_tobacco + percentage_dep_tobacco + percentage_tobacco_alcoholism_depression)
    
    print("zeros:{}".format(zeros))
    np.random.seed(2020)
    condition=np.random.choice(8,size,p=[zeros, percentage_alc_only, percentage_dep_only,percentage_alc_dep,percentage_tobacco_only,percentage_alc_tobacco, percentage_dep_tobacco, percentage_tobacco_alcoholism_depression])
    alcoholism=np.where((condition == 1) | (condition==3) | (condition==5) | (condition==7),1,0)
    depression=np.where((condition == 2) | (condition==3) | (condition==6) | (condition==7),1,0)
    
    alc_only=np.where(condition == 1,1,0)
    dep_only=np.where(condition == 2,1,0)
    tobacco_only=np.where(condition == 4,1,0)

    alc_dep=np.where(condition == 3,1,0)
    alc_tobacco=np.where(condition == 5,1,0)
    dep_tobacco=np.where(condition == 6,1,0)

    tobacco=np.where((condition == 4) | (condition==5) | (condition==6) | (condition==7),1,0)
    all_three=np.where(condition == 7,1,0)
    age_1= np.random.normal(loc=age[2], scale=age[3], size=size)
    age_1=np.clip(age_1, a_min=age[0], a_max=age[1])
    age_1 = np.round(age_1).astype(int)
    sex=np.random.choice(2,size,p=[gender[0],gender[1]])
    body_mass=np.random.choice(3,size,p=[bmi[0],bmi[1],bmi[2]])
    education=np.random.choice(4,size,p=[edu[0],edu[1],edu[2],edu[3]])
    cavitation = np.random.choice(2,size,p=[0.5,0.5])
    ttd = np.random.normal(loc=7, scale=3, size=size)
    df = pd.DataFrame(
      {
          'idx': np.arange(1, size+1),
          'age': age_1,
          'gender': sex,
          'bmi': body_mass,
          'education': education,
          'cavitation': cavitation,
          'TTD': ttd,
          'alcoholism': alcoholism,
          'depression': depression,
          'tobacco': tobacco,
          'alcohol_only':alc_only,
          'depression_only':dep_only,
          'tobacco_only': tobacco_only,
          'alcoholism+depression':alc_dep,
          'alcoholism+tobacco':alc_tobacco,
          'depression+tobacco':dep_tobacco,
          'tobacco+alcohol+smoking':all_three
      }
    )
    intervention_arr=[]
    choices_alc_only=['NAlc','A']
    choices_dep_only=['ND','D']
    choices_tobacco_only=['NT','T']
    choices_alc_dep_only=['NAD','AD']
    choices_alc_tobacco_only=['NAT','AT']
    choices_dep_tobacco_only=['NDT','DT']
    choices_all_3=['NADT','ADT']
    random.seed(seed)
    weights=[0.5,0.5]
    for i in range(size):
      if(df['alcohol_only'][i]==1):
        #intervention_arr.append(np.random.binomial(1,0.5,size=1))
        intervention_arr.append(random.choices(choices_alc_only,weights=weights)[0])
      if(df['depression_only'][i]==1):
        intervention_arr.append(random.choices(choices_dep_only,weights=weights)[0])
      if(df['tobacco_only'][i]==1):
        intervention_arr.append(random.choices(choices_tobacco_only,weights=weights)[0])
      if(df['alcoholism+depression'][i]==1):
        intervention_arr.append(random.choices(choices_alc_dep_only,weights=weights)[0])
      if(df['alcoholism+tobacco'][i]==1):
        intervention_arr.append(random.choices(choices_alc_tobacco_only,weights=weights)[0])
      if(df['depression+tobacco'][i]==1):
        intervention_arr.append(random.choices(choices_dep_tobacco_only,weights=weights)[0])
      if(df['tobacco+alcohol+smoking'][i]==1):
        intervention_arr.append(random.choices(choices_all_3,weights=weights)[0])
      if(df['alcoholism'][i]==0 and df['depression'][i]==0 and df['tobacco'][i]==0):
        intervention_arr.append('UNAFFECTED')
    df['Intervention'] = intervention_arr
    #print(np.unique(intervention_arr,return_councentage_ts=True))
    df['treatment_outcomes'] = " "
    treatment_outcomes_single_ni = []
    treatment_outcomes_two_ni = []
    treatment_outcomes_three_ni = []
    treatment_outcomes_i = []
    #treatment_outcomes_noth = []

    list_noth = list(np.where(df['Intervention'] == 'UNAFFECTED')[0])
    values_noth = np.random.choice(2,len(list_noth),p=[1-treatment_noth,treatment_noth])
    for i in range(len(list_noth)):
      df.loc[list_noth[i],"treatment_outcomes"] = values_noth[i]
    
    list_single_ni = list(np.where((df['Intervention'] == 'NAlc') | (df['Intervention'] == 'ND') | (df['Intervention'] == 'NT'))[0])
    values_single_ni = np.random.choice(2,len(list_single_ni),p=[1-treatment_1_conditions,treatment_1_conditions])
    for i in range(len(list_single_ni)):
      df.loc[list_single_ni[i],"treatment_outcomes"] = values_single_ni[i]

    list_two_ni = list(np.where((df['Intervention'] == 'NAD') | (df['Intervention'] == 'NDT') | (df['Intervention'] == 'NAT'))[0])
    values_two_ni = np.random.choice(2,len(list_two_ni),p=[1-treatment_2_conditions,treatment_2_conditions])
    for i in range(len(list_two_ni)):
      df.loc[list_two_ni[i],"treatment_outcomes"] = values_two_ni[i]

    list_three_ni = list(np.where(df['Intervention'] == 'NADT')[0])
    values_three_ni = np.random.choice(2,len(list_three_ni),p=[1-treatment_3_conditions,treatment_3_conditions])
    for i in range(len(list_three_ni)):
      df.loc[list_three_ni[i],"treatment_outcomes"] = values_three_ni[i]    
    list_single_inter=list(np.where((df['Intervention'] == 'A') | (df['Intervention'] == 'D') | (df['Intervention'] == 'T'))[0])
    s_int=(1-treatment_1_conditions)/2
    values_single_inter = np.random.choice(2,len(list_single_inter),p=[s_int,treatment_1_conditions+s_int])
    for i in range(len(list_single_inter)):
      df.loc[list_single_inter[i],"treatment_outcomes"] = values_single_inter[i]

    list_double_inter=list(np.where((df['Intervention'] == 'AD') | (df['Intervention'] == 'DT') | (df['Intervention'] == 'AT'))[0])
    d_int=(1-treatment_2_conditions)/2
    values_double_inter = np.random.choice(2,len(list_double_inter),p=[d_int,treatment_2_conditions + d_int])
    for i in range(len(list_double_inter)):
      df.loc[list_double_inter[i],"treatment_outcomes"] = values_double_inter[i]

    list_triple_inter=list(np.where((df['Intervention'] == 'ADT'))[0])
    t_int = (1-treatment_3_conditions)/2
    values_triple_inter = np.random.choice(2,len(list_triple_inter),p=[t_int,treatment_3_conditions+ t_int])
    for i in range(len(list_triple_inter)):
      df.loc[list_triple_inter[i],"treatment_outcomes"] = values_triple_inter[i]


    return StreamingResponse(
        iter([df.to_csv(index=False)]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=init_dataset.csv"})
  except Exception as ex:
    print(ex)

def create_dataset(size,percentage_alc_only, percentage_dep_only, percentage_tobacco_only,percentage_alc_dep, percentage_alc_tobacco,percentage_dep_tobacco, 
                    percentage_tobacco_alcoholism_depression,treatment_noth,treatment_1_conditions,treatment_2_conditions,treatment_3_conditions,
                    gender,bmi,edu,seed,age):
  """
  User Defined Inputs:
  1.size=Population Sample size
  2.percentage_alcoholism/depression/tobacco=Percentage of the the respective condition in the population
  3.percentage_alcoholism_depression(and other dual variable overlaps)= Percentage of the overlap of respective two condtions in the populations
  4.percentage_tobacco_alcoholism_depression= Percentage of the overlap of all 3 cases in the population
  5.gender, bmi, edu: A list consisting of the percentages of distribution in the various categorical buckets for the following variables:
  eg;  
  buckets; gender: 2 buckets; bmi: 3 buckets; edu: 4 buckets
  6.age has 4 parameters - a_min, a_max, mean, std dev
  6.Treatment_noth/treatment_1_conditions=The percentage of good treatment outcomes for the population with the following number of conditions
  """
  zeros=1-(percentage_alc_only + percentage_dep_only  + percentage_tobacco_only + percentage_alc_dep + percentage_alc_tobacco + percentage_dep_tobacco + percentage_tobacco_alcoholism_depression)
  
  print("zeros:{}".format(zeros))
  np.random.seed(2020)
  condition=np.random.choice(8,size,p=[zeros, percentage_alc_only, percentage_dep_only,percentage_alc_dep,percentage_tobacco_only,percentage_alc_tobacco, percentage_dep_tobacco, percentage_tobacco_alcoholism_depression])
  alcoholism=np.where((condition == 1) | (condition==3) | (condition==5) | (condition==7),1,0)
  depression=np.where((condition == 2) | (condition==3) | (condition==6) | (condition==7),1,0)
  
  alc_only=np.where(condition == 1,1,0)
  dep_only=np.where(condition == 2,1,0)
  tobacco_only=np.where(condition == 4,1,0)

  alc_dep=np.where(condition == 3,1,0)
  alc_tobacco=np.where(condition == 5,1,0)
  dep_tobacco=np.where(condition == 6,1,0)

  tobacco=np.where((condition == 4) | (condition==5) | (condition==6) | (condition==7),1,0)
  all_three=np.where(condition == 7,1,0)
  age_1= np.random.normal(loc=age[2], scale=age[3], size=size)
  age_1=np.clip(age_1, a_min=age[0], a_max=age[1])
  age_1 = np.round(age_1).astype(int)
  sex=np.random.choice(2,size,p=[gender[0],gender[1]])
  body_mass=np.random.choice(3,size,p=[bmi[0],bmi[1],bmi[2]])
  education=np.random.choice(4,size,p=[edu[0],edu[1],edu[2],edu[3]])
  cavitation = np.random.choice(2,size,p=[0.5,0.5])
  ttd = np.random.normal(loc=7, scale=3, size=size)
  df = pd.DataFrame(
    {
        'idx': np.arange(1, size+1),
        'age': age_1,
        'gender': sex,
        'bmi': body_mass,
        'education': education,
        'cavitation': cavitation,
      	'TTD': ttd,
        'alcoholism': alcoholism,
        'depression': depression,
        'tobacco': tobacco,
        'alcohol_only':alc_only,
        'depression_only':dep_only,
        'tobacco_only': tobacco_only,
        'alcoholism+depression':alc_dep,
        'alcoholism+tobacco':alc_tobacco,
        'depression+tobacco':dep_tobacco,
        'tobacco+alcohol+smoking':all_three
     }
  )
  intervention_arr=[]
  choices_alc_only=['NAlc','A']
  choices_dep_only=['ND','D']
  choices_tobacco_only=['NT','T']
  choices_alc_dep_only=['NAD','AD']
  choices_alc_tobacco_only=['NAT','AT']
  choices_dep_tobacco_only=['NDT','DT']
  choices_all_3=['NADT','ADT']
  random.seed(seed)
  weights=[0.5,0.5]
  for i in range(size):
    if(df['alcohol_only'][i]==1):
      #intervention_arr.append(np.random.binomial(1,0.5,size=1))
      intervention_arr.append(random.choices(choices_alc_only,weights=weights)[0])
    if(df['depression_only'][i]==1):
      intervention_arr.append(random.choices(choices_dep_only,weights=weights)[0])
    if(df['tobacco_only'][i]==1):
      intervention_arr.append(random.choices(choices_tobacco_only,weights=weights)[0])
    if(df['alcoholism+depression'][i]==1):
      intervention_arr.append(random.choices(choices_alc_dep_only,weights=weights)[0])
    if(df['alcoholism+tobacco'][i]==1):
      intervention_arr.append(random.choices(choices_alc_tobacco_only,weights=weights)[0])
    if(df['depression+tobacco'][i]==1):
      intervention_arr.append(random.choices(choices_dep_tobacco_only,weights=weights)[0])
    if(df['tobacco+alcohol+smoking'][i]==1):
      intervention_arr.append(random.choices(choices_all_3,weights=weights)[0])
    if(df['alcoholism'][i]==0 and df['depression'][i]==0 and df['tobacco'][i]==0):
      intervention_arr.append('UNAFFECTED')
  df['Intervention'] = intervention_arr
  #print(np.unique(intervention_arr,return_councentage_ts=True))
  df['treatment_outcomes'] = " "
  treatment_outcomes_single_ni = []
  treatment_outcomes_two_ni = []
  treatment_outcomes_three_ni = []
  treatment_outcomes_i = []
  #treatment_outcomes_noth = []

  list_noth = list(np.where(df['Intervention'] == 'UNAFFECTED')[0])
  values_noth = np.random.choice(2,len(list_noth),p=[1-treatment_noth,treatment_noth])
  for i in range(len(list_noth)):
    df.loc[list_noth[i],"treatment_outcomes"] = values_noth[i]
  
  list_single_ni = list(np.where((df['Intervention'] == 'NAlc') | (df['Intervention'] == 'ND') | (df['Intervention'] == 'NT'))[0])
  values_single_ni = np.random.choice(2,len(list_single_ni),p=[1-treatment_1_conditions,treatment_1_conditions])
  for i in range(len(list_single_ni)):
    df.loc[list_single_ni[i],"treatment_outcomes"] = values_single_ni[i]

  list_two_ni = list(np.where((df['Intervention'] == 'NAD') | (df['Intervention'] == 'NDT') | (df['Intervention'] == 'NAT'))[0])
  values_two_ni = np.random.choice(2,len(list_two_ni),p=[1-treatment_2_conditions,treatment_2_conditions])
  for i in range(len(list_two_ni)):
    df.loc[list_two_ni[i],"treatment_outcomes"] = values_two_ni[i]

  list_three_ni = list(np.where(df['Intervention'] == 'NADT')[0])
  values_three_ni = np.random.choice(2,len(list_three_ni),p=[1-treatment_3_conditions,treatment_3_conditions])
  for i in range(len(list_three_ni)):
    df.loc[list_three_ni[i],"treatment_outcomes"] = values_three_ni[i]

  #list_i = list(np.where((df['Intervention'] == 'A') | (df['Intervention'] == 'D') | (df['Intervention'] == 'T') | (df['Intervention'] == 'AD') | (df['Intervention'] == 'AT') | (df['Intervention'] == 'DT') | (df['Intervention'] == 'ADT'))[0])
  #values_i = np.random.choice(2,len(list_i),p=[1-treatment_intervention,treatment_intervention])
  #for i in range(len(list_i)):
   # df.loc[list_i[i],"treatment_outcomes"] = values_i[i]
  
  list_single_inter=list(np.where((df['Intervention'] == 'A') | (df['Intervention'] == 'D') | (df['Intervention'] == 'T'))[0])
  s_int=(1-treatment_1_conditions)/2
  values_single_inter = np.random.choice(2,len(list_single_inter),p=[s_int,treatment_1_conditions+s_int])
  for i in range(len(list_single_inter)):
    df.loc[list_single_inter[i],"treatment_outcomes"] = values_single_inter[i]

  list_double_inter=list(np.where((df['Intervention'] == 'AD') | (df['Intervention'] == 'DT') | (df['Intervention'] == 'AT'))[0])
  d_int=(1-treatment_2_conditions)/2
  values_double_inter = np.random.choice(2,len(list_double_inter),p=[d_int,treatment_2_conditions + d_int])
  for i in range(len(list_double_inter)):
    df.loc[list_double_inter[i],"treatment_outcomes"] = values_double_inter[i]

  list_triple_inter=list(np.where((df['Intervention'] == 'ADT'))[0])
  t_int = (1-treatment_3_conditions)/2
  values_triple_inter = np.random.choice(2,len(list_triple_inter),p=[t_int,treatment_3_conditions+ t_int])
  for i in range(len(list_triple_inter)):
    df.loc[list_triple_inter[i],"treatment_outcomes"] = values_triple_inter[i]

  return df


@app.post('/calc_power')
def calculate_power(x:str,y:str,file:UploadFile=File(...)):
    """ x = value of treatment variable (gives us idea of the population - NAlc, a, nt, t, adt, etc)
        y = value of control variable """
    try:
        df = pd.read_csv(file.file)
        file.file.close
        #print(df.head())
        power_analysis=TTestIndPower()
        treatment_arr=[]
        control_arr=[]
        treatment_locs=np.where((df['Intervention']==x))
        control_locs=np.where((df['Intervention']==y))
        for i in treatment_locs:
            treatment_arr.append(df['treatment_outcomes'].iloc[i])
        for j in control_locs:
            control_arr.append(df['treatment_outcomes'].iloc[j])
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
    except Exception as ex:
      print(ex) 

def calculate_power1(x, y, df):
    power_analysis=TTestIndPower()
    treatment_arr=[]
    control_arr=[]
    treatment_locs=np.where((df['Intervention']==x))
    control_locs=np.where((df['Intervention']==y))
    for i in treatment_locs:
        treatment_arr.append(df['treatment_outcomes'].iloc[i])
    for j in control_locs:
        control_arr.append(df['treatment_outcomes'].iloc[j])
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

#@app.post('/check_samp_sizes')
def check_sample_sizes(dfx):
  sample_sizes={}
  sample_sizes['Alcohol']=calculate_power1('A','NAlc',dfx)
  sample_sizes['Depression']=calculate_power1('D','ND',dfx)  
  sample_sizes['Tobacco']=calculate_power1('T','NT',dfx)  
  sample_sizes['Alcohol-Depression']=calculate_power1('AD','NAD',dfx)  
  sample_sizes['Alcohol-Tobacco']=calculate_power1('AT','NAT',dfx)  
  sample_sizes['Depression-Tobacco']=calculate_power1('DT','NDT',dfx) 
  sample_sizes['Alcohol-Depression-Tobacco']=calculate_power1('ADT','NADT',dfx) 
  #for i in sample_sizes.values():
  all_above_threshold = all(value >= 0.8 for value in sample_sizes.values())
  print(all_above_threshold)
  print(sample_sizes.values())
  return sample_sizes.values
# check_sample_sizes(df_2)

def try_sample_sizes3(dfx):
    sample_sizes = {}
    sample_sizes['Alcohol'] = calculate_power1('A', 'NAlc', dfx)
    sample_sizes['Depression'] = calculate_power1('D', 'ND', dfx)
    sample_sizes['Tobacco'] = calculate_power1('T', 'NT', dfx)
    sample_sizes['Alcohol-Depression'] = calculate_power1('AD', 'NAD', dfx)
    sample_sizes['Alcohol-Tobacco'] = calculate_power1('AT', 'NAT', dfx)
    sample_sizes['Depression-Tobacco'] = calculate_power1('DT', 'NDT', dfx)
    sample_sizes['Alcohol-Depression-Tobacco'] = calculate_power1('ADT', 'NADT', dfx)

    size = dfx.shape[0]
    #total_power = sum(sample_sizes.values())
    print(sample_sizes.values())
    all_above_threshold = all(value >= 0.79 for value in sample_sizes.values())
    print(all_above_threshold)
    #print(type(dfx))
    print(type(dfx.shape[0]))
    if not all_above_threshold:
        if sample_sizes['Alcohol'] < 0.79:
            size += 100
        elif sample_sizes['Depression'] < 0.79:
            size += 100
        elif sample_sizes['Tobacco'] < 0.79:
            size += 100
        elif sample_sizes['Alcohol-Depression'] < 0.79:
            size += 80
        elif sample_sizes['Alcohol-Tobacco'] < 0.79:
            size += 80
        elif sample_sizes['Depression-Tobacco'] < 0.79:
            size += 80
        elif sample_sizes['Alcohol-Depression-Tobacco'] < 0.79:
            size += 20
          
        ratio_alc_only = round(dfx['Intervention'].value_counts(normalize=True)['A']+dfx['Intervention'].value_counts(normalize=True)['NAlc'],2)
        ratio_dep_only = round(dfx['Intervention'].value_counts(normalize=True)['D']+dfx['Intervention'].value_counts(normalize=True)['ND'],2)
        ratio_tob_only = round(dfx['Intervention'].value_counts(normalize=True)['T']+dfx['Intervention'].value_counts(normalize=True)['NT'],2)
        ratio_at = round(dfx['Intervention'].value_counts(normalize=True)['AT']+dfx['Intervention'].value_counts(normalize=True)['NAT'],2)
        ratio_ad = round(dfx['Intervention'].value_counts(normalize=True)['AD']+dfx['Intervention'].value_counts(normalize=True)['NAD'],2)
        ratio_dt = round(dfx['Intervention'].value_counts(normalize=True)['DT']+dfx['Intervention'].value_counts(normalize=True)['NDT'],2)
        ratio_adt = round(dfx['Intervention'].value_counts(normalize=True)['ADT']+dfx['Intervention'].value_counts(normalize=True)['NADT'],2)
        
        #print(dfx['Intervention'].value_counts(normalize=True))
        print(dfx.shape[0])
        #dfx = create_dataset2(size, 0.08, 0.08, 0.08, 0.04, 0.04, 0.04, 0.03, 0.9, 0.80, 0.70, 0.60, gender=[0.5, 0.5], bmi=[0.2,0.5,0.3],edu=[0.1,0.2,0.2,0.5],seed=52)
        print('inside growing')
        dfx = create_dataset(size, ratio_alc_only, ratio_dep_only, ratio_tob_only, ratio_ad, ratio_at, ratio_dt, ratio_adt, 0.95, 0.90, 0.85, 0.80, gender=[0.5, 0.5], bmi=[0.2,0.5,0.3],edu=[0.1,0.2,0.2,0.5],age=[18,60,35,15],seed=52)
        dfx = try_sample_sizes3(dfx)
    while all(value > 0.85 for value in sample_sizes.values()):
            print('inside shrinking')
                #for group in sample_sizes.keys:
                    #if sample_sizes[group] > 0.95:
                    #   size -= 1500
            if sample_sizes['Alcohol'] > 0.95:
                size -= 1500 
            elif sample_sizes['Tobacco'] > 0.95:
                size -= 1500
            elif sample_sizes['Depression'] > 0.95:
                size -= 1500
            ratio_alc_only = round(dfx['Intervention'].value_counts(normalize=True)['A']+dfx['Intervention'].value_counts(normalize=True)['NAlc'],2)
            ratio_dep_only = round(dfx['Intervention'].value_counts(normalize=True)['D']+dfx['Intervention'].value_counts(normalize=True)['ND'],2)
            ratio_tob_only = round(dfx['Intervention'].value_counts(normalize=True)['T']+dfx['Intervention'].value_counts(normalize=True)['NT'],2)
            ratio_at = round(dfx['Intervention'].value_counts(normalize=True)['AT']+dfx['Intervention'].value_counts(normalize=True)['NAT'],2)
            ratio_ad = round(dfx['Intervention'].value_counts(normalize=True)['AD']+dfx['Intervention'].value_counts(normalize=True)['NAD'],2)
            ratio_dt = round(dfx['Intervention'].value_counts(normalize=True)['DT']+dfx['Intervention'].value_counts(normalize=True)['NDT'],2)
            ratio_adt = round(dfx['Intervention'].value_counts(normalize=True)['ADT']+dfx['Intervention'].value_counts(normalize=True)['NADT'],2)
                
            dfx = create_dataset(size, ratio_alc_only, ratio_dep_only, ratio_tob_only, ratio_ad, ratio_at, ratio_dt, ratio_adt, 0.95, 0.90, 0.85, 0.80, gender=[0.5, 0.5], bmi=[0.2,0.5,0.3],edu=[0.1,0.2,0.2,0.5],age=[18,60,35,15],seed=52)
            #dfx = create_dataset2(size, 0.08, 0.08, 0.08, 0.04, 0.04, 0.04, 0.03, 0.9, 0.80, 0.70, 0.60, gender=[0.5, 0.5], bmi=[0.2,0.5,0.3],edu=[0.1,0.2,0.2,0.5],seed=52)
            dfx = try_sample_sizes3(dfx)
    return dfx

@app.post('/try_samp_sizes')
def process_file(file:UploadFile = File(...)):
  try:
      dfx=pd.read_csv(file.file)
      dfx1 = try_sample_sizes3(dfx)
      #csvx = dfx.to_csv(index = False)    
      return StreamingResponse(
                    iter([dfx1.to_csv(index=False)]),
                    media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=resim_screening.csv"})
  except Exception as ex:
      print(ex)

def try_sample_sizes4(file:UploadFile=File(...)):
    try:
        dfx = pd.read_csv(file.file)
        file.file.close
        #dfx = pd.read_csv(io.BytesIO(contents), encoding='utf-8')
        #file.file.close
        sample_sizes = {}
        sample_sizes['Alcohol'] = calculate_power1('A', 'NAlc', dfx)
        sample_sizes['Depression'] = calculate_power1('D', 'ND', dfx)
        sample_sizes['Tobacco'] = calculate_power1('T', 'NT', dfx)
        sample_sizes['Alcohol-Depression'] = calculate_power1('AD', 'NAD', dfx)
        sample_sizes['Alcohol-Tobacco'] = calculate_power1('AT', 'NAT', dfx)
        sample_sizes['Depression-Tobacco'] = calculate_power1('DT', 'NDT', dfx)
        sample_sizes['Alcohol-Depression-Tobacco'] = calculate_power1('ADT', 'NADT', dfx)

        size = dfx.shape[0]
        #total_power = sum(sample_sizes.values())
        all_above_threshold = all(value >= 0.79 for value in sample_sizes.values())
        if not all_above_threshold:
            if sample_sizes['Alcohol'] < 0.79:
                size += 100
            elif sample_sizes['Depression'] < 0.79:
                size += 100
            elif sample_sizes['Tobacco'] < 0.79:
                size += 100
            elif sample_sizes['Alcohol-Depression'] < 0.79:
                size += 80
            elif sample_sizes['Alcohol-Tobacco'] < 0.79:
                size += 80
            elif sample_sizes['Depression-Tobacco'] < 0.79:
                size += 80
            elif sample_sizes['Alcohol-Depression-Tobacco'] < 0.79:
                size += 20
              
            ratio_alc_only = round(dfx['Intervention'].value_counts(normalize=True)['A']+dfx['Intervention'].value_counts(normalize=True)['NAlc'],2)
            ratio_dep_only = round(dfx['Intervention'].value_counts(normalize=True)['D']+dfx['Intervention'].value_counts(normalize=True)['ND'],2)
            ratio_tob_only = round(dfx['Intervention'].value_counts(normalize=True)['T']+dfx['Intervention'].value_counts(normalize=True)['NT'],2)
            ratio_at = round(dfx['Intervention'].value_counts(normalize=True)['AT']+dfx['Intervention'].value_counts(normalize=True)['NAT'],2)
            ratio_ad = round(dfx['Intervention'].value_counts(normalize=True)['AD']+dfx['Intervention'].value_counts(normalize=True)['NAD'],2)
            ratio_dt = round(dfx['Intervention'].value_counts(normalize=True)['DT']+dfx['Intervention'].value_counts(normalize=True)['NDT'],2)
            ratio_adt = round(dfx['Intervention'].value_counts(normalize=True)['ADT']+dfx['Intervention'].value_counts(normalize=True)['NADT'],2)
            
            #dfx = create_dataset2(size, 0.08, 0.08, 0.08, 0.04, 0.04, 0.04, 0.03, 0.9, 0.80, 0.70, 0.60, gender=[0.5, 0.5], bmi=[0.2,0.5,0.3],edu=[0.1,0.2,0.2,0.5],seed=52)
            dfx = create_dataset(size, ratio_alc_only, ratio_dep_only, ratio_tob_only, ratio_ad, ratio_at, ratio_dt, ratio_adt, 0.95, 0.90, 0.85, 0.80, gender=[0.5, 0.5], bmi=[0.2,0.5,0.3],edu=[0.1,0.2,0.2,0.5],seed=52,age=[18,60,35,15])
            return try_sample_sizes3(dfx)
        while all(value > 0.85 for value in sample_sizes.values()):
                print('inside shrinking')
                #for group in sample_sizes.keys:
                    #if sample_sizes[group] > 0.95:
                    #   size -= 1500
                if sample_sizes['Alcohol'] > 0.95:
                  size -= 1500 
                elif sample_sizes['Tobacco'] > 0.95:
                  size -= 1500
                elif sample_sizes['Depression'] > 0.95:
                  size -= 1500
                    #if any(value < 0.80 for value in sample_sizes.values()):
                
                ratio_alc_only = round(dfx['Intervention'].value_counts(normalize=True)['A']+dfx['Intervention'].value_counts(normalize=True)['NAlc'],2)
                ratio_dep_only = round(dfx['Intervention'].value_counts(normalize=True)['D']+dfx['Intervention'].value_counts(normalize=True)['ND'],2)
                ratio_tob_only = round(dfx['Intervention'].value_counts(normalize=True)['T']+dfx['Intervention'].value_counts(normalize=True)['NT'],2)
                ratio_at = round(dfx['Intervention'].value_counts(normalize=True)['AT']+dfx['Intervention'].value_counts(normalize=True)['NAT'],2)
                ratio_ad = round(dfx['Intervention'].value_counts(normalize=True)['AD']+dfx['Intervention'].value_counts(normalize=True)['NAD'],2)
                ratio_dt = round(dfx['Intervention'].value_counts(normalize=True)['DT']+dfx['Intervention'].value_counts(normalize=True)['NDT'],2)
                ratio_adt = round(dfx['Intervention'].value_counts(normalize=True)['ADT']+dfx['Intervention'].value_counts(normalize=True)['NADT'],2)
                    
                dfx = create_dataset(size, ratio_alc_only, ratio_dep_only, ratio_tob_only, ratio_ad, ratio_at, ratio_dt, ratio_adt, 0.95, 0.90, 0.85, 0.80, gender=[0.5, 0.5], bmi=[0.2,0.5,0.3],edu=[0.1,0.2,0.2,0.5],seed=52,age=[18,60,35,15])
                #dfx = create_dataset2(size, 0.08, 0.08, 0.08, 0.04, 0.04, 0.04, 0.03, 0.9, 0.80, 0.70, 0.60, gender=[0.5, 0.5], bmi=[0.2,0.5,0.3],edu=[0.1,0.2,0.2,0.5],seed=52)
                return try_sample_sizes3(dfx)
        #return dfx
        return StreamingResponse(
                    iter([dfx.to_csv(index=False)]),
                    media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=resim_screening.csv"})
    except Exception as ex:
      print(ex) 

@app.post('/clinical_dataset')
def clinical_ss(file:UploadFile=File(...)):
    try:
        dfx = pd.read_csv(file.file)
        file.file.close
        counts= dfx['Intervention'].value_counts()[1]

        max_unaffected= dfx['Intervention'].value_counts()[0]
        column_name = 'Intervention'

        condition = dfx[column_name] == 'UNAFFECTED'
        selected_rows = dfx[condition]

        new_value = 'unaffected'
        count_to_reduce = max_unaffected - (max_unaffected-counts)
        count_reduced = 0

        for index, row in selected_rows.iterrows():
            if count_reduced < count_to_reduce:
                dfx.at[index, column_name] = new_value
                count_reduced += 1
            else:
                break
        condition = dfx['Intervention'] != 'UNAFFECTED'
        clinical = dfx[condition]

        clinical = clinical.dropna()
        #return clinical
        return StreamingResponse(
                    iter([clinical.to_csv(index=False)]),
                    media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=clinical_dataset.csv"})
    except Exception as ex:
        print(ex)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

