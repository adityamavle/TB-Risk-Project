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
from power_analysis import power_analysis
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post('/simulate_dataset')
def create_dataset(data:simulation):

  """
  User Defined Inputs:
  1.size=Population Sample size
  2.percentage_alcoholism/depression/tobacco=Percentage of the the respective condition in the population
  3.percentage_alcoholism_depression(and other dual variable overlaps)= Percentage of the overlap of respective two condtions in the populations
  4.percentage_tobacco_alcoholism_depression= Percentage of the overlap of all 3 cases in the population
  5.age, gender, bmi, edu: A list consisting of the percentages of distribution in the various categorical buckets for the following variables:
  eg; age=[0.3,0.4,0.3] 
  age:3 buckets; gender: 2 buckets; bmi: 3 buckets; edu: 4 buckets
  6.Treatment_noth/treatment_1_conditions=The percentage of good treatment outcomes for the population with the following number of conditions
  """
  try:
    data= data.dict()
    size = data['size']
    percentage_alcoholism = data['percentage_alcoholism']
    percentage_depression = data['percentage_depression'] 
    percentage_tobacco = data['percentage_tobacco']
    percentage_alcoholism_depression = data['percentage_alcoholism_depression']
    percentage_tobacco_alcoholism = data['percentage_tobacco_alcoholism']
    percentage_tobacco_depression = data['percentage_tobacco_depression'] 
    percentage_tobacco_alcoholism_depression = data['percentage_tobacco_alcoholism_depression']
    treatment_noth = data['treatment_noth']
    treatment_1_conditions = data['treatment_1_conditions']
    treatment_2_conditions = data['treatment_2_conditions']
    treatment_3_conditions = data['treatment_3_conditions']
    treatment_intervention = data['treatment_intervention']
    age = data['age']
    gender = data['gender']
    bmi = data['bmi']
    edu = data['edu']
    seed = data['seed']

    if (percentage_alcoholism + percentage_depression  + percentage_tobacco - percentage_alcoholism_depression - percentage_tobacco_alcoholism - percentage_tobacco_depression + percentage_tobacco_alcoholism_depression) > 1:
      raise ValueError("The union of the condition percentages comes out to be more than 1,change parameters accordingly.")
    percentage_alc_only = percentage_alcoholism - percentage_alcoholism_depression-percentage_tobacco_alcoholism + percentage_tobacco_alcoholism_depression

    percentage_dep_only = percentage_depression - percentage_alcoholism_depression-percentage_tobacco_depression + percentage_tobacco_alcoholism_depression

    percentage_tobacco_only = percentage_tobacco - percentage_tobacco_alcoholism - percentage_tobacco_depression + percentage_tobacco_alcoholism_depression

    percentage_alc_dep=percentage_alcoholism_depression-percentage_tobacco_alcoholism_depression
    percentage_alc_tobacco = percentage_tobacco_alcoholism - percentage_tobacco_alcoholism_depression
    percentage_dep_tobacco = percentage_tobacco_depression - percentage_tobacco_alcoholism_depression

    zeros=1-(percentage_alcoholism + percentage_depression  + percentage_tobacco - percentage_alcoholism_depression - percentage_tobacco_alcoholism - percentage_tobacco_depression + percentage_tobacco_alcoholism_depression)

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
    age_1= np.random.choice(3,size,p=[age[0],age[1],age[2]])
    sex=np.random.choice(2,size,p=[gender[0],gender[1]])
    body_mass=np.random.choice(3,size,p=[bmi[0],bmi[1],bmi[2]])
    education=np.random.choice(4,size,p=[edu[0],edu[1],edu[2],edu[3]])
    df = pd.DataFrame(
      {
          'idx': np.arange(1, size+1),
          'age': age_1,
          'gender': sex,
          'bmi': body_mass,
          'education': education,
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
    choices_alc_only=['NA','A']
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
        intervention_arr.append('NO INTER')
    df['Intervention'] = intervention_arr
    #print(np.unique(intervention_arr,return_counts=True))
    df['treatment_outcomes'] = " "
    treatment_outcomes_single_ni = []
    treatment_outcomes_two_ni = []
    treatment_outcomes_three_ni = []
    treatment_outcomes_i = []
    treatment_outcomes_noth = []

    list_noth = list(np.where(df['Intervention'] == 'NO INTER')[0])
    values_noth = np.random.choice(2,len(list_noth),p=[1-treatment_noth,treatment_noth])
    for i in range(len(list_noth)):
      df.loc[list_noth[i],"treatment_outcomes"] = values_noth[i]
    
    list_single_ni = list(np.where((df['Intervention'] == 'NA') | (df['Intervention'] == 'ND') | (df['Intervention'] == 'NT'))[0])
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

    list_i = list(np.where((df['Intervention'] == 'A') | (df['Intervention'] == 'D') | (df['Intervention'] == 'T') | (df['Intervention'] == 'AD') | (df['Intervention'] == 'AT') | (df['Intervention'] == 'DT') | (df['Intervention'] == 'ADT'))[0])
    values_i = np.random.choice(2,len(list_i),p=[1-treatment_intervention,treatment_intervention])
    for i in range(len(list_i)):
      df.loc[list_i[i],"treatment_outcomes"] = values_i[i]
    # print(df.shape())
    print(df.head(10))
    # json = df.to_json(orient = 'records')
    # generate_csv = df.to_csv("generate.csv", index=True)
    # print('\nCSV String:\n', generate_csv)
    # return generate_csv
    return StreamingResponse(
        iter([df.to_csv(index=False)]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=data.csv"})
  except Exception as ex:
    print(ex)
    


@app.post('/power_analysis')
def statspower(data: power_analysis):
 """ x = value of treatment variable (gives us idea of the population - na, a, nt, t, adt, etc)
    y = value of control variable """
 data = data.dict()
 x = data['x']
 y = data['y']
 case_no = data['case_no']
#  csv_file= create_dataset("size","percentage_alcoholism", "percentage_depression", "percentage_tobacco","percentage_alcoholism_depression",  "percentage_tobacco_alcoholism", "percentage_tobacco_depression", "percentage_tobacco_alcoholism_depression","treatment_noth,treatment_1_conditions","treatment_2_conditions","treatment_3_conditions","treatment_intervention","age","gender","bmi","edu","seed")
 df= pd.read_csv(r"D:\tb_api\data.csv")
 power_analysis=TTestIndPower()
#  results=pd.DataFrame({'Effect Size':[np.nan],'Samples':[np.nan],'Power':[np.nan]})
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
#  print(l1)
#  print(l2)
 s = np.sqrt(((l1 - 1) * std1 + (l2 - 1) * std2) / (l1 + l2 - 2))
 d = (mu1 - mu2) / s #cohen's effect size
 effect = round(d,2)
#  print(effect)
 sample_size=power_analysis.solve_power(effect_size=effect,alpha=0.05,power=0.8,alternative='two-sided')
 #power_analysis.solve_power()
 power=power_analysis.power(effect_size=effect,alpha=0.05,nobs1=l1,ratio=(l1/l2),alternative='two-sided')
 #power_analysis.power()
 #power=power_analysis.power()
 t=Counter(treatment_arr[0])
 c=Counter(control_arr[0])
 print('Case No:',case_no)
 print('Outcomes in Treatment array',t)
 print('Outcomes in Control array',c)
 #print(c)
 print('The effect size is =',effect)
 print('The required sample size =',sample_size)
 print('The current statistical power is',power)
 
 graph = power_analysis.plot_power(dep_var='nobs',nobs=np.arange(5, sample_size),
                          effect_size=np.array([effect-(0.2*effect), effect,effect+(0.2*effect)]),
                          alpha=0.05)
 return case_no,t,c,effect,sample_size,power
#  return a,b,c,d,e,f

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)