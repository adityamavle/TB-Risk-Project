from pydantic import BaseModel
class simulation(BaseModel):
    size: int
    percentage_alc_only: float
    percentage_dep_only: float 
    percentage_tobacco_only: float
    percentage_alc_dep: float
    percentage_alc_tobacco: float
    percentage_dep_tobacco: float 
    percentage_tobacco_alcoholism_depression: float
    treatment_noth: float
    treatment_1_conditions: float
    treatment_2_conditions: float
    treatment_3_conditions: float
    male: float
    female : float
    low_bmi:float
    normal_bmi: float
    high_bmi: float
    l1_edu: float
    l2_edu: float
    l3_edu: float
    l4_edu: float
    min_age: int
    max_age: int
    mean_age: int
    sd_age: int 
    seed: int