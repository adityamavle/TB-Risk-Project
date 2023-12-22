from pydantic import BaseModel

class simulation(BaseModel):
    size: int = 10000
    percentage_alc_only: float = 0.08
    percentage_dep_only: float = 0.08
    percentage_tobacco_only: float = 0.08
    percentage_alc_dep: float = 0.04
    percentage_alc_tobacco: float = 0.04
    percentage_dep_tobacco: float = 0.04
    percentage_tobacco_alcoholism_depression: float = 0.03
    treatment_noth: float = 0.95
    treatment_1_conditions: float = 0.90
    treatment_2_conditions: float = 0.85
    treatment_3_conditions: float = 0.80
    male: float = 0.5
    female: float = 0.5
    low_bmi: float = 0.2
    normal_bmi: float = 0.5
    high_bmi: float = 0.3
    l1_edu: float = 0.2
    l2_edu: float = 0.2
    l3_edu: float = 0.2
    l4_edu: float = 0.4
    min_age: int = 18
    max_age: int = 60
    mean_age: int = 35
    sd_age: int = 15
    seed: int = 52
