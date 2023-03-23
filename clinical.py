from pydantic import BaseModel
class clinical(BaseModel):
    prob_2_2_and_1_1: float
    prob_3_3: float
    prob_2_1: float
    prob_3_2: float
    prob_3_1: float
    prob_1_0: float
    prob_2_0: float
    prob_3_0: float
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
    mean_ttd : float
    std_ttd : float
    size: int