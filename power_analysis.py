from pydantic import BaseModel
class power_analysis(BaseModel):
    x: str
    y: str
    case_no: int

# /*

# {
#   "size": 50000,
#   "percentage_alcoholism": 0.25,
#   "percentage_depression": 0.25,
#   "percentage_tobacco": 0.25,
#   "percentage_alcoholism_depression": 0.1,
#   "percentage_tobacco_alcoholism": 0.1,
#   "percentage_tobacco_depression": 0.1,
#   "percentage_tobacco_alcoholism_depression": 0.05,
#   "treatment_noth": 0.9,
#   "treatment_1_conditions": 0.8,
#   "treatment_2_conditions": 0.7,
#   "treatment_3_conditions": 0.6,
#   "treatment_intervention": 0.85,
#   "age": [
#     0.3,0.4,0.3
#   ],
#   "gender": [
#     0.5,0.5
#   ],
#   "bmi": [
#     0.2,0.5,0.3
#   ],
#   "edu": [
#     0.1,0.2,0.2,0.5
#   ],
#   "seed": 52
# }
# */