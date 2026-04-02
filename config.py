MEV_REL = [
    ('GPR_EU', 'HICP_NRG'),
    ('HICP_NRG','HICP_TOT'),
    ('HICP_TOT', 'EUR_1M'),
    ('EUR_1M', 'LOAN_CORP'),
    ('EUR_1M', 'LOAN_HH'),
    ('HICP_TOT', 'UNEMP'),
    ('CISS_FIN', 'CISS'),
    ('CISS','AAA_1Y'),
    ('GPR_EU','CISS_EQ'),
    ('GPR_EU','CISS_BOND'),
    (['CISS_BOND','CISS_EQ'],'CISS')
]

MODELS = ['VAR','VECM','ARDL']
MIN_LAGS = 0
MAX_LAGS = 12


TRAIN_START_DT = '2004-09-01'
TRAIN_END_DT = '2023-12-31'

# Template for MODEL_DICT
# MODEL_DICT = {
#       'HICP_NRG_model': {
#           'model': None #fitted_model_object,
#           'endogenous': 'HICP_NRG',
#           'exogenous': ['GPR_EU','CISS', 'EUR_1M', ...],
#           'lags_exogenous': [2,3,1,...], #implies GPR_EU(t-1) and GPR_EU(t-2) is used, similarly for CISS
#           'lags_endogenous': 2 #must be non-zero
#           'model_type': either of config.py models ['VAR'| 'BVAR'| 'VECM' | 'ARDL']
#       }
#   }