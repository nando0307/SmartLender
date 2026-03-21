"""Central configuration file. All constants live here."""
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Feature definitions
NUMERICAL_FEATURES = [
    'loan_amnt', 'annual_inc', 'dti', 'open_acc', 'revol_bal',
    'revol_util', 'total_acc', 'installment', 'pub_rec',
    'mort_acc', 'pub_rec_bankruptcies',
]

CATEGORICAL_FEATURES = [
    'term', 'grade', 'sub_grade', 'emp_length', 'home_ownership',
    'verification_status', 'purpose', 'addr_state',
    'initial_list_status', 'application_type',
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Target columns
TARGET_CLASSIFICATION = 'loan_status_binary'   # 0=Fully Paid, 1=Charged Off
TARGET_INTEREST_RATE = 'int_rate'              # float
TARGET_LOSS_AMOUNT = 'charged_off_amount'      # float (only for defaults)

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Temporal split date for Module E
TEMPORAL_SPLIT_DATE = '2017-01-01'  # Train on pre-2017, test on 2017-2018

# Optuna
N_OPTUNA_TRIALS = 50
