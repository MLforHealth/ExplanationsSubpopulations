# link to file
link = "/scratch/ssd001/home/aparna/explanations-subpopulations/lib/datasets/compas_two_groups.csv"

# columns
columns = ['sex','age','race','juv_fel_count','juv_misd_count','juv_other_count','priors_count','c_charge_degree','low_risk']

# all training columns
train_cols = ['age','juv_fel_count','juv_misd_count','juv_other_count','priors_count','c_charge_degree']

# label column
label = 'low_risk'

# sensitive columns
sensitive_attributes = ["sex","race"]

# whether to use sensitive attributes while training or not
use_sensitive = 0

# whether data already contains splits
already_split = False

# list of all categorical columns
categorical_columns = ['c_charge_degree']

# balanced groups
balance_groups = 0

has_header = True
