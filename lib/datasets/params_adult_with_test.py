# link to file
link = "/scratch/ssd001/home/aparna/iclr2021/explanations-subpopulations/lib/datasets/adult_all_cleaned.csv"

# columns
columns = ['age', 'workclass', 'education', 'marital_status', 'relationship',
       'occupation', 'race', 'gender', 'capital_gain', 'capital_loss',
       'hours_per_week', 'income']

# all training columns
train_cols = ['age', 'workclass', 'education', 'marital_status', 'relationship',
       'occupation','capital_gain', 'capital_loss',
       'hours_per_week']

# label column
label = 'income'


# sensitive columns
sensitive_attributes = ["gender","race"]

# whether to use sensitive attributes while training or not
use_sensitive = 0

# whether data already contains splits
already_split = False

# list of all categorical columns
categorical_columns = ['workclass','education','marital_status','relationship',
                      'occupation']

has_header = True