import pandas as pd

file1 = "/home/hisham246/uwaterloo/6_steps/robot_state_pickplace_test_1.csv"
file2 = "/home/hisham246/uwaterloo/6_steps/joint_pos_desired.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

print(f"{file1} has {len(df1)} rows.")
print(f"{file2} has {len(df2)} rows.")