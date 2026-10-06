import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


df = pd.read_csv('agent_class_label_info_1_new.csv')

print("*** STEP 1 ***")
print("Dataset has been loaded successfully.")
print()
print(df)


print("\n*** STEP 2 ***")

print("\n>> First 8 rows:")
print(df.head(8))

print("\n>> First 12 rows:")
print(df.head(12))

print("\n>> First 25 rows:")
print(df.head(25))


print("\n*** STEP 3 ***")

for idx in range(min(5, len(df))):
    print(f"\n--- Row index: {idx} ---")
    current_row = df.iloc[idx]
    for column_name in df.columns:
        val = current_row[column_name]
        print(f"    Column -> {column_name} | Value -> {val} | Type -> {type(val)}")


print("\n*** STEP 4 ***")
print("Data types of all columns:")
print(df.dtypes)


print("\n*** STEP 5 ***")
print("Column names in the dataframe:")
print(df.columns.tolist())


print("\n*** STEP 6 ***")

df_new = df.copy()

int_cols = df_new.select_dtypes(include=['int64', 'int32']).columns
if len(int_cols) > 0:
    df_new[int_cols] = df_new[int_cols].astype(float)
    print(f"\nInteger -> Float: {list(int_cols)}")
else:
    print("\nNo integer columns were found.")

float_cols = df_new.select_dtypes(include=['float64', 'float32']).columns
if len(float_cols) > 0:
    df_new[float_cols] = df_new[float_cols].astype('Int64')
    print(f"Float -> Integer: {list(float_cols)}")
else:
    print("No float columns were found.")

print("\nUpdated data types:")
print(df_new.dtypes)

print("\nDataFrame after type conversion:")
print(df_new)


print("\n*** STEP 7 ***")
print("Non-numeric columns only:")
df_text_only = df.select_dtypes(exclude=['number'])
print(df_text_only)

print("\n*** STEP 8 ***")

print("\n-> id > 3")
print(df[df['id'] > 3])

print("\n-> id >= 3")
print(df[df['id'] >= 3])

print("\n-> id < 4")
print(df[df['id'] < 4])

print("\n-> id <= 4")
print(df[df['id'] <= 4])

print("\n-> id == 4")
print(df[df['id'] == 4])

print("\n*** STEP 9 ***")

grouped = df.groupby('class')['id']

print("\nMean:")
print(grouped.mean())

print("\nMaximum:")
print(grouped.max())

print("\nMinimum:")
print(grouped.min())

print("\nCount:")
print(grouped.count())


print("\n*** STEP 10 ***")
print("Rows 1-5, columns 1st/2nd/4th/6th:")

target_cols = [0, 1, 3, 5]
valid_cols = [c for c in target_cols if c < len(df.columns)]
print(df.iloc[0:5, valid_cols])


print("\n*** STEP 11 ***")
print("Rows 3 through 10, first 4 columns:")
print(df.iloc[2:10, 0:4])


print("\n*** STEP 12 ***")
result = df[df['id'] > 3.0]
print("Filtered rows where id is greater than 3.0:")
print(result)


print("\n*** STEP 13 ***")
print("Slicing using loc (index + column names):")
print(df.loc[2:5, ['id', 'class']])


print("\n*** STEP 14 ***")
print("Slicing using iloc (index only):")
print(df.iloc[2:6])


print("\n*** STEP 15 ***")

for col_name in df.columns:
    print(f"\nNull values in column '{col_name}':")
    print(df[col_name].isnull())


print("\n*** STEP 16 ***")
print("Total null values per column:")
print(df.isnull().sum())


print("\n*** STEP 17 ***")

cols_with_null = df.columns[df.isnull().any()]
print("Columns that contain null values:")
print(list(cols_with_null))

df_filled_mean = df.copy()
if 'id' in df_filled_mean.columns:
    mean_val = df_filled_mean['id'].mean()
    df_filled_mean['id'] = df_filled_mean['id'].fillna(mean_val)
    print(f"\nFilled with mean value ({mean_val}):")
    print(df_filled_mean)

df_filled_min = df.copy()
if 'id' in df_filled_min.columns:
    min_val = df_filled_min['id'].min()
    df_filled_min['id'] = df_filled_min['id'].fillna(min_val)
    print(f"\nFilled with minimum value ({min_val}):")
    print(df_filled_min)

df_filled_max = df.copy()
if 'id' in df_filled_max.columns:
    max_val = df_filled_max['id'].max()
    df_filled_max['id'] = df_filled_max['id'].fillna(max_val)
    print(f"\nFilled with maximum value ({max_val}):")
    print(df_filled_max)


print("\n*** STEP 18 ***")

output_file = 'agent_class_label_info_1_new_TarikEmirYilmaz_2321051014.csv'
df_filled_mean.to_csv(output_file, index=False)
print(f"File exported successfully: {output_file}")


print("\n*** STEP 19 ***")

df_chart = pd.read_csv(output_file)

# Bar Chart
bar_data = df_chart.groupby('class')['id'].mean()
plt.figure(figsize=(10, 5))
plt.bar(bar_data.index, bar_data.values, color='skyblue', edgecolor='black')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Class')
plt.ylabel('Average ID')
plt.title('Bar Chart - Average ID by Class')
plt.tight_layout()
plt.savefig('bar_chart.png')
plt.close()
print("Bar chart saved.")

# Pie Chart
pie_data = df_chart['class'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart - Class Distribution')
plt.tight_layout()
plt.savefig('pie_chart.png')
plt.close()
print("Pie chart saved.")

# Histogram
plt.figure(figsize=(8, 5))
plt.hist(df_chart['id'], bins=5, color='green', edgecolor='black')
plt.xlabel('ID Values')
plt.ylabel('Frequency')
plt.title('Histogram - ID Value Distribution')
plt.tight_layout()
plt.savefig('histogram.png')
plt.close()
print("Histogram saved.")

# Line Graph
line_data = df_chart.groupby('class')['id'].mean()
plt.figure(figsize=(10, 5))
plt.plot(line_data.index, line_data.values, marker='o', color='orange',
         linestyle='-', linewidth=2, markersize=8)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Class')
plt.ylabel('Average ID')
plt.title('Line Graph - ID Change by Class')
plt.tight_layout()
plt.savefig('line_graph.png')
plt.close()
print("Line graph saved.")

print("\nAll visualizations have been created successfully.")
