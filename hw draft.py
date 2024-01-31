from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris_data = load_iris()

# Create a Pandas DataFrame from the dataset
iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
iris_df['target'] = iris_data.target_names[iris_data.target]

# Display the first few rows of the dataset
print("Iris Dataset:")
print(iris_df.head())

# Calculate and print the number of rows and columns
num_rows, num_columns = iris_df.shape

print(f"The dataset contains {num_rows} rows and {num_columns} columns.")

# Get all values of the last column
last_column_values = iris_df.iloc[:, -1]

# Print distinct values of the last column
distinct_values = last_column_values.unique()

print("Distinct values of the last column:")
print(distinct_values)

# Filter rows where the last column has value "Iris-setosa"
setosa_rows = iris_df[iris_df.iloc[:, -1] == "setosa"]

# Calculate the number of rows
num_setosa_rows = setosa_rows.shape[0]

# Calculate the average value of the first column
average_first_column = setosa_rows.iloc[:, 0].mean()

# Calculate the maximum value of the second column
max_second_column = setosa_rows.iloc[:, 1].max()

# Calculate the minimum value of the third column
min_third_column = setosa_rows.iloc[:, 2].min()

# Print the results
print(f"Number of rows with value 'Iris-setosa': {num_setosa_rows}")
print(f"Average value of the first column: {average_first_column:.2f}")
print(f"Maximum value of the second column: {max_second_column:.2f}")
print(f"Minimum value of the third column: {min_third_column:.2f}")

import matplotlib.pyplot as plt

# Define colors and markers for each distinct value in the last column
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
markers = {'setosa': 'o', 'versicolor': '^', 'virginica': 's'}

# Create a scatter plot
plt.figure(figsize=(10, 6))

for species, group in iris_df.groupby('target'):
    plt.scatter(group.iloc[:, 0], group.iloc[:, 1], label=species, color=colors[species], marker=markers[species])

# Set labels and title
plt.xlabel('First Column')
plt.ylabel('Second Column')
plt.title('Scatter Plot of the First and Second Columns')

# Add legend
plt.legend()

# Show the plot
plt.show()
