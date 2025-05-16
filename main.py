"""
Python Week 7 Assignment: Analyzing Data with Pandas and Visualizing Results with Matplotlib
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set the style for prettier plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Task 1: Load and Explore the Dataset
print("TASK 1: LOADING AND EXPLORING THE DATASET")
print("-" * 50)

# Load the dataset with error handling
try:
    # Try to load the dataset
    df = pd.read_csv('world_bank_indicators.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Dataset file not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# Display the first few rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Get information about the dataset structure
print("\nDataset information:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Display column data types
print("\nColumn data types:")
print(df.dtypes)

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Calculate percentage of missing values
missing_percentage = (df.isnull().sum() / len(df)) * 100
print("\nPercentage of missing values per column:")
print(missing_percentage)

# Clean the dataset - handling missing values
# For numerical columns, fill with mean
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)
        print(f"Filled missing values in {col} with mean: {mean_value:.2f}")

# For categorical columns, fill with mode
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    if df[col].isnull().sum() > 0:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
        print(f"Filled missing values in {col} with mode: {mode_value}")

# Verify no missing values remain
print("\nRemaining missing values after cleaning:")
print(df.isnull().sum().sum())

# Task 2: Basic Data Analysis
print("\n\nTASK 2: BASIC DATA ANALYSIS")
print("-" * 50)

# Compute basic statistics for numerical columns
print("\nBasic statistics for numerical columns:")
print(df.describe())

# Assuming 'country' or 'region' is a categorical column for grouping
# Adjust this based on your actual dataset
try:
    # Try to find a categorical column for grouping
    if 'country' in df.columns:
        categorical_col = 'country'
    elif 'region' in df.columns:
        categorical_col = 'region'
    elif 'continent' in df.columns:
        categorical_col = 'continent'
    else:
        # Find the first object column that has less than 30 unique values (likely categorical)
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() < 30:
                categorical_col = col
                break
        else:
            # If no suitable categorical column is found
            categorical_col = df.select_dtypes(include=['object']).columns[0]
    
    # Find a numerical column for aggregation
    numerical_col = df.select_dtypes(include=['int64', 'float64']).columns[0]
    
    print(f"\nGrouping by '{categorical_col}' and calculating mean of '{numerical_col}':")
    grouped_data = df.groupby(categorical_col)[numerical_col].mean().sort_values(ascending=False)
    print(grouped_data)
    
    # Identify patterns or interesting findings
    print("\nInteresting findings:")
    print(f"- Highest {numerical_col}: {grouped_data.index[0]} with value {grouped_data.iloc[0]:.2f}")
    print(f"- Lowest {numerical_col}: {grouped_data.index[-1]} with value {grouped_data.iloc[-1]:.2f}")
    print(f"- Average {numerical_col} across all {categorical_col}s: {grouped_data.mean():.2f}")
    
except Exception as e:
    print(f"Could not perform grouping due to error: {e}")
    print("Please check your dataset structure and modify the code accordingly.")

# Task 3: Data Visualization
print("\n\nTASK 3: DATA VISUALIZATION")
print("-" * 50)

# Create a figure with subplots
plt.figure(figsize=(20, 16))

# 1. Line chart - assuming there's a time-related column
try:
    # Try to find a time-related column
    time_columns = [col for col in df.columns if any(year in col.lower() for year in ['year', 'date', 'time'])]
    
    if time_columns:
        time_col = time_columns[0]
        plt.subplot(2, 2, 1)
        
        # If the time column is already in datetime format
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            time_series_data = df.groupby(time_col)[numerical_col].mean()
            time_series_data.plot(kind='line', ax=plt.gca())
        else:
            # Get the top 5 entities in the categorical column
            top_entities = df[categorical_col].value_counts().nlargest(5).index
            
            # Plot line for each of the top entities
            for entity in top_entities:
                entity_data = df[df[categorical_col] == entity]
                plt.plot(entity_data[time_col], entity_data[numerical_col], label=entity)
            
            plt.legend()
        
        plt.title(f'Trend of {numerical_col} over {time_col}', fontsize=14)
        plt.xlabel(time_col, fontsize=12)
        plt.ylabel(numerical_col, fontsize=12)
        plt.grid(True)
        plt.xticks(rotation=45)
    else:
        # If no time column is found, create a line chart of top N entities
        plt.subplot(2, 2, 1)
        top_10 = grouped_data.head(10)
        plt.plot(top_10.index, top_10.values, marker='o', linewidth=2)
        plt.title(f'Top 10 {categorical_col}s by {numerical_col}', fontsize=14)
        plt.xlabel(categorical_col, fontsize=12)
        plt.ylabel(numerical_col, fontsize=12)
        plt.grid(True)
        plt.xticks(rotation=90)
        
except Exception as e:
    print(f"Could not create line chart due to error: {e}")

# 2. Bar chart
try:
    plt.subplot(2, 2, 2)
    top_10 = grouped_data.head(10)
    plt.bar(top_10.index, top_10.values, color=sns.color_palette("deep", 10))
    plt.title(f'Top 10 {categorical_col}s by Average {numerical_col}', fontsize=14)
    plt.xlabel(categorical_col, fontsize=12)
    plt.ylabel(f'Average {numerical_col}', fontsize=12)
    plt.grid(axis='y')
    plt.xticks(rotation=90)
except Exception as e:
    print(f"Could not create bar chart due to error: {e}")

# 3. Histogram
try:
    plt.subplot(2, 2, 3)
    plt.hist(df[numerical_col].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {numerical_col}', fontsize=14)
    plt.xlabel(numerical_col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y')
except Exception as e:
    print(f"Could not create histogram due to error: {e}")

# 4. Scatter plot
try:
    # Find another numerical column for the scatter plot
    if len(numeric_columns) > 1:
        numerical_col2 = numeric_columns[1]
        
        plt.subplot(2, 2, 4)
        plt.scatter(df[numerical_col], df[numerical_col2], alpha=0.5)
        plt.title(f'Relationship between {numerical_col} and {numerical_col2}', fontsize=14)
        plt.xlabel(numerical_col, fontsize=12)
        plt.ylabel(numerical_col2, fontsize=12)
        plt.grid(True)
        
        # Add a trend line
        z = np.polyfit(df[numerical_col].dropna(), df[numerical_col2].dropna(), 1)
        p = np.poly1d(z)
        plt.plot(df[numerical_col].dropna(), p(df[numerical_col].dropna()), "r--", linewidth=2)
except Exception as e:
    print(f"Could not create scatter plot due to error: {e}")

# Adjust layout and save figure
plt.tight_layout()
plt.savefig('data_visualization.png')
print("\nAll visualizations have been saved to 'data_visualization.png'")

# Show the plots
plt.show()

# Final observations and conclusions
print("\nCONCLUSIONS AND OBSERVATIONS")
print("-" * 50)
print("1. Data Exploration:")
print(f"   - The dataset contains information about {df.shape[0]} entries with {df.shape[1]} different variables.")
print(f"   - We analyzed the relationship between different {categorical_col}s and their {numerical_col}.")

print("\n2. Data Analysis:")
print(f"   - There is significant variation in {numerical_col} across different {categorical_col}s.")
print(f"   - The highest value for {numerical_col} was found in {grouped_data.index[0]}.")
print(f"   - The lowest value for {numerical_col} was found in {grouped_data.index[-1]}.")

print("\n3. Visualizations:")
print("   - The line chart shows trends in the data over time or across categories.")
print("   - The bar chart clearly identifies the top performers in our categorical variable.")
print("   - The histogram reveals the distribution pattern of our primary numerical variable.")
print("   - The scatter plot helps us understand the relationship between two numerical variables.")

print("\nThis analysis provides valuable insights into the patterns and relationships within the world bank indicators dataset.")

