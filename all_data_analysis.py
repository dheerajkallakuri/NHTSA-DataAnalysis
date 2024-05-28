import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the dataset
df = pd.read_csv('YOUR_DATA_FILE.csv') #ADS/ADAS

column_name = 'Report ID'  # Replace with the actual column name

unique_count = df[column_name].nunique()
total_count = df.shape[0]

if unique_count == total_count:
    print(f'All values in column {column_name} are unique.')
else:
    print(f'Column {column_name} has duplicate values.')

# to check unique values in a column

# Get unique values in the specified column
unique_values = df[column_name].unique()

# Display unique values
print(f'Unique values in column {column_name}:')
for value in unique_values:
    print(value)

# to show which count of each unique value in a column
# Count unique values and create a bar graph
value_counts = df[column_name].value_counts()

# Sort the values by count in descending order
value_counts = value_counts.sort_values(ascending=False)

# Plot the bar graph
plt.figure(figsize=(10, 6))  # Set the figure size as needed
value_counts.plot(kind='bar', color='skyblue')
plt.title(f'Count of Unique Values in {column_name}')
plt.xlabel(column_name)
plt.ylabel('Count')
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility

# Display the bar graph
plt.tight_layout()
# Display the counts on top of the bars
for i, v in enumerate(value_counts):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.show()

# relationship between reporting entity and reporting version
# Group the data by 'Reporting Entity' and 'Report Version', then count the unique values
grouped = df.groupby(['Reporting Entity', 'Report Version'])['Report Version'].count().unstack()

# Plot the data
grouped.plot(kind='barh', stacked=True, figsize=(12, 6))
plt.title('Count of Unique Report Versions by Reporting Entity')
plt.xlabel('Count')
plt.ylabel('Reporting Entity')
plt.legend(title='Report Version', loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(rotation=0)
plt.show()

# Group the data by "Reporting Entity" and count unique values of "Report Version"
grouped = df.groupby('Reporting Entity')['Report Version'].value_counts().unstack(fill_value=0)

# Save the result to a new CSV file
grouped.to_csv('report_version_counts.csv')

# Count the unique values of "Report Entity"
entity_counts = df['Report Entity'].value_counts()

# Create a horizontal bar graph
plt.figure(figsize=(10, 6))
entity_counts.plot(kind='barh')

# Add labels to the bars with counts
for i, count in enumerate(entity_counts):
    plt.text(count, i, str(count), va='center')

# Set labels and title
plt.xlabel('Count')
plt.ylabel('Report Entity')
plt.title('Count of Unique Report Entity')

# Show the bar graph
plt.show()

# Count the unique values of "Report Type"
entity_counts = df['Report Type'].value_counts()

# Create a horizontal bar graph
plt.figure(figsize=(10, 6))
entity_counts.plot(kind='barh')

# Add labels to the bars with counts
for i, count in enumerate(entity_counts):
    plt.text(count, i, str(count), va='center')

# Set labels and title
plt.xlabel('Count')
plt.ylabel('Report Type')
plt.title('Count of Unique Report Type')

# Show the bar graph
plt.show()


# Group the data by 'Reporting Entity' and 'Report Type', then count the unique values
grouped = df.groupby(['Reporting Entity', 'Report Type'])['Report Type'].count().unstack()

# Plot the data
grouped.plot(kind='barh', stacked=True, figsize=(12, 6))
plt.title('Count of Unique Report Types by Reporting Entity')
plt.xlabel('Count')
plt.ylabel('Reporting Entity')
plt.legend(title='Report Type', loc='center', bbox_to_anchor=(1, 0.5))
plt.xticks(rotation=0)
#plt.savefig('adas_reporting_entity_and_report_type.png')
plt.show()

# Group the data by "Reporting Entity" and count unique values of "Report Type"
grouped = df.groupby('Reporting Entity')['Report Type'].value_counts().unstack(fill_value=0)
# Save the result to a new CSV file
grouped.to_csv('adas_Report_Type_counts.csv')

#report submission day plot
column_name="Report Submission Date"

df[column_name] = pd.to_datetime(df[column_name], format='%b-%Y')

# Extract the month-year from the date and create a new column
df['MonthYear'] = df[column_name].dt.to_period('M')

# Group and count unique values
date_counts = df['MonthYear'].value_counts().sort_index()

# Plot the bar graph
plt.figure(figsize=(10, 6))  # Set the figure size as needed
date_counts.plot(kind='bar', color='skyblue')
plt.title(f'Count of Unique Values in {column_name}')
plt.xlabel(column_name)
plt.ylabel('Count')
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility

# Display the bar graph
plt.tight_layout()
# Display the counts on top of the bars
for i, v in enumerate(date_counts):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.show()

# incident date plot
column_name="Incident Date"

df[column_name] = pd.to_datetime(df[column_name], format='%b-%Y')

# Extract the month-year from the date and create a new column
df['MonthYear'] = df[column_name].dt.to_period('M')

# Group and count unique values
date_counts = df['MonthYear'].value_counts().sort_index()

# Plot the bar graph
plt.figure(figsize=(10, 6))  # Set the figure size as needed
date_counts.plot(kind='bar', color='skyblue')
plt.title(f'Count of Unique Values in {column_name}')
plt.xlabel(column_name)
plt.ylabel('Count')
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility

# Display the bar graph
plt.tight_layout()
# Display the counts on top of the bars
for i, v in enumerate(date_counts):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.show()

# days to submit the crash report from the date of incident
# Convert the date columns to datetime format
df['Report Submission Date'] = pd.to_datetime(df['Report Submission Date'], format='%b-%Y')
df['Incident Date'] = pd.to_datetime(df['Incident Date'], format='%b-%Y')

# Calculate the time difference in days
df['Days Between'] = (df['Report Submission Date'] - df['Incident Date']).dt.days

value_counts = df['Days Between'].value_counts().sort_index()

# Plot the bar graph
plt.figure(figsize=(10, 6))  # Set the figure size as needed
value_counts.plot(kind='bar', color='skyblue')
plt.title(f'Count of Unique Values in Days Between')
plt.xlabel('Days Between')
plt.ylabel('Count')
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
# Display the bar graph
plt.tight_layout()
# Display the counts on top of the bars
for i, v in enumerate(value_counts):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.show()

# Display the result
filtered_df = df[df['Days Between'] > 100]
filtered_df[['Report ID', 'Reporting Entity', 'Days Between']].to_csv('adas_diff_incident_reportsub_dates.csv', index=False)


entity_counts = df[['Make','Model']].value_counts()

entity_counts = df[['Make', 'Model']].value_counts().reset_index()
entity_counts.columns = ['Make', 'Model', 'Count']

# Filter the counts based on a condition
filtered_entity_counts = entity_counts[entity_counts['Count'] > 15]

print(filtered_entity_counts)

plt.figure(figsize=(10, 6))
plt.bar(filtered_entity_counts.index, filtered_entity_counts['Count'])
plt.xlabel('Make-Model')
plt.ylabel('Count')
plt.title('Counts of Unique Make and Model Combinations')
plt.xticks(filtered_entity_counts.index, filtered_entity_counts['Make'] + ' - ' + filtered_entity_counts['Model'], rotation=0)
plt.tight_layout()

# Annotate the bar graph with count values
for i, count in enumerate(filtered_entity_counts['Count']):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.show()

# difference between make year and incident year
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('ADAS.csv')

df['Model Year'] = pd.to_datetime(df['Model Year'], format='%Y')
df['Incident Date'] = pd.to_datetime(df['Incident Date'], format='%b-%Y')

# Calculate the time difference in days
df['Days Between'] = (df['Incident Date'] - df['Model Year']).dt.days//365

value_counts = df['Days Between'].value_counts().sort_index()
# Plot the bar graph
plt.figure(figsize=(10, 6))  # Set the figure size as needed
value_counts.plot(kind='bar', color='skyblue')
plt.title(f'Year between model year and incident year')
plt.xlabel('Year count')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate x-axis labels for better visibility

# Display the bar graph
plt.tight_layout()
# Display the counts on top of the bars
for i, v in enumerate(value_counts):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.show()


#get the count of reporting enitites whuch has same vehicel id more than 1
column_name='Same Vehicle ID'

# Use the value_counts() function to count the repeated values
value_counts = df[column_name].value_counts()

value_counts = df[column_name].value_counts().reset_index()
value_counts.columns = [column_name,'Count']

# Filter the counts based on a condition
filtered_entity_counts = value_counts[value_counts['Count'] > 1]
print(filtered_entity_counts)

filtered_data = df[df[column_name].isin(filtered_entity_counts[column_name])]
df_no_duplicates = filtered_data.drop_duplicates(subset=column_name)

# Display the filtered DataFrame
entity_counts = df_no_duplicates['Reporting Entity'].value_counts()

# Create a horizontal bar graph
plt.figure(figsize=(10, 6))
entity_counts.plot(kind='barh')

# Add labels to the bars with counts
for i, count in enumerate(entity_counts):
    plt.text(count, i, str(count), va='center')

# Set labels and title
plt.xlabel('Count')
plt.ylabel('Report Entity')
plt.title('Incidences of Same Vehicle ID > 1')

# Show the bar graph
plt.show()

# stats analysis on mileage
column_name='Mileage'

# Remove rows with empty mileage
df = df[df[column_name].notnull()]

# Convert the mileage column to integers
df[column_name] = df[column_name].astype(int)

# Summary statistics
mileage_stats = df[column_name].describe()

# Median (50th percentile)
median_mileage = df[column_name].median()

# Variance
mileage_variance = df[column_name].var()

# Standard deviation
mileage_std_dev = df[column_name].std()

# Plot a histogram for mileage distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x=column_name, bins=20, kde=True)
plt.title('Mileage Distribution')
plt.xlabel(column_name)
plt.ylabel('Frequency')

# Show the plot
plt.show()

print("Summary Statistics:")
print(mileage_stats)
print("\nMedian Mileage:", median_mileage)
print("Mileage Variance:", mileage_variance)
print("Mileage Standard Deviation:", mileage_std_dev)

# driver opertor pie chart in plotly
column_name='Driver / Operator Type'

# Count the occurrences of each driver/operator type
driver_type_counts = df['Driver / Operator Type'].value_counts().reset_index()
driver_type_counts.columns = ['Driver / Operator Type', 'Count']

# Create a pie chart using Plotly Express
fig = px.pie(driver_type_counts, names='Driver / Operator Type', values='Count')

# Customize the pie chart layout
fig.update_traces(textinfo='percent+label')
fig.update_layout(
    title='Driver / Operator Type Distribution',
    dragmode='select',  # Enable dragging legend
    showlegend=True
)

# Show the interactive plot
fig.show()

#source analysis

# Specify the columns to count 'Y'
source_columns = ['Source - Complaint/Claim', 'Source - Telematics', 'Source - Law Enforcement',
                    'Source - Field Report', 'Source - Testing', 'Source - Media',
                    'Source - Other', 'Source - Other Text']

# Create a new DataFrame with the selected columns
source_data = df[source_columns]
counts = {}

for col in source_columns:
    non_empty_count = df[col].apply(lambda x: x if isinstance(x, str) and x.strip() != '' else None).count()
    print(f"Count of {col}: {non_empty_count}")
    counts[col] = non_empty_count

# Convert the counts to a DataFrame
counts_df = pd.DataFrame(list(counts.items()), columns=['Column', 'Non-Empty Count'])

# Save the results to a CSV file
counts_df.to_csv('ads_source_analysis.csv', index=False)
counts_df = pd.read_csv('ads_source_analysis.csv')

# Create a pie chart
fig = px.pie(counts_df, values='Non-Empty Count', names='Column', title='ADS Source Analysis')

# Show the pie chart
fig.show()

#incident time analysis

# Check for missing values in the 'Incident Time (24:00)' column
missing_values = df['Incident Time (24:00)'].isnull().sum()
print(f"Missing values in 'Incident Time (24:00)': {missing_values}")

# Convert the 'Incident Time (24:00)' column to datetime format
df['Incident Time (24:00)'] = pd.to_datetime(df['Incident Time (24:00)'], format='%H:%M', errors='coerce')

# Extract useful information from the datetime column
df['Hour'] = df['Incident Time (24:00)'].dt.hour
df['Minute'] = df['Incident Time (24:00)'].dt.minute

# Count incidents by hour
hourly_counts = df['Hour'].value_counts().sort_index()

# Visualize the distribution of incident times
plt.figure(figsize=(12, 6))
ax = hourly_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Incidents by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Incident Count')

# Add count labels on top of the bars
for i, count in enumerate(hourly_counts):
    ax.text(i, count + 0.1, str(count), ha='center', va='bottom')

plt.show()

# city analysis
# Check for missing values in the 'City' column
missing_values = df['City'].isnull().sum()
print(f"Missing values in 'City' column: {missing_values}")

# Display unique values and their counts
city_counts = df['City'].value_counts()
city_counts.to_csv('adas_city_analysis.csv')
print(city_counts)

#State analysis
# Count occurrences of each state
state_counts = df['State'].value_counts()

# Save the state counts to a new CSV file
state_counts.to_csv('adas_state_analysis.csv')

# Display the first few rows of the state counts
print(state_counts.head())


#roadtype and surface analysis
# Combine 'Roadway Type' and 'Roadway Surface' columns
combined_road_analysis = df.groupby(['Roadway Type', 'Roadway Surface']).size().reset_index(name='Count')

# Save the combined analysis to a new CSV file
combined_road_analysis.to_csv('adas_combined_road_analysis.csv', index=False)

# Display the first few rows of the combined analysis
print(combined_road_analysis.head())

plt.figure(figsize=(12, 8))
sns.countplot(x='Roadway Type', hue='Roadway Surface', data=df, palette='viridis')

# Display count on top of each bar
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.title('ADAS Roadway Type and Surface Analysis')
plt.xlabel('Roadway Type')
plt.ylabel('Count')
plt.legend(title='Roadway Surface')
plt.show()

#speed limit
# Calculate summary statistics
mean_speed_limit = df['Posted Speed Limit (MPH)'].mean()
median_speed_limit = df['Posted Speed Limit (MPH)'].median()
std_dev_speed_limit = df['Posted Speed Limit (MPH)'].std()

print(f"Mean Speed Limit: {mean_speed_limit}")
print(f"Median Speed Limit: {median_speed_limit}")
print(f"Standard Deviation of Speed Limit: {std_dev_speed_limit}")

plt.figure(figsize=(10, 6))
sns.histplot(df['Posted Speed Limit (MPH)'], bins=20, kde=True, color='skyblue')
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.title('Distribution of Posted Speed Limits')
plt.xlabel('Posted Speed Limit (MPH)')
plt.ylabel('Frequency')
plt.show()

#lignting analysis
# Plot a horizontal bar chart for the 'Lighting' column
plt.figure(figsize=(10, 6))
sns.countplot(y='Lighting', data=df, palette='viridis')

# Add count labels on the right side of each bar
for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                 ha='center', va='center', fontsize=10, color='black', xytext=(10, 0),
                 textcoords='offset points')

plt.title('Distribution of Incidents by Lighting Condition')
plt.xlabel('Frequency')
plt.ylabel('Lighting Condition')
plt.show()

#weather analysis
source_columns = ['Weather - Clear', 'Weather - Snow', 'Weather - Cloudy', 
                  'Weather - Rain', 'Weather - Fog/Smoke','Weather - Severe Wind',
                  'Weather - Unknown',	'Weather - Other',	'Weather - Other Text']
# Create a new DataFrame with the selected columns
source_data = df[source_columns]
counts = {}

for col in source_columns:
    non_empty_count = df[col].apply(lambda x: x if isinstance(x, str) and x.strip() != '' else None).count()
    print(f"Count of {col}: {non_empty_count}")
    counts[col] = non_empty_count

# Convert the counts to a DataFrame
counts_df = pd.DataFrame(list(counts.items()), columns=['Column', 'Non-Empty Count'])

# Save the results to a CSV file
counts_df.to_csv('ads_weather_analysis.csv', index=False)
counts_df = pd.read_csv('ads_weather_analysis.csv')

# Create a pie chart
fig = px.pie(counts_df, values='Non-Empty Count', names='Column', title='ADS Weather Analysis')

# Show the pie chart
fig.show()

# Plot a bar chart for the 'Crash With' column
plt.figure(figsize=(12, 6))
col='Crash With'
df[col].value_counts().plot(kind='bar', color='skyblue')

# Add count labels on top of each bar
for i, count in enumerate(df[col].value_counts()):
    plt.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=10)

plt.title('Distribution of Incidents by Crash With')
plt.xlabel('Crash With')
plt.ylabel('Incident Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#injury report
# Plot a bar chart for the 'Crash With' column
plt.figure(figsize=(12, 6))
col='Highest Injury Severity Alleged'
df[col].value_counts().plot(kind='bar', color='skyblue')

# Add count labels on top of each bar
for i, count in enumerate(df[col].value_counts()):
    plt.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=10)

plt.title('Distribution of Injury Severity Alleged')
plt.xlabel('Injury Severity Alleged')
plt.ylabel('Incident Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# pre-crash movement of crash and subject vehicle
# airbag deeployed for crash ans subject vehicles
# Towed for crash and subject vehicles
# Passengers belted in subject vehicle
# Plot a bar chart for the 'Crash With' column
plt.figure(figsize=(12, 6))
col='CP Pre-Crash Movement'
df[col].value_counts().plot(kind='bar', color='skyblue')

# Add count labels on top of each bar
for i, count in enumerate(df[col].value_counts()):
    plt.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=10)

plt.title('Distribution of Crash Vehicle Pre-Crash Movement')
plt.xlabel('Crash Vehicle Pre-Crash Movement')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# subject vehicle precrash speed analysis
mean_speed_limit = df['SV Precrash Speed (MPH)'].mean()
median_speed_limit = df['SV Precrash Speed (MPH)'].median()
std_dev_speed_limit = df['SV Precrash Speed (MPH)'].std()

print(f"Mean Speed Limit: {mean_speed_limit}")
print(f"Median Speed Limit: {median_speed_limit}")
print(f"Standard Deviation of Speed Limit: {std_dev_speed_limit}")

plt.figure(figsize=(10, 6))
sns.histplot(df['SV Precrash Speed (MPH)'], bins=20, kde=True, color='skyblue')
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.title('Distribution of Posted Speed Limits')
plt.xlabel('Subj Vehicle Precrash Speed (MPH)')
plt.ylabel('Frequency')
plt.show()

# contact area analysis of crash and subject vehicles
source_columns = ['CP Contact Area - Rear Left', 'CP Contact Area - Left', 'CP Contact Area - Front Left',
    'CP Contact Area - Rear', 'CP Contact Area - Top', 'CP Contact Area - Front',
    'CP Contact Area - Rear Right', 'CP Contact Area - Right', 'CP Contact Area - Front Right',
    'CP Contact Area - Bottom', 'CP Contact Area - Unknown',
    'SV Contact Area - Rear Left', 'SV Contact Area - Left', 'SV Contact Area - Front Left',
    'SV Contact Area - Rear', 'SV Contact Area - Top', 'SV Contact Area - Front',
    'SV Contact Area - Rear Right', 'SV Contact Area - Right', 'SV Contact Area - Front Right',
    'SV Contact Area - Bottom', 'SV Contact Area - Unknown']
# Create a new DataFrame with the selected columns
source_data = df[source_columns]
counts = {}

for col in source_columns:
    non_empty_count = df[col].apply(lambda x: x if isinstance(x, str) and x.strip() != '' else None).count()
    print(f"Count of {col}: {non_empty_count}")
    counts[col] = non_empty_count

# Convert the counts to a DataFrame
counts_df = pd.DataFrame(list(counts.items()), columns=['Column', 'Non-Empty Count'])

# Save the results to a CSV file
counts_df.to_csv('ads_CP_SV_contact_area_analysis.csv', index=False)

df = pd.read_csv('ads_CP_SV_contact_area_analysis.csv', header=None, names=['Column', 'Non-Empty Count'])

# Extract Contact Type and create a pivot table
df['Contact Type'] = df['Column'].str.split(' - ', expand=True)[0]
df['Area'] = df['Column'].str.split(' - ', expand=True)[1]
pivot_df = df.pivot_table(index='Area', columns='Contact Type', values='Non-Empty Count', fill_value=0).reset_index()

# Rename columns
pivot_df.columns.name = None
pivot_df.columns = ['Area', 'CP Count', 'SV Count']

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35
bar_positions = range(len(pivot_df))

cp_bars = ax.bar(bar_positions, pivot_df['CP Count'], width=bar_width, label='Crash Vehicle Contact Area')
sv_bars = ax.bar([pos + bar_width for pos in bar_positions], pivot_df['SV Count'], width=bar_width, label='Subj Vehicle Contact Area')

# Add counts on top of the bars
for bar, count in zip(cp_bars, pivot_df['CP Count']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(int(count)), ha='center')

for bar, count in zip(sv_bars, pivot_df['SV Count']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(int(count)), ha='center')

# Set labels and title
ax.set_xticks([pos + bar_width / 2 for pos in bar_positions])
ax.set_xticklabels(pivot_df['Area'])
ax.set_xlabel('Contact Area')
ax.set_ylabel('Count')
ax.set_title('Contact Area Count')
ax.legend()

plt.show()


# crash with passenger vehicle by year
# Assuming your DataFrame is named 'df' and has a column 'Incident Date'
df['Incident Date'] = pd.to_datetime(df['Incident Date'], errors='coerce')

# Extract year from the 'Incident Date'
df['Incident Year'] = df['Incident Date'].dt.year

# Filter rows where 'Vehicle Type' is 'Passenger Car'
passenger_car_crashes = df[df['Crash With'] == 'Passenger Car']

# Count the number of passenger car crashes for each incident year
crash_counts_by_year = passenger_car_crashes['Incident Year'].value_counts().sort_index()

# Plot the results
plt.figure(figsize=(10, 6))
crash_counts_by_year.plot(kind='bar', color='skyblue')
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.title('Number of Passenger Car Crashes by Incident Year')
plt.xlabel('Incident Year')
plt.ylabel('Number of Crashes')
plt.show()

# correalltion between mileage, precrash speed and posted speed limit
numerical_columns = ['Mileage', 'SV Precrash Speed (MPH)', 'Posted Speed Limit (MPH)']

# Create a correlation matrix
correlation_matrix = df[numerical_columns].corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()

#how many accident happened when 'SV Precrash Speed (MPH)', is higher than 'Posted Speed Limit (MPH)'
# Filter the DataFrame to include only rows where 'SV Precrash Speed (MPH)' is higher than 'Posted Speed Limit (MPH)'
speeding_accidents = df[df['SV Precrash Speed (MPH)'] > df['Posted Speed Limit (MPH)']]

# Get the count of accidents
num_speeding_accidents = len(speeding_accidents)

print(f'Number of accidents where SV Precrash Speed is higher than Posted Speed Limit in ADAS: {num_speeding_accidents}')

# Arizona analysis

#city analysis
az_cities_count_df = df[df['State'] == 'AZ ']['City'].value_counts().reset_index()
az_cities_count_df.columns = ['City', 'Count']
print(az_cities_count_df)

#road type
az_cities_count_df = df[df['State'] == 'AZ '].groupby(['Roadway Type', 'Roadway Surface']).size().reset_index(name='Count')
print(az_cities_count_df)

#count of speedinga accidents
speeding_accidents = df[df['State'] == 'AZ '][df['SV Precrash Speed (MPH)'] > df['Posted Speed Limit (MPH)']]
num_speeding_accidents = len(speeding_accidents)
print(f"{num_speeding_accidents}")

#crash with?
az_cities_count_df = df[df['State'] == 'AZ ']['Crash With'].value_counts().reset_index()
az_cities_count_df.columns = ['Crash With', 'Count']
print(az_cities_count_df)


#peak time of accidents
df = pd.read_csv('ADS.csv')

df = df[df['State'] == 'AZ ']
missing_values = df['Incident Time (24:00)'].isnull().sum()
print(f"Missing values in 'Incident Time (24:00)': {missing_values}")

# Convert the 'Incident Time (24:00)' column to datetime format
df['Incident Time (24:00)'] = pd.to_datetime(df['Incident Time (24:00)'], format='%H:%M', errors='coerce')

# Extract useful information from the datetime column
df['Hour'] = df['Incident Time (24:00)'].dt.hour
df['Minute'] = df['Incident Time (24:00)'].dt.minute

# Count incidents by hour
hourly_counts = df['Hour'].value_counts().sort_index()
print(hourly_counts)


# weather analysis
source_columns = ['Weather - Clear', 'Weather - Snow', 'Weather - Cloudy', 
                  'Weather - Rain', 'Weather - Fog/Smoke','Weather - Severe Wind',
                  'Weather - Unknown',	'Weather - Other',	'Weather - Other Text']
# Create a new DataFrame with the selected columns
source_data = df[source_columns]
counts = {}

for col in source_columns:
    non_empty_count = df[col].apply(lambda x: x if isinstance(x, str) and x.strip() != '' else None).count()
    print(f"Count of {col}: {non_empty_count}")
    counts[col] = non_empty_count

# Convert the counts to a DataFrame
counts_df = pd.DataFrame(list(counts.items()), columns=['Column', 'Non-Empty Count'])
print(counts_df)