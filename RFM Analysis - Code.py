# # RFM Analysis of Retail Customers

# In[16]:


get_ipython().system('pip install squarify')


# In[17]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import squarify
from datetime import datetime


# In[18]:


# Changing our current working directory to match the project file in my pc
os.getcwd()
os.chdir("D:\Projects\RFM Analysis\CSV Files")


# In[19]:


# Importing the csv files into separate data frames
df = pd.read_csv('online_retail0910.csv')
df1 = pd.read_csv('online_retail1011.csv')


# ## Data Preprocessing

# In[20]:


# Checking our data types for each column in our data frames to make sure they are readable by
#our language packs.
# InvoiceDate is an object type that must be converted to datetime.
df1.dtypes


# In[21]:


#Setting our InvoiceDate column to a datetime64 data type and Invoice to a Object data type for 2009-2010 df
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%m/%d/%Y %H:%M')
df['Invoice'] = df['Invoice'].astype('object')

#Setting our InvoiceDate column to a datetime64 data type and Invoice to a Object data type for 2010-2011
df1['InvoiceDate'] = pd.to_datetime(df1['InvoiceDate'], format='%m/%d/%Y %H:%M')
df1['Invoice'] = df1['Invoice'].astype('object')


# # RFM Analysis

# ### 2009-2010 Data Frame

# ### Summary Statistics and Outliers

# In[22]:


numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns].describe([0.01, 0.05, 0.10, 0.20,0.50,0.75, 0.90, 0.95, 0.99]).T


# ### Recency and Frequency Scoring

# In[23]:


# Setting the latest date in the InvoiceDate column and assigning it to a variable
latest_date = df['InvoiceDate'].max()

# Setting the reference date as one day after the latest invoice date.
# The recency metric is a measure of how many days have elapsed since the customer's last transaction. 
# By setting the reference date to one day after the last transaction in the data, 
#we ensure that the most recent transaction has a recency of at least one day.
reference_date = latest_date + pd.Timedelta(days=1)

# Calculate Recency for each transaction as the number of days from the reference date
df['Recency'] = (reference_date - df['InvoiceDate']).dt.days

# Group by Customer ID to calculate the minimum Recency and Frequency
rfm_df = df.groupby('Customer ID').agg({
    'Recency': 'min',
    'Invoice': 'count'
}).rename(columns={'Invoice': 'Frequency'})

# Assign Recency and Frequency scores using quintiles
# The labels are reversed for Recency because a lower Recency is better (more recent)
rfm_df['RecencyScore'] = pd.qcut(rfm_df['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
# Higher frequency is better, so the labels are not reversed
rfm_df['FrequencyScore'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')

# Defining the segmentation map
seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At Risk',
    r'[1-2]5': 'Can\'t Lose',
    r'3[1-2]': 'About to Sleep',
    r'33': 'Need Attention',
    r'[3-4][4-5]': 'Loyal Customers',
    r'41': 'Promising',
    r'51': 'New Customers',
    r'[4-5][2-3]': 'Potential Loyalists',
    r'5[4-5]': 'Champions'
}

# Function to assign segments based on the segmentation map.
def assign_segment(row):
    # Concatenate RecencyScore and FrequencyScore to a string.
    score_str = f"{row['RecencyScore']}{row['FrequencyScore']}"
    for pattern, segment in seg_map.items():
        if re.match(pattern, score_str):
            return segment

# Apply the function to the RFM data frame to create the 'Segment' column.
rfm_df['Segment'] = rfm_df.apply(assign_segment, axis=1)


# First, calculate the total Monetary value for each customer.
monetary_df = df.groupby('Customer ID')['Price'].sum().reset_index()
monetary_df.rename(columns={'Price': 'Monetary'}, inplace=True)

# Merge this with your existing rfm_df to add the Monetary column.
rfm_df = pd.merge(rfm_df, monetary_df, on='Customer ID', how='left')

# Assigning Monetary scores using quintiles (1-5 scoring).
# The labels are not reversed for Monetary (unlike recency) because a higher monetary value is better.
rfm_df['MonetaryScore'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])

# Combining our RFM scores into their own str column, R / F / M
rfm_df["RFMScore"] = (rfm_df['RecencyScore'].astype(str) +
                      rfm_df['FrequencyScore'].astype(str) +
                      rfm_df['MonetaryScore'].astype(str))

rfm_df


# ### 2010-2011 Data Frame

# ### Summary Statistics and Outliers

# In[24]:


numeric_columns1 = df1.select_dtypes(include=['number']).columns
df1[numeric_columns1].describe([0.01, 0.05, 0.10, 0.20, 0.90, 0.95, 0.99]).T


# ### Recency and Frequency Scoring

# In[25]:


# Finding the latest date in the InvoiceDate column and assign it to a variable.
latest_date = df1['InvoiceDate'].max()

# Set the reference date as one day after the latest invoice date.
# The recency metric is a measure of how many days have elapsed since the customer's last transaction. 
# By setting the reference date to one day after the last transaction in the data, 
#we ensure that the most recent transaction has a recency of at least one day.
reference_date = latest_date + pd.Timedelta(days=1)

# Calculate Recency for each transaction as the number of days from the reference date.
df1['Recency'] = (reference_date - df1['InvoiceDate']).dt.days

# Group by Customer ID to calculate the minimum Recency and Frequency.
rfm_df1 = df1.groupby('Customer ID').agg({
    'Recency': 'min',
    'Invoice': 'count'
}).rename(columns={'Invoice': 'Frequency'})

# Assign Recency and Frequency scores using quintiles.
# The labels are reversed for Recency because a lower Recency numerical amount
#is better (more recent).
rfm_df1['RecencyScore'] = pd.qcut(rfm_df1['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')

# Higher frequency is better, so the labels are not reversed.
rfm_df1['FrequencyScore'] = pd.qcut(rfm_df1['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')

# Defining the segmentation map.
seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At Risk',
    r'[1-2]5': 'Can\'t Lose',
    r'3[1-2]': 'About to Sleep',
    r'33': 'Need Attention',
    r'[3-4][4-5]': 'Loyal Customers',
    r'41': 'Promising',
    r'51': 'New Customers',
    r'[4-5][2-3]': 'Potential Loyalists',
    r'5[4-5]': 'Champions'
}

# Function to assign segments based on the segmentation map.
def assign_segment(row):
    # Concatenate RecencyScore and FrequencyScore to a string.
    score_str = f"{row['RecencyScore']}{row['FrequencyScore']}"
    for pattern, segment in seg_map.items():
        if re.match(pattern, score_str):
            return segment

# Apply the function to the RFM DataFrame to create the 'Segment' column.
rfm_df1['Segment'] = rfm_df1.apply(assign_segment, axis=1)

# Assuming you have a 'Price' column in the original df that represents the monetary value of each purchase.
# First, calculate the total Monetary value for each customer.
monetary_df = df1.groupby('Customer ID')['Price'].sum().reset_index()
monetary_df.rename(columns={'Price': 'Monetary'}, inplace=True)

# Merge this with your existing rfm_df1 to add the Monetary column.
rfm_df1 = pd.merge(rfm_df1, monetary_df, on='Customer ID', how='left')

# Assigning Monetary scores using quintiles (1-5 scoring).
# The labels are not reversed for Monetary (unlike recency) because a higher monetary value is better.
rfm_df1['MonetaryScore'] = pd.qcut(rfm_df1['Monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])

# Combining our RFM scores into there own str column, R / F / M.
rfm_df1["RFMScore"] = (rfm_df1['RecencyScore'].astype(str) +
                       rfm_df1['FrequencyScore'].astype(str)+
                       rfm_df1['MonetaryScore'].astype(str))


# In[26]:


rfm_df1


# ## Visualization of Results
# ### 2009-2010 Data Frame

# In[27]:


rfmstats = rfm_df[["Segment", "Recency", "Frequency", "Monetary"]].groupby("Segment").agg(["mean", "median", "count", "std"])
rfmstats.columns = rfmstats.columns.map('_'.join).str.strip('|')


# In[28]:


rfmstats


# In[29]:


rfm_df


# In[30]:


# Define a color map for each segment in or segmentation map
color_map = {
    'Hibernating': 'slategrey',
    'At Risk': 'tomato',
    'Can\'t Lose': 'red',
    'About to Sleep': 'sandybrown',
    'Need Attention': 'gold',
    'Loyal Customers': 'royalblue',
    'Promising': 'lightgreen',
    'New Customers': 'limegreen',
    'Potential Loyalists': 'darkturquoise',
    'Champions': 'navy'
}

# Calculating the count of customers in each segment
segment_counts = rfm_df.groupby('Segment').size().reset_index(name='counts')

# Assigning colors based on the defined color map
colors = [color_map[segment] for segment in segment_counts['Segment']]

# Normalize sizes of the treemap.
sizes = segment_counts['counts'].values
normed_sizes = sizes / sizes.max()

# Plotting our Segmentation map into a treemap.
plt.figure(figsize=(12, 8))
squarify.plot(sizes=normed_sizes, label=segment_counts['Segment'], alpha=0.8, color=colors)
plt.title('Number of Customers by Segment Treemap 2009-2010')
plt.axis('off')
plt.show()


# ### 2010-2011 Data Frame

# In[31]:


rfmstats1 = rfm_df1[["Segment", "Recency", "Frequency", "Monetary"]].groupby("Segment").agg(["mean", "median", "count", "std"])
rfmstats1.columns = rfmstats1.columns.map('_'.join).str.strip('|')


# In[32]:


rfmstats1


# In[33]:


# First, calculate the count of customers in each segment
segment_counts1 = rfm_df1.groupby('Segment').size().reset_index(name='counts')

# Assigning colors based on the defined color map, in the previous code cell.
colors = [color_map[segment] for segment in segment_counts1['Segment']]

# Normalize sizes of the treemap.
sizes = segment_counts1['counts'].values
normed_sizes = sizes / sizes.max()

# Plotting our Segmentation map into a treemap.
plt.figure(figsize=(12, 8))
squarify.plot(sizes=normed_sizes, label=segment_counts1['Segment'], alpha=0.8, color=colors)
plt.title('Number of Customers by Segment Treemap 2010-2011')
plt.axis('off')  # Hide the axes for a cleaner look
plt.show()


# In[34]:


#Exporting our RFM data frames in an excel file to be furth analyzed in SQL and Tableau
rfm_df.to_excel('rfm_df.xlsx')
rfm_df1.to_excel('rfm_df1.xlsx')


# In[ ]:




