
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import warnings; warnings.simplefilter('ignore')


# In[2]:


datasets_path = '/home/monodeepdas112/container-vrts/AppliedAICourse/Datasets/'
dataset = 'haberman.csv'
haberman = pd.read_csv(datasets_path+dataset)


# ## Description of the dataset

# In[3]:


#Converting to proper dataset column names
haberman.columns = pd.Series(['Age', 'Op_Year', 'Pos_Ax_nodes', 'Survival_Status'])


# In[4]:


#Shape
print('Dataset Shape : ', haberman.shape)

#Columns in the dataset
print('\n\nColumns in the dataset : ', haberman.columns)


# In[5]:


#Information about the dataset
print('\n\nInformation about the dataset : ', haberman.info())

#Description of dataset
print('\nDescription :\n', haberman.describe())


# As we can see there are no nulls present in the database

# ### Conclusions from dataset description

# #### 1. Total number of samples is 305 without any null values
# #### 2. There are 4 Attributes Age, Operation Year, Number of Axil nodes found, Surival Status
# #### 3. Relevant statistical indormation can be found out in the cell above

# ### Scatter Plot Visualization

# In[6]:


sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="Survival_Status", size=4)    .map(plt.scatter, "Age", "Pos_Ax_nodes")    .add_legend();
plt.show();


# #### Not much sense can be made out of this plot as the scatter plot seems to show no particular pattern so we can say that there exists no relation between age of the patient and the number of positive axill nodes found.

# In[7]:


sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="Survival_Status", size=4)    .map(plt.scatter, "Op_Year", "Pos_Ax_nodes")    .add_legend();
plt.show();


# #### Even this plot is almost evenly spread with no particular pattern

# In[8]:


sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="Survival_Status", size=4)    .map(plt.scatter, "Age", "Survival_Status")    .add_legend();
plt.show();


# #### Same as we cannot form any kind of inference from the plot as both groups of patients share the common a large common range of age

# In[9]:


sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="Survival_Status", size=4)    .map(plt.scatter, "Pos_Ax_nodes", "Survival_Status")    .add_legend();
plt.show();


# #### Scatter plot does show a density of class 1 nearer to 0 where are the data points for class 2 seem more widely spread but not much of inference can be drawn as we still see a large common range of Pos_Ax_nodes which are occupied by both classes

# ### Pair-Plot

# In[10]:


plt.close();
sns.set_style("whitegrid");
sns.pairplot(haberman, hue="Survival_Status", size=3, vars=['Age', 'Op_Year', 'Pos_Ax_nodes']);
plt.show()


# ## PDF, CDF Analysis

# In[11]:


sns.FacetGrid(haberman, hue="Survival_Status", size=5)    .map(sns.distplot, "Pos_Ax_nodes")    .add_legend();
plt.show();


# In[12]:


sns.FacetGrid(haberman, hue="Survival_Status", size=5)    .map(sns.distplot, "Age")    .add_legend();
plt.show();


# In[13]:


sns.FacetGrid(haberman, hue="Survival_Status", size=5)    .map(sns.distplot, "Op_Year")    .add_legend();
plt.show();


# In[14]:


counts, bin_edges = np.histogram(haberman['Pos_Ax_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.xlabel('Pos_Ax_nodes')
plt.title('PDF, CDF of Num of positive axill nodes found')
plt.legend(labels=['PDF', 'CDF'])
plt.show();


# In[15]:


counts, bin_edges = np.histogram(haberman['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.xlabel('Age')
plt.title('PDF, CDF of Age of the patients when they were operated upon')
plt.legend(labels=['PDF', 'CDF'])
plt.show();


# In[16]:


counts, bin_edges = np.histogram(haberman['Op_Year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.xlabel('Op_Year')
plt.title('PDF, CDF of Year of operation')
plt.legend(labels=['PDF', 'CDF'])
plt.show();


# #### Q. How many people lived more than 5 years and how many less than that ?

# In[17]:


sns.distplot(haberman.Survival_Status, kde=False, rug=True);


# #### Approx 225 people who lived more than 5 years after their operation and approx 75 people who lived less than 5 years

# #### Q. How many Number of Axil nodes were found for the people who lived more than 5 years ?

# In[18]:


long_lived=haberman.loc[haberman['Survival_Status'] == 1]
print('Number of people who lived for more than 5 years : \n\n', long_lived.describe())

short_lived=haberman.loc[haberman['Survival_Status']==2]
print('\n\nNumber of people who lived for less than 5 years : \n', short_lived.describe())


# #### Mean Number of Positive Axil Nodes : 7.456 for the people who could not live for more than 5 years
# #### Mean Number of Positive Axil Nodes : 2.799 for the people who could live for more than 5 years
# #### Gives us a clear picture of the spread of the disease and thus (intuitively) can be used as a important reason for the early death of the people as in their case the disease spread a lot more.

# In[19]:


#Finding out the medians
print('Median Number of Positive Axil Nodes found for class 1 : ', np.median(long_lived))
print('Median Number of Positive Axil Nodes found for class 2 : ', np.median(short_lived))


# #### Observations are confusing as the median of both the classes differ only by 4 so it cannot be the spread of the disease that might have been the crucial reason behind their deaths. Next Step trying out Box and Violin plots so understand more about the density of the Number of positive Axial Nodes found

# #### Trying out box plot to see the information visually and to determine the difference between the variance of the Num_Ax_nodes between both classes

# In[20]:


sns.boxplot(y='Pos_Ax_nodes', x='Survival_Status',data=haberman)
plt.show()


# In[21]:


#Denser regions are darther and sparser regions are thinner
sns.violinplot(y='Pos_Ax_nodes', x='Survival_Status', size=10, data=haberman)
plt.show()


# #### As is clearly evident from the data is that 75 percentile of the people who could not live more than 5 years had 10 positive axial nodes where are the 75 percentile of the people who could live longer than 5 years is very much lesser (approx <5)

# #### As is also evident from the violin plots is that the median is really very close to the 25 percentile in case 1 and the density of people having lesser number of the positive nodes is higher and is below 10.
# #### But there are more people with more positive nodes in class 2 which can be a reason why most of them could not live for more than 5 years.
# #### Although there are quite some outliers in class 1 who survived to live more than 5 years despite having more number of positive axial nodes

# In[22]:


sns.jointplot(y='Age', x='Pos_Ax_nodes', data=long_lived, kind='kde')


# In[23]:


sns.jointplot(y='Age', x='Pos_Ax_nodes', data=short_lived, kind='kde')


# #### The most dense region falls around 50 for Age in both the categories so is the same for both classes but as is evident from the contour plot is that the number positive axill nodes is widely spread for the short lived people which indicates an increased number of positive lymph axill nodes for all the age groups.

# ## Final Conclusion
# #### The given dataset is an imbalanced dataset.
# #### It is not linearly seperable using a simple if-else construct.
# #### The Number of Positive Axill nodes says a lot more about the story as it shows the extent of the spread of the disease so intuitively it is something which helps determining the life expectancy as it is shown above in the data exploratory analysis.
