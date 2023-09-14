# Mall Customer Segmentation

Customer segmentation using clustering is a valuable application of unsupervised learning. Here's a step-by-step guide to this project:

### **Step 1: Data Collection and Preparation**

1. **Data Collection**: Download the Mall Customer Segmentation Data from the provided **[Dataset Link](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)**.
2. **Data Cleaning and Preprocessing**:
    - Handle any missing values if present.
    - Explore the data to understand its structure.
    - Encode categorical variables (if any).
    - Scale or normalize numerical features if needed.

### **Step 2: Exploratory Data Analysis (EDA)**

1. **Descriptive Statistics**: Get an overview of our data using summary statistics, visualizations, and data plots.
2. **Feature Selection and Engineering**:
    - Identify relevant features for clustering.
    - Consider using techniques like PCA (Principal Component Analysis) for dimensionality reduction if necessary.

### **Step 3: Choosing Clustering Algorithms**

1. **Select Clustering Algorithms**:
    - Start with K-Means for simplicity, but we can also experiment with other clustering algorithms like Hierarchical Clustering or DBSCAN.
2. **Hyperparameter Tuning**:
    - Determine the optimal number of clusters (k) using methods like the Elbow Method or Silhouette Score.

### **Step 4: Apply Clustering**

1. **Implement Clustering**:
    - Use a suitable library like scikit-learn in Python to apply the chosen clustering algorithm to the preprocessed data.

### **Step 5: Interpret Results**

1. **Visualize Clusters**:
    - Create scatter plots or other visualizations to plot the clusters and gain insights into customer segmentation.
2. **Cluster Profiling**:
    - Analyze the characteristics of each cluster to understand the purchasing behavior of different customer segments.

### **Step 6: Validate and Iterate**

1. **Evaluate Cluster Quality**:
    - Use internal metrics like Silhouette Score to assess the quality of your clusters.
2. **Iterate and Refine**:
    - If needed, adjust your approach and parameters based on the evaluation results.

### **Step 7: Business Insights**

1. **Derive Business Insights**:
    - Based on your cluster analysis, draw actionable insights that can inform marketing strategies or other business decisions.
2. **Report and Visualize**:
    - Create a presentation or report summarizing your findings and recommendations.

### **Step 8: Conclusion**

# Data Collection and Cleaning

1. **Document and Reflect**:
    - Document your process, findings, and any challenges you faced. Reflect on what you've learned from the project.
    
    ### **Step 1: Data Collection and Preparation**
    
    1. **Data Collection**:
        - Download the Mall Customer Segmentation Data from the provided **[Dataset Link](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)**.
    2. **Data Cleaning and Preprocessing**:
        - Load the dataset using a library like pandas in Python.
    
    ```python
    import pandas as pd
    
    # Load the dataset
    data = pd.read_csv('Mall_Customers.csv')
    ```
    
    Explore the dataset to understand its structure
    
    ```python
    # Display the first few rows of the dataset
    print(data.head())
    
    # Check for missing values
    print(data.isnull().sum())
    
    # Get summary statistics
    print(data.describe())
    ```
    
    The following output is thrown:
    
    ```python
      	CustomerID  Gender  Age  Annual Income (k$)  Spending Score (1-100)
    0           1    Male   19                  15                      39
    1           2    Male   21                  15                      81
    2           3  Female   20                  16                       6
    3           4  Female   23                  16                      77
    4           5  Female   31                  17                      40
    CustomerID                0
    Gender                    0
    Age                       0
    Annual Income (k$)        0
    Spending Score (1-100)    0
    dtype: int64
           CustomerID         Age  Annual Income (k$)  Spending Score (1-100)
    count  200.000000  200.000000          200.000000              200.000000
    mean   100.500000   38.850000           60.560000               50.200000
    std     57.879185   13.969007           26.264721               25.823522
    min      1.000000   18.000000           15.000000                1.000000
    25%     50.750000   28.750000           41.500000               34.750000
    50%    100.500000   36.000000           61.500000               50.000000
    75%    150.250000   49.000000           78.000000               73.000000
    max    200.000000   70.000000          137.000000               99.000000
    ```
    
    ## Data Scaling
    
    We might need to scale or normalize numerical features to ensure they have a similar range.
    
    ```python
    from sklearn.preprocessing import StandardScaler
    
    # Assuming 'Annual Income' and 'Spending Score' are numerical features
    scaler = StandardScaler()
    data[['Annual Income (k$)', 'Spending Score (1-100)']] = scaler.fit_transform(data[['Annual Income (k$)', 'Spending Score (1-100)']])
    ```
    
    ## Data Exploration and Visualisation
    
    • Visualize the data to gain insights into the distribution of features. You can use histograms, scatter plots, or any other suitable visualization.
    
    ```python
    import matplotlib.pyplot as plt
    
    # Example: Histogram of 'Age'
    plt.hist(data['Age'], bins=20, color='blue', edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Distribution of Age')
    plt.show()
    ```
    
    ![age-histogram.png](Mall%20Customer%20Segmentation%203e16f97a2a84447ab428cfb2ab1806ad/age-histogram.png)
    
    ## Feature Selection and Engineering
    
    1. **Identify Relevant Features**:
        
        In the Mall Customer Segmentation Data, the relevant features are likely to include 'Age', 'Annual Income', and 'Spending Score'. These features are likely to have a significant impact on customer segmentation.
        
    2. **Dimensionality Reduction** :
        
        For the Mall Customer Segmentation Data, considering the relatively small number of features (Age, Annual Income, Spending Score), dimensionality reduction may not be necessary.
        
        Dimensionality reduction techniques like PCA are typically employed when dealing with high-dimensional datasets, which can lead to computational efficiency and potentially improved clustering performance. In this case, with only three features, it's more important to maintain the interpretability of the results.
        
        So, for this project, we can skip the dimensionality reduction step and proceed with clustering using the original features ('Age', 'Annual Income', and 'Spending Score').
        

### **Step 3: Choosing Clustering Algorithms**

Since you're just starting with this project, I recommend beginning with the K-Means clustering algorithm. It's a popular choice for customer segmentation tasks and relatively straightforward to implement.

### **K-Means Clustering**

1. **Selecting the Number of Clusters (k)**:
    
    To determine the optimal number of clusters, you can use techniques like the Elbow Method or Silhouette Score:
    
    - **Elbow Method**:
    
    ```python
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    
    # X is our feature matrix
    
    X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(X)
        distortions.append(kmeanModel.inertia_)
    
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    ```
    
    ![optimal-k-value.png](Mall%20Customer%20Segmentation%203e16f97a2a84447ab428cfb2ab1806ad/optimal-k-value.png)
    
    The plot generated using the Elbow Method is a way to help determine the optimal number of clusters (k) for your K-Means clustering algorithm.
    
    Here's what the plot shows:
    
    - **x-axis (k)**: This represents the number of clusters you are considering.
    - **y-axis (Distortion)**: In the context of the Elbow Method, distortion refers to the average of the squared distances from the cluster centers of the respective clusters. It quantifies how spread out the clusters are. Lower values of distortion are generally better, as they indicate that the points are closer to the centroids of their respective clusters.
    - **The Curve**: The plot shows how the distortion changes as you vary the number of clusters (k). Initially, as we increase k, the distortion will decrease because each point will be closer to its nearest centroid. However, after a certain point, the reduction in distortion starts to slow down, and the curve may start to form an "elbow" shape.
        - **Elbow Point**: The point where the distortion starts to level off and form an "elbow" is considered the optimal number of clusters. This is because adding more clusters beyond this point does not significantly reduce the distortion.
        - In the plot, the "elbow" is the point where the reduction in distortion starts to slow down.
    
    ### **Interpretation:**
    
    - **Choosing k**: In this specific plot, we’d look for the point where the distortion starts to level off. This is where adding more clusters does not lead to a significant reduction in distortion. That point is considered the optimal number of clusters.
    
    k=4 or k=5 would be a good choice for this plot.
    
    # Step 5: Interpret results
    
    Since we've chosen k=4 and applied the K-Means clustering algorithm, you can now proceed to visualise the clusters and interpret the results.
    
    ```python
    data['Cluster'] = cluster_labels
    
    # Visualize clusters
    plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('Customer Segmentation')
    plt.show()
    ```
    
    In this code, we're using 'Annual Income' on the x-axis and 'Spending Score' on the y-axis to visualize the clusters. Each point is colored based on its cluster assignment.
    
    ![customer-segmentation.png](Mall%20Customer%20Segmentation%203e16f97a2a84447ab428cfb2ab1806ad/customer-segmentation.png)
    
    ### **Cluster Profiling:**
    
    Next, we want to analyse the characteristics of each cluster. You can calculate the mean or median values of 'Age', 'Annual Income', and 'Spending Score' for each cluster to get a sense of the typical profile of customers in each segment.
    
    ```python
    cluster_profiles = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].groupby('Cluster').median()
    print(cluster_profiles)
    ```
    
    ```python
    Age  Annual Income (k$)  Spending Score (1-100)
    Cluster                                                  
    0        34.0            0.532082                0.826899
    1        64.0           -0.250391               -0.124229
    2        22.5           -0.193137                0.186343
    3        48.0           -0.097714               -0.357158
    ```
    
    Here's an interpretation of the cluster profiles based on the provided median values:
    
    ### **Cluster Profiles:**
    
    **Cluster 0**:
    
    - Median Age: 34 years
    - Median Annual Income: $53,208
    - Median Spending Score: 82.7
    
    **Cluster 1**:
    
    - Median Age: 64 years
    - Median Annual Income: -$25,039
    - Median Spending Score: -12.4
    
    **Cluster 2**:
    
    - Median Age: 22.5 years
    - Median Annual Income: -$19,313.70
    - Median Spending Score: 18.6
    
    **Cluster 3**:
    
    - Median Age: 48 years
    - Median Annual Income: -$9,771.40
    - Median Spending Score: -35.7
    
    ### **Interpretation:**
    
    - **Cluster 0**: This cluster includes younger customers with relatively high annual income and high spending scores. These are potentially high-value customers who spend a significant portion of their income.
    - **Cluster 1**: This cluster represents older customers with lower annual income and spending scores. They are likely more conservative in their spending habits.
    - **Cluster 2**: This cluster consists of younger customers with lower annual income, but with a relatively higher spending score compared to their income. They might be more budget-conscious but still spend a significant portion of their income.
    - **Cluster 3**: This cluster includes middle-aged customers with moderate to low annual income and negative spending scores. These customers may not spend as much as their income allows.
    
    # Step 6: Fine Tuning and Insights
    
    Now that you have your clusters and their profiles, you can proceed with fine-tuning your segmentation and extracting insights. Here are some suggestions:
    
    1. **Visualizations**: Create additional visualizations to gain a deeper understanding of the clusters. For example, you could plot histograms or box plots of 'Age', 'Annual Income', and 'Spending Score' for each cluster to see the distribution of values within each group.
    2. **Naming Clusters**: Based on the characteristics of each cluster, you can assign meaningful names to them. For example, Cluster 0 could be named "High Spenders", Cluster 1 could be "Low Spenders", and so on.
    3. **Customer Personas**: Create customer personas for each cluster. Describe the typical customer in each segment, including their demographics, spending behavior, and potential preferences.
    4. **Marketing Strategies**: Tailor marketing strategies for each cluster. Consider different approaches to engage and retain customers in each segment.
    5. **A/B Testing**: Implement A/B tests to validate marketing strategies. Experiment with different approaches and measure their effectiveness in converting and retaining customers.
    6. **Customer Feedback**: Collect feedback from customers in each segment to understand their needs and preferences better. This can help refine your marketing strategies.
    7. **Monitoring and Iteration**: Continuously monitor the performance of your strategies and iterate as needed. Customer behavior and preferences may change over time.
    
    # **Step 7: Implement Marketing Strategies**
    
    Based on the insights gained from the customer segmentation, you can now start implementing targeted marketing strategies for each cluster. Here are some tailored strategies you can consider for each segment:
    
    ### Cluster 0 ("High Spenders"):
    
    - Offer premium products or services.
    - Provide personalized recommendations based on their preferences and purchase history.
    - Implement loyalty programs or VIP perks to reward their high spending behavior.
    - Send exclusive offers or promotions to encourage repeat purchases.
    
    ### Cluster 1 ("Low Spenders"):
    
    - Focus on building brand loyalty and trust.
    - Provide value-added services or products at competitive prices.
    - Offer incentives for referrals to expand the customer base.
    - Use content marketing to educate and engage customers.
    
    ### Cluster 2 ("Budget-conscious, Active Spenders"):
    
    - Emphasize value for money and affordability in your offerings.
    - Highlight special deals, discounts, and bundled offers.
    - Engage with them through social media platforms to promote special promotions.
    - Encourage customer reviews and testimonials to build trust.
    
    ### Cluster 3 ("Moderate Income, Low Spenders"):
    
    - Focus on building trust and providing excellent customer service.
    - Offer budget-friendly options and emphasize cost-effectiveness.
    - Provide incentives for upsells or cross-sells to increase average transaction value.
    - Use targeted email campaigns with personalized product recommendations.
    
    # **Conclusion:**
    
    In this project, we successfully conducted customer segmentation using unsupervised learning techniques, specifically K-Means clustering. Here's a recap of the steps we followed:
    
    1. **Data Exploration and Preprocessing**:
        - Loaded and inspected the Mall Customer Segmentation dataset.
        - Checked for missing values and performed basic data summary statistics.
        - Scaled the numerical features for clustering.
    2. **Choosing the Number of Clusters (k)**:
        - Utilized the Elbow Method to determine an appropriate value for k.
        - Chose k=4 based on the analysis.
    3. **Applying K-Means Clustering**:
        - Applied the K-Means clustering algorithm with k=4 to segment customers.
    4. **Interpreting Clusters**:
        - Visualized and interpreted the clusters using scatter plots.
        - Calculated cluster profiles with median values for 'Age', 'Annual Income', and 'Spending Score'.
    5. **Fine-Tuning and Insights**:
        - Further analyzed the clusters through additional visualizations and naming of clusters.
        - Derived insights for targeted marketing strategies.
    6. **Implementing Marketing Strategies**:
        - Formulated tailored marketing strategies for each customer segment.
    
    ### **Final Thoughts:**
    
    Customer segmentation is a powerful tool for businesses to understand their customer base better and target their efforts more effectively. By tailoring marketing strategies to specific customer segments, businesses can improve customer satisfaction, retention, and ultimately, their bottom line.
    
    # Author
    
    Arjith Praison
    
    University of Siegen
