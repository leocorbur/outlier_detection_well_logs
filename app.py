# Outliers in Well Logs
# Import the required Libraries
import streamlit as st
import lasio as ls
import pandas as pd
import io
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import missingno as msno

def read_las_file(u_file):
    las_file_contents = u_file.read()
    las_file_contents_str = las_file_contents.decode("utf-8")
    las_file_buffer = io.StringIO(las_file_contents_str)
    las = ls.read(las_file_buffer)
    df = las.df()
    df.reset_index(inplace=True)
    df.index = df.index + 1
    return df


# Function to filter by range
def filter_by_range(df, column_ranges):
    filtered_df = df.copy()
    for column, (lower_limit, upper_limit) in column_ranges.items():
        filtered_df = filtered_df.loc[(filtered_df[column] >= lower_limit - 0.01) & (filtered_df[column] <= upper_limit + 0.01)]
    return filtered_df

# Function for data exploration
def explore_data(df):
    st.header('Main Data')
    # Display first and last rows
    st.subheader('First Rows')
    st.write(df.head())
    st.subheader('Last Rows')
    st.write(df.tail())
    st.subheader('Statistics')
    st.write(df.describe())
    dim = st.radio('Data dimension:', ('Rows', 'Columns'), horizontal=True)
    if dim == 'Rows':
        st.write('Number of rows: ', df.shape[0])
    else:
        st.write('Number of columns:', df.shape[1])
    
    # Missingno
    st.subheader('A nullity matrix')
    df_msno = df.copy()
    msno.matrix(df_msno)
    st.pyplot()

    # Columns selection
    st.subheader('Column selection')
    selected_columns = st.multiselect('Select at least 03 columns you want to interact with, and include a depth column for log visualization purposes:', 
                                      list(df.columns))
    
    # Filter the original DataFrame based on the selected columns
    df_filtered = df[selected_columns]

    # Initialize filtered_data
    filtered_data = None

    if selected_columns:
        # Remove rows with null values or impute values
        remove_missing = st.checkbox('Remove rows containing missing values', value=True)
        impute_missing = not remove_missing

        if remove_missing:
            df_filtered = df_filtered.dropna()
        else:
            # Imputation of values
            imputation_method = st.selectbox('Select global imputation method:', ['Mean', 'Median', 'Specific Value', 'Zero'],
                                            help="Choose a method to fill missing values for all selected columns.")

            if imputation_method == 'Zero':
                df_filtered[selected_columns] = df_filtered[selected_columns].fillna(0)
            else:
                for column in selected_columns:
                    if imputation_method == 'Mean':
                        imputer = SimpleImputer(strategy='mean')
                        df_filtered[column] = imputer.fit_transform(df_filtered[[column]])
                    elif imputation_method == 'Median':
                        imputer = SimpleImputer(strategy='median')
                        df_filtered[column] = imputer.fit_transform(df_filtered[[column]])
                    elif imputation_method == 'Specific Value':
                        specific_value = st.number_input(f'Enter the specific value for {column}:')
                        df_filtered[column] = df_filtered[column].fillna(specific_value)



        st.title('Filter Data')
        st.subheader('Filter Values Manually')
        column_ranges = {}
        lower_limit, upper_limit = st.columns(2)

        for selected_column in selected_columns:
            lower_limit_value = lower_limit.number_input(f'{selected_column} min_value:', value=round(df_filtered[selected_column].min(), 2), step=None)
            upper_limit_value = upper_limit.number_input(f'{selected_column} max_value:', value=round(df_filtered[selected_column].max(),2), step=None)
            column_ranges[selected_column] = (lower_limit_value,upper_limit_value)


        # Filter the DataFrame based on the selected range
        filtered_data = filter_by_range(df_filtered, column_ranges)

        log_scale_columns = st.multiselect('Select logarithmic columns:', filtered_data.columns, key='log_scale_columns')
        if log_scale_columns:
            for column in log_scale_columns:
                filtered_data[column] = np.log10(filtered_data[column])

        col1, col2 = st.columns(2)

        with col1:
            st.write("Data Statistic:")
            st.write(df_filtered.describe())

        with col2:
            st.write("Filtered Data Statistics:")
            st.write(filtered_data.describe())

    return filtered_data, selected_columns


# Function for boxplot
def boxplot(df_filtered):
    # Configuration for the red circle in the boxplot
    red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')

    # Create the Streamlit application
    st.title('Boxplots')


    # Create box plots 
    fig, axs = plt.subplots(1, len(df_filtered.columns), figsize=(30, 10))

    for i, ax in enumerate(axs.flat):
        ax.boxplot(df_filtered.iloc[:, i], flierprops=red_circle)
        ax.set_title(df_filtered.columns[i], fontsize=20, fontweight='bold')
        ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout()

    # Show the plot in Streamlit
    st.pyplot(fig)

# Function for scatterplot
def scatter_plot(df_filtered):
    st.title('Scatter Plot')

    x = st.selectbox('x axis:',options=['Select a column'] + selected_columns)
    if x != 'Select a column':
        col1, col2 = st.columns(2)
        with col1:
            xlim_min = st.number_input(f'min value:', value=round(df_filtered[x].min(), 2), step=0.01)
        with col2:
            xlim_max = st.number_input(f'max value:', value=round(df_filtered[x].max(), 2), step=0.01)
        xlim = (xlim_min, xlim_max)

    y = st.selectbox('y axis:', options=['Select a column'] + selected_columns)
    if y != 'Select a column':
        col1, col2 = st.columns(2)
        with col1:
            ylim_min = st.number_input(f'min value:', value=round(df_filtered[y].min(), 2), step=0.01)
        with col2:
            ylim_max = st.number_input(f'max value:', value=round(df_filtered[y].max(), 2), step=0.01)
        ylim = (ylim_min, ylim_max)

    c = st.selectbox('categorical variable:', options=['Select a column'] + selected_columns)
    if c != 'Select a column':
        col1, col2 = st.columns(2)
        with col1:
            vmin = st.number_input(f'min_value:', value=round(df_filtered[c].min(), 2), step=0.01)
        with col2:
            vmax = st.number_input(f'max_value:', value=round(df_filtered[c].max(), 2), step=0.01)

        if vmin and vmax and xlim and ylim:
            fig, ax = plt.subplots()
            scatter = ax.scatter(x=df_filtered[x], y=df_filtered[y], c=df_filtered[c], vmin=vmin, vmax=vmax, cmap='rainbow')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            ax.set_xlabel(x)
            ax.set_ylabel(y)

            ax.set_title('Scatter Plot')
            ax.grid(True)

            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(c, fontsize=12)

            st.pyplot(fig)
        else:
            st.warning('Complete all values correctly to generate the scatter plot.')

    else:
        st.warning('Complete all values correctly to generate the scatter plot.')

# ML models
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

def create_outlier_plot(dataframe, curves_to_plot, depth_curve, outlier_method):
    '''
        dataframe: dataframe
        curves_to_plot: list of columns
        depth_curve : a column from dataframe 
        outlier_method : a column from dataframe
    '''

    num_tracks = len(curves_to_plot)
    
    outlier_shading = dataframe[outlier_method]
    
    fig, ax = plt.subplots(nrows=1, ncols=num_tracks, figsize=(num_tracks*2, 10))
    fig.suptitle(f'{outlier_method}', fontsize=20, y=1.0, fontweight='bold')
    

    for i, curve in enumerate(curves_to_plot):
        
        
        ax[i].plot(dataframe[curve], depth_curve, linewidth=0.5)
        
        ax[i].fill_betweenx(depth_curve, dataframe[curve].min(), 
                            dataframe[curve].max(), where=outlier_shading>= 1, 
                            color='white', alpha=0.2)
        
        ax[i].fill_betweenx(depth_curve, dataframe[curve].min(), 
                            dataframe[curve].max(), where=outlier_shading<= -1, 
                            color='red', alpha=0.2)
        
        ax[i].set_title(curve, fontsize=14)
        ax[i].set_ylim(depth_curve.max(), depth_curve.min())
        ax[i].grid(which='major', color='lightgrey', linestyle='-')

        # Move x-axis labels to the top
        ax[i].xaxis.set_ticks_position('top')
        ax[i].xaxis.set_label_position('top')
        
        if i == 0:
            ax[i].set_ylabel('DEPTH', fontsize=14)
        else:
            plt.setp(ax[i].get_yticklabels(), visible=False)

    plt.tight_layout()
    st.pyplot(fig)



def ml_models(df_filtered):
    # Hyperparameters
    st.title('Hyperparameters')
    st.markdown("Minimal hyperparameters by default:")
    col_if, col_svm, col_lof = st.columns(3)

    # Input para IsolationForest
    with col_if:
        st.subheader("IsolationForest")
        n_estimators_if = st.number_input("n_estimators", min_value=50, max_value=500, value=100, step=50)
        contamination_if = st.number_input("contamination_if", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        random_state_if = st.number_input("random_state", min_value=0, value=42)
        max_features_if = st.number_input("max_features", min_value=1, value=1, step=1)

            # Opción para max_samples
        max_samples_option = st.radio("max_samples", ['auto', 'number'])
        if max_samples_option == 'auto':
            max_samples_if = 'auto'
        else:
            max_samples_if = st.number_input("max_samples", min_value=1, value=100, step=1)

    # Input para OneClassSVM
    with col_svm:
        st.subheader("OneClassSVM")
        nu_svm = st.number_input("nu", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

    # Input para LocalOutlierFactor
    with col_lof:
        st.subheader("LocalOutlierFactor")
        contamination_lof = st.number_input("contamination_lof", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        novelty_lof = st.checkbox("novelty", value=True)

    

    model_IF = IsolationForest(n_estimators=n_estimators_if, max_samples=max_samples_if, 
                           contamination=contamination_if, max_features=max_features_if, 
                           random_state=random_state_if)
    
    model_SVM = OneClassSVM(nu=nu_svm)

    model_lof = LocalOutlierFactor(contamination=contamination_lof, novelty=True)

    st.subheader("Column Evaluation")
    outlier_inputs = st.multiselect('Select columns:', df_filtered.columns)

    # Create a dictionary of our models and their names
    models = {'IF':model_IF, 
         'SVM': model_SVM,
         'LOF': model_lof}
    
    # Loop through each model and create columns containing outlier score
    if outlier_inputs:
        # ML models
        for name, model in models.items():
            print(f'Fitting: {name}')
            model.fit(df_filtered[outlier_inputs])
            df_filtered[f'{name}_outlier_scores'] = model.decision_function(df_filtered[outlier_inputs])
            df_filtered[f'{name}_outlier'] = model.predict(df_filtered[outlier_inputs])

        st.header('ML Models Perfomance')
        st.write(df_filtered.reset_index(drop=True).head(10))

        # Scatterplots
        st.subheader('Scatter Plots')

        x = st.selectbox('X axis:', options=['Select a column'] + selected_columns)
        y = st.selectbox('Y axis:', options=['Select a column'] + selected_columns)

        if x != 'Select a column' and y != 'Select a column':

            for name, model in models.items():

                method = f'{name}_outlier'

                g = sns.FacetGrid(df_filtered, col=method, height=4, hue=method, hue_order=[1, -1])
                g.map(sns.scatterplot, x, y)
                g.fig.suptitle(f'Outlier Method: {name}', y=1.10, fontweight='bold')

                axes = g.axes.flatten()
                axes[0].set_title(f"Outliers\n{len(df_filtered[df_filtered[method] == -1])} points")
                axes[1].set_title(f"Inliers\n{len(df_filtered[df_filtered[method] == 1])} points")

                st.pyplot(plt.show())
        else:
            st.warning('Select columns to deploy the scatter plot')

        # Logs
        st.subheader('Outliers in Well Logs')

        depth = st.selectbox('Select depth column:', selected_columns)
        
        columns_to_select = selected_columns.copy()
        if depth in columns_to_select:
            columns_to_select.remove(depth)

        #model_columns = [column for column in df_filtered.columns if column not in selected_columns]
        outlier_method = ['IF_outlier', 'SVM_outlier', 'LOF_outlier']
        outlier_method = st.selectbox('Select outlier method column:', options=['Select a column'] + outlier_method)

        if depth and outlier_method !='Select a column':
            create_outlier_plot(df_filtered, columns_to_select, df_filtered[depth], outlier_method )
        else:
            st.warning('Select outlier method.')



    else:
        st.warning('Make columns selection')


# Config Setup
st.set_page_config(page_title="Outlier Detection", page_icon="📊", layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Add a title and intro text
st.title('OUTLIERS IN WELL LOGS')
st.text('Welcome to the Outlier Detection Web App. Upload a LAS file to get started.')

# Sidebar setup
st.sidebar.title('Instructions')
st.sidebar.write('1. Upload a LAS file.')
st.sidebar.write('2. Choose an option from the sidebar navigation.')
u_file = st.sidebar.file_uploader('Upload a las file format')

# Sidebar navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select what you want to display:', 
                           ['Explore Data', 'Box Plot', 'Scatterplots', 'ML Models' ])

# Check if file has been uploaded
if u_file is not None:
    st.success('File uploaded successfully!')
    df = read_las_file(u_file)

    # Capture the filtered DataFrame from the explore_data function
    filtered_data, selected_columns= explore_data(df)

else:
    st.warning('Please upload a LAS file to begin.')
    filtered_data = None  # Set filtered_data to None if the file is not uploaded

# Navigation options
if options == 'Explore Data':
    pass  # No need to explicitly call the function as it's already invoked within the condition above
elif options == 'Box Plot':
    if u_file is None:
        st.warning('Please upload a LAS file to view Box Plot.')
    elif not selected_columns:
        st.warning('Select columns first.')
    else:
        boxplot(filtered_data)
elif options == 'Scatterplots':
    if u_file is None:
        st.warning('Please upload a LAS file to view Scatter Plot.')
    elif not selected_columns:
        st.warning('Select columns first.')
    else:
        scatter_plot(filtered_data)

elif options == 'ML Models':
    if u_file is None:
        st.warning('Please upload a LAS file to view Outliers in Well Logs.')
    elif not selected_columns:
        st.warning('Select columns first.')
    else:
        ml_models(filtered_data)


# Footer with additional information or links
st.sidebar.markdown('---')
st.sidebar.subheader('Additional Information')
st.sidebar.write('For more information, please contact me at leocorbur@gmail.com or via ' 
                 '[LinkedIn](https://www.linkedin.com/in/leonelcortez/). ' 
                 'Also, I invite you to see my lastest projects on [GitHub](https://github.com/leocorbur).')