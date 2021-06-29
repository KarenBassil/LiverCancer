import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

#Setting dahsboard title tool
st.set_page_config(layout='wide', page_title='Liver Cancer', page_icon = 'app_icon.png')

#Loading data and adding new columns
df = pd.read_csv('Liver Cancer Dataset.csv')
df['End Age'] = df['Age'] + df['SurvivalInMonths']/12
df['End Age'] = df['End Age'].astype('int')
#Adding year of life lost
df['YLL'] = np.where(df['End Age']>75, 0, (75 - df['End Age']))

st.sidebar.header('Dashboard')
dash = st.sidebar.selectbox('Select a tool', ['Analysis Tool', 'Predictive Tool'])

def descriptive(dataframe):

    #Header Title using html #c8b3af
    html_title = """
    <div style = "background-color:#fcd2ca;padding:1px">
    <h1 style = "color:#4a7d66; text-align:center; font-size:30px">Liver Cancer Analysis Tool</h1>
    </div>
    """
    st.markdown(html_title, unsafe_allow_html=True)

    st.sidebar.header('Settings:')

    #adding option for the histograms and bar chart to visualize the data as a count or percentage
    count_option = st.sidebar.radio('Count Option:', ['Count', 'Percentage'])

    per_option = ''

    #based on the option of count or percentage, a title is given to allow the figures to display the percentage
    if count_option == 'Percentage':
        per_option = 'percent'


    st.sidebar.header('Filter by:')

    #Adding different options for the user to select from
    #user should select the gender type
    gender_type = st.sidebar.selectbox('Gender:', ['Both', 'Male', 'Female'])

    #user should select a range for the age
    age_range = st.sidebar.slider('Age Range:', int(dataframe['Age'].min()), int(dataframe['Age'].max()), 
        (int(dataframe['Age'].min()), int(dataframe['Age'].max())))

    dataframe = dataframe[(dataframe['Age']>=age_range[0]) & (dataframe['Age']<=age_range[1])]

    #user should select the range for the tumor size
    tumor_range = st.sidebar.slider('Tumor Size:',int(dataframe['TumorSize'].min()), int(dataframe['TumorSize'].max()), 
        (int(dataframe['TumorSize'].min()), int(dataframe['TumorSize'].max())))

    dataframe = dataframe[(dataframe['TumorSize']>=tumor_range[0]) & (dataframe['TumorSize']<=tumor_range[1])]


    #Filterting by cancer stages
    stage_option = st.sidebar.checkbox('Cancer Stages Manual Selection')

    if stage_option:
        stages = st.sidebar.multiselect('Cancer Stages:', sorted(dataframe['CancerStages'].unique()))
        dataframe = dataframe[dataframe['CancerStages'].isin(stages)]
    else:
        stages = False


    side1, side2 = st.sidebar.beta_columns(2)

    #filter data by participants with or without diabetes
    diabetes_option = side1.radio('Diabetes:', ['Both', 'With', 'Without'])

    if diabetes_option == 'With':
        dataframe = dataframe[dataframe['Diabetes'] == 'Yes']
    elif diabetes_option == 'Without':
        dataframe = dataframe[dataframe['Diabetes'] == 'No']


    smokers_option = side2.radio('Smokers:', ['Both', 'Yes', 'No'])

    if smokers_option == 'Yes':
        dataframe = dataframe[dataframe['Smokers'] == 'Yes']
    elif smokers_option == 'No':
        dataframe = dataframe[dataframe['Smokers'] == 'No']


    thrombose_option = side1.radio('Thrombose:', ['Both', 'With', 'Without'])

    if thrombose_option == 'With':
        dataframe = dataframe[dataframe['Thrombose'] == 'Yes']
    elif thrombose_option == 'Without':
        dataframe = dataframe[dataframe['Thrombose'] == 'No']


    cirr_option = side2.radio('Cirrhosis:', ['Both', 'With', 'Without'])

    if cirr_option == 'With':
        dataframe = dataframe[dataframe['Cirrhosis'] == 'Yes']
    elif cirr_option == 'Without':
        dataframe = dataframe[dataframe['Cirrhosis'] == 'No']


    difuse_option = side1.radio('Cancer Diffused:', ['Both', 'Yes', 'No'])

    if difuse_option == 'Yes':
        dataframe = dataframe[dataframe['DifuseCancer'] == 'Yes']
    elif difuse_option == 'No':
        dataframe = dataframe[dataframe['DifuseCancer'] == 'No']


    metastatic_option = side2.radio('Metastatic Cancer:', ['Both', 'With', 'Without'])

    if metastatic_option == 'With':
        dataframe = dataframe[dataframe['MetastaticCancer'] == 'Yes']
    elif metastatic_option == 'Without':
        dataframe = dataframe[dataframe['MetastaticCancer'] == 'No']


    #based on the selected gender type, we will assign different variables
    if gender_type=='Both':
        data = dataframe.copy()
        #adding title for the histograms
        cont_title =''
        #saving the gender name
        gender_title = ''
        #extracting the total participants
        num_part = str(len(data)) 
        text_part = 'participants'
        #extracting the mean age
        if len(data['Age']) == 0:
            num_age = 0
        else:
            #extracting the mean age
            num_age = str(int(data['Age'].mean()))

        #extracting the mean tumor size
        if len(data['TumorSize']) == 0:
            num_liver = 0
        else:
            #extracting the mean tumor size
            num_liver = str(int(data['TumorSize'].mean())) + ' mm'
        text_age = 'is the average age'
        text_liver = 'is the average tumor size'

        #extracting the mean survival in days
        if len(data['SurvivalInDays']) == 0:
            num_survival = 0
        else:
            #extracting the mean survival in days
            num_survival = str(int(data['SurvivalInDays'].mean())) + ' days'
        text_survival = 'is the average survival days'

        #extracting the mean years of life lost
        if len(data['YLL']) == 0:
            num_lifelost = 0
        else:
            #extracting the mean years of life lost
            num_lifelost = str(int(data['YLL'].mean())) + ' years'
        text_lifelost = 'is the average years of life lost'


    elif gender_type=='Male':
        #extracting data for only male
        data = dataframe[dataframe['Gender'] == gender_type]
        #adding title for the histograms
        cont_title = ' of Male'
        #saving the gender name
        gender_title = 'Male '
        #extracting the total participants
        num_part = str(len(data)) 
        text_part = 'Male participants'
        #extracting the mean age
        if len(data['Age']) == 0:
            num_age = 0
        else:
            #extracting the mean age
            num_age = str(int(data['Age'].mean()))

        #extracting the mean tumor size
        if len(data['TumorSize']) == 0:
            num_liver = 0
        else:
            #extracting the mean tumor size
            num_liver = str(int(data['TumorSize'].mean())) + ' mm'
        text_age = 'is the average male age'
        text_liver = 'is the average tumor size for a male'

        #extracting the mean survival in days
        if len(data['SurvivalInDays']) == 0:
            num_survival = 0
        else:
            #extracting the mean survival in days
            num_survival = str(int(data['SurvivalInDays'].mean())) + ' days'
        text_survival = 'is the average survival days for a male'

        #extracting the mean years of life lost
        if len(data['YLL']) == 0:
            num_lifelost = 0
        else:
            #extracting the mean years of life lost
            num_lifelost = str(int(data['YLL'].mean())) + ' years'
        text_lifelost = 'is the average years of life lost for a male'


    elif gender_type=='Female':
        #extracting data for only female
        data = dataframe[dataframe['Gender'] == gender_type]
        #adding title for the histograms
        cont_title = ' of Female'
        #saving the gender name
        gender_title = 'Female '
        #extracting the total participants
        num_part = str(len(data))
        text_part = 'Female participants'
        #extracting the mean age
        if len(data['Age']) == 0:
            num_age = 0
        else:
            #extracting the mean age
            num_age = str(int(data['Age'].mean()))
        text_age = 'is the average female age'
        #extracting the mean tumor size

        if len(data['TumorSize']) == 0:
            num_liver = 0
        else:
            #extracting the mean tumor size
            num_liver = str(int(data['TumorSize'].mean())) + ' mm'
        text_liver = 'is the average tumor size for a female'

        #extracting the mean survival in days
        if len(data['SurvivalInDays']) == 0:
            num_survival = 0
        else:
            #extracting the mean survival in days
            num_survival = str(int(data['SurvivalInDays'].mean())) + ' days'
        text_survival = 'is the average survival days for a female'

        #extracting the mean years of life lost
        if len(data['YLL']) == 0:
            num_lifelost = 0
        else:
            #extracting the mean years of life lost
            num_lifelost = str(int(data['YLL'].mean())) + ' years'
        text_lifelost = 'is the average years of life lost for a female'

    col1, col2, col3, col4, col5 = st.beta_columns((1,0.001,3,3,3))

    #Adding a divider between the descrptive numbers and the figures
    for i in range(0,28):
        col2.write('|')

    #displaying an image for the participants
    col1.image('participants.png', use_column_width=True)

    #displaying the number of total participants
    html_part = f"""
    <h3 style = "color:#c9988f; text-align:center; font-size:35px; padding:0px; margin:0px; margin-bottom:10px">{num_part}</h3>
    <h5 style = "color:#4a7d66; text-align:center; font-size:15px; padding:0px; margin:0px">{text_part}</h5>
    """
    col1.markdown(html_part, unsafe_allow_html=True)
    col1.write(' ')

    #displaying an image for the age
    col1.image('Age.png', use_column_width=True)

    #displaying the average age
    html_part = f"""
    <h3 style = "color:#c9988f; text-align:center; font-size:35px; padding:0px; margin:0px; margin-bottom:10px">{num_age}</h3>
    <h5 style = "color:#4a7d66; text-align:center; font-size:15px; padding:0px; margin:0px">{text_age}</h5>
    """
    col1.markdown(html_part, unsafe_allow_html=True)
    col1.write(' ')

    #displaying an image for the liver
    col1.image('liver.png', use_column_width=True)

    #displaying the average of liver size
    html_part = f"""
    <h3 style = "color:#c9988f; text-align:center; font-size:33px; padding:0px; margin:0px; margin-bottom:10px">{num_liver}</h3>
    <h5 style = "color:#4a7d66; text-align:center; font-size:15px; padding:0px; margin:0px">{text_liver}</h5>
    """
    col1.markdown(html_part, unsafe_allow_html=True)
    col1.write(' ')

    #displaying an image for survival
    col1.image('Heart.png', use_column_width=True)

    #displaying the average survival in days
    html_part = f"""
    <h3 style = "color:#c9988f; text-align:center; font-size:33px; padding:0px; margin:0px; margin-bottom:10px">{num_survival}</h3>
    <h5 style = "color:#4a7d66; text-align:center; font-size:15px; padding:0px; margin:0px">{text_survival}</h5>
    """
    col1.markdown(html_part, unsafe_allow_html=True)
    col1.write(' ')

    #displaying an image for life lost
    col1.image('LifeLost.png', use_column_width=True)

    #displaying the average years of life lost
    html_part = f"""
    <h3 style = "color:#c9988f; text-align:center; font-size:33px; padding:0px; margin:0px; margin-bottom:10px">{num_lifelost}</h3>
    <h5 style = "color:#4a7d66; text-align:center; font-size:15px; padding:0px; margin:0px">{text_lifelost}</h5>
    """
    col1.markdown(html_part, unsafe_allow_html=True)

    #starting to build the descriptive part

    #Age distribution
    fig = px.histogram(data, x='Age', nbins=50,
        title = 'Age Distribution' + cont_title,
        color_discrete_sequence =['#4a7d66']*len(data),
        histnorm=per_option)

    #Age distribution figure layout
    fig.update_layout(
        bargap = 0.14,
        title_font_size = 14,
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=True,
            ticks='outside'), 
            margin=dict(t=50,l=5,b=1,r=5),
            plot_bgcolor='white', height = 250, width= 300)

    col3.plotly_chart(fig)

    #Tumor Size distribution
    fig = px.histogram(data, x='TumorSize', nbins=50,
        title = 'Tumor Size Distribution' + cont_title, 
        labels={'TumorSize': 'Tumor Size'},  
        color_discrete_sequence =['#4a7d66']*len(data), histnorm=per_option)

    #Tumor Size distribution figure layout
    fig.update_layout(
        bargap = 0.14,
        title_font_size = 14,
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=True,
            ticks='outside'), 
            margin=dict(t=50,l=5,b=1,r=5),
            plot_bgcolor='white', height = 250, width= 300)

    col4.plotly_chart(fig)

    #Survival in days distribution
    fig = px.histogram(data, x='SurvivalInDays', nbins=50,
        title = 'Survival in Days Distribution' + cont_title,
        labels={'SurvivalInDays': 'Survival In Days'},  
        color_discrete_sequence =['#4a7d66']*len(data), histnorm=per_option)

    #Survival in days distribution figure layout
    fig.update_layout(
        bargap = 0.14,
        title_font_size = 14,
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=True,
            ticks='outside'), 
            margin=dict(t=50,l=5,b=1,r=5),
            plot_bgcolor='white', height = 250, width= 300)

    col5.plotly_chart(fig)

    #based on previous selected count option
    if count_option == 'Count':
        #extracting data based on count
        data_count = data.groupby('CancerStages', as_index=False)['ID'].count()
    elif count_option == 'Percentage':
        #extracting data and adding the percentage values
        data_count = data.groupby('CancerStages', as_index=False)['ID'].count()
        data_count['ID'] = data_count['ID']/sum(data_count['ID'])        

    #Adding colors for the bar chart and assinging the highest bar count with a different color
    col = ['#4a7d66']*len(data_count)
    indx = list(data_count['ID']).index(max(data_count['ID']))
    col[indx] = '#fcd2ca'

    #Cancer stages bar chart
    fig = px.bar(data_count, x='CancerStages', y='ID',
        labels={'CancerStages':'Cancer Stages', 'ID':'Count'},
        title = gender_title + 'Participants by Cancer Stages')

    #Cancer stages bar chart layout settings
    fig.update_layout(title_font_size = 14, 
        yaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        showticklabels=True,
        ticks='outside'), 
        margin=dict(t=30,l=5,b=1,r=5),
        plot_bgcolor='white', height = 230, width=300)

    fig.update_traces(marker_color =col)

    col3.plotly_chart(fig)


    fig = px.histogram(data, x='YLL', nbins=50,
        labels={'YLL':'Years of Life Lost'},
        title = 'Years of Life Lost Distribution' + cont_title, 
        color_discrete_sequence =['#4a7d66']*len(data), histnorm=per_option)

    fig.update_layout(
        bargap = 0.14,
        title_font_size = 14,
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=True,
            ticks='outside'), 
            margin=dict(t=30,l=5,b=1,r=5),
            plot_bgcolor='white', height = 230, width= 300)

    col4.plotly_chart(fig)


    fig1 = px.pie(data, names='Diabetes', values='ID',
                     labels={'ID':'Count of ' + gender_title + ' participants'},  
                     title = gender_title + 'Participants with/without Diabetes',
                     color_discrete_sequence =['#4a7d66','#fcd2ca'])

    fig1.update_traces(textposition='inside', textinfo='percent+label')

    fig1.update_layout(plot_bgcolor='white', height = 230, width=350, 
        title_font_size = 13, showlegend=False, margin=dict(t=50,l=10,b=1,r=50))

    col5.plotly_chart(fig1)


    fig2 = px.pie(data, names='Smokers', values='ID',
                     labels={'ID':'Count of ' + gender_title + ' participants'},  
                     title = gender_title + 'Participants who are/not Smokers',
                     color_discrete_sequence =['#4a7d66','#fcd2ca'])

    fig2.update_traces(textposition='inside', textinfo='percent+label')

    fig2.update_layout(plot_bgcolor='white', height = 230, width=350, 
        title_font_size = 13, showlegend=False, margin=dict(t=50,l=10,b=1,r=50))

    col3.plotly_chart(fig2)


    fig3 = px.pie(data, names='Thrombose', values='ID',
                     labels={'ID':'Count of ' + gender_title + ' participants'},  
                     title = gender_title + 'Participants with/without Thrombose',
                     color_discrete_sequence =['#4a7d66','#fcd2ca'])

    fig3.update_traces(textposition='inside', textinfo='percent+label')

    fig3.update_layout(plot_bgcolor='white', height = 230, width=350, 
        title_font_size = 13, showlegend=False, margin=dict(t=50,l=10,b=1,r=50))

    col4.plotly_chart(fig3)


    fig4 = px.pie(data, names='Cirrhosis', values='ID',
                     labels={'ID':'Count of ' + gender_title + ' participants'},  
                     title = gender_title + 'Participants with/without Cirrhosis',
                     color_discrete_sequence =['#fcd2ca','#4a7d66'])

    fig4.update_traces(textposition='inside', textinfo='percent+label')

    fig4.update_layout(plot_bgcolor='white', height = 230, width=350, 
        title_font_size = 13, showlegend=False, margin=dict(t=50,l=10,b=1,r=50))

    col5.plotly_chart(fig4)


    fig5 = px.pie(data, names='MilanCriteria', values='ID',
                     labels={'ID':'Count of ' + gender_title + ' participants'},  
                     title = gender_title + 'Participants who met/not Milan Criteria',
                     color_discrete_sequence =['#4a7d66','#fcd2ca'])

    fig5.update_traces(textposition='inside', textinfo='percent+label')

    fig5.update_layout(plot_bgcolor='white', height = 230, width=350, 
        title_font_size = 13, showlegend=False, margin=dict(t=50,l=10,b=1,r=50))

    col3.plotly_chart(fig5)


    fig6 = px.pie(data, names='DifuseCancer', values='ID',
                     labels={'ID':'Count of ' + gender_title + ' participants'},  
                     title = gender_title + 'Participants with/without Difuse Cancer',
                     color_discrete_sequence =['#4a7d66','#fcd2ca'])

    fig6.update_traces(textposition='inside', textinfo='percent+label')

    fig6.update_layout(plot_bgcolor='white', height = 230, width=350, 
        title_font_size = 13, showlegend=False, margin=dict(t=50,l=10,b=1,r=50))

    col4.plotly_chart(fig6)


    fig7 = px.pie(data, names='MetastaticCancer', values='ID',
                     labels={'ID':'Count of ' + gender_title + ' participants'},  
                     title = gender_title + 'Participants with/without Metastatic Cancer',
                     color_discrete_sequence =['#4a7d66','#fcd2ca'])

    fig7.update_traces(textposition='inside', textinfo='percent+label')

    fig7.update_layout(plot_bgcolor='white', height = 230, width=350, 
        title_font_size = 13, showlegend=False, margin=dict(t=50,l=10,b=1,r=50))

    col5.plotly_chart(fig7)


def predictive(dataframe):

    #Header Title using html #c8b3af
    html_title = """
    <div style = "background-color:#fcd2ca;padding:1px">
    <h1 style = "color:#4a7d66; text-align:center; font-size:30px">Liver Cancer Predictive Tool</h1>
    </div>
    """
    st.markdown(html_title, unsafe_allow_html=True)

    st.write('')

    #extracting only the needed data features and label
    pred = dataframe[['Gender', 'Age', 'TumorSize', 'Diabetes',
             'Smokers', 'Thrombose', 'Cirrhosis', 'MilanCriteria',
             'CancerStages', 'DifuseCancer', 'MetastaticCancer', 'SurvivalInDays']]

    #getting the features and labels
    features = pred.drop('SurvivalInDays', axis=1)
    label = pred['SurvivalInDays']

    #Encoding categorical variables
    features['Gender'] = features['Gender'].replace({'Male': 0, 'Female': 1})
    features['Diabetes'] = features['Diabetes'].replace({'No': 0, 'Yes':1})
    features['Smokers'] = features['Smokers'].replace({'No': 0, 'Yes':1})
    features['Thrombose'] = features['Thrombose'].replace({'No': 0, 'Yes':1})
    features['Cirrhosis'] = features['Cirrhosis'].replace({'No': 0, 'Yes':1})
    features['MilanCriteria'] = features['MilanCriteria'].replace({'Out': 0, 'In':1})
    features['CancerStages'] = features['CancerStages'].replace({'A': 1, 'B': 2, 'C': 3, 'D': 4})
    features['DifuseCancer'] = features['DifuseCancer'].replace({'No': 0, 'Yes':1})
    features['MetastaticCancer'] = features['MetastaticCancer'].replace({'No': 0, 'Yes':1})

    #Initiating the machine learning model
    lr = LinearRegression()
    #training the model
    lr.fit(features, label) 

    st.write('Input patient\'s private information to predict the survival in days:')

    col1, col2, col3, col4 = st.beta_columns(4)

    #user must select the gender
    patient_gender = col1.selectbox('Gender:', ['Male', 'Female'])
    if patient_gender == 'Male':
        patient_gender = 0
    elif patient_gender == 'Female':
        patient_gender = 1

    #user must select the age
    patient_age = col2.selectbox('Age:', list(range(0,120)))

    #user must select the tumor size
    patient_tumor = col3.selectbox('Tumor Size:', list(range(0,1500)))

    #user must select diabetes option
    patient_diabetes = col4.selectbox('Diabetes:', ['Yes', 'No'])
    if patient_diabetes == 'No':
        patient_diabetes = 0
    elif patient_diabetes == 'Yes':
        patient_diabetes = 1

    #user must select smokers option
    patient_smokers = col1.selectbox('Smokers:', ['Yes', 'No'])
    if patient_smokers == 'No':
        patient_smokers = 0
    elif patient_smokers == 'Yes':
        patient_smokers = 1

    #user must select thrombose option
    patient_thrombose = col2.selectbox('Thrombose:', ['Yes', 'No'])
    if patient_thrombose == 'No':
        patient_thrombose = 0
    elif patient_thrombose == 'Yes':
        patient_thrombose = 1

    #user must select cirrhosis option
    patient_cirrhosis = col3.selectbox('Cirrhosis:', ['Yes', 'No'])
    if patient_cirrhosis == 'No':
        patient_cirrhosis = 0
    elif patient_cirrhosis == 'Yes':
        patient_cirrhosis = 1

    #user must select milan criteris option
    patient_milan = col4.selectbox('Milan Criteria:', ['In', 'Out'])
    if patient_milan == 'Out':
        patient_milan = 0
    elif patient_milan == 'In':
        patient_milan = 1
    
    #user must select difuse cancer option
    patient_difuse = col1.selectbox('Difuse Cancer:', ['Yes', 'No'])
    if patient_difuse == 'No':
        patient_difuse = 0
    elif patient_difuse == 'Yes':
        patient_difuse = 1   

    #user must select metastatic cancer option
    patient_metastatic = col2.selectbox('Metastatic Cancer:', ['Yes', 'No'])
    if patient_metastatic == 'No':
        patient_metastatic = 0
    elif patient_metastatic == 'Yes':
        patient_metastatic = 1  

    #user must select cancer stage
    patient_cancer_stage = col3.selectbox('Cancer Stage:', ['A', 'B', 'C', 'D'])
    if patient_cancer_stage == 'A':
        patient_cancer_stage = 1
    elif patient_cancer_stage == 'B':
        patient_cancer_stage = 2 
    elif patient_cancer_stage == 'C':
        patient_cancer_stage = 3 
    elif patient_cancer_stage == 'D':
        patient_cancer_stage = 4

    #Inserting patient information to be able to predict them
    patient_input = pd.DataFrame(np.array([[patient_gender, patient_age, patient_tumor,
        patient_diabetes, patient_smokers, patient_thrombose, patient_cirrhosis,
        patient_milan, patient_cancer_stage, patient_difuse, patient_metastatic]]), 
        columns = features.columns)
    
    col4.header('')

    predicting = col4.button('Predict Survival In Days')
    #predicting the survival in days
    survivial_prediction = lr.predict(patient_input)

    if predicting:
        st.header('')
        col1, col2 = st.beta_columns((1,1))
        col1.info('The patient is expected to live for: ' + str(int(survivial_prediction[0])) + ' days')

    st.sidebar.header('')
    with st.sidebar.beta_expander('Model Information'):
        coef = pd.DataFrame({'Feature name':features.columns, 'Effect in days': lr.coef_})
        coef['Effect in days'] = [round(cf) for cf in coef['Effect in days']]

        st.table(coef)

#based on selection, the dashboard will be opened
if dash == 'Analysis Tool':
    descriptive(dataframe=df)
elif dash == 'Predictive Tool':
    predictive(dataframe=df)
