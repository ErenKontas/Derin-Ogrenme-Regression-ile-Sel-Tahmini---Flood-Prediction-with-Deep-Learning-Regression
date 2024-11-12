import pandas as pd
import streamlit as st
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Load data and update column names
df = pd.read_csv('train.csv')

# Select dependent and independent variables
x = df.drop(['id', 'FloodProbability'], axis=1)
y = df[['FloodProbability']]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Preprocessing (StandardScaler)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
                                    'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation',
                                    'AgriculturalPractices', 'Encroachments', 'IneffectiveDisasterPreparedness',
                                    'DrainageSystems', 'CoastalVulnerability', 'Landslides', 'Watersheds',
                                    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
                                    'InadequatePlanning', 'PoliticalFactors'])
    ]
)

# Streamlit application
def sel_pred(MonsoonIntensity, TopographyDrainage, RiverManagement, Deforestation, Urbanization,
             ClimateChange, DamsQuality, Siltation, AgriculturalPractices, Encroachments,
             IneffectiveDisasterPreparedness, DrainageSystems, CoastalVulnerability, Landslides,
             Watersheds, DeterioratingInfrastructure, PopulationScore, WetlandLoss,
             InadequatePlanning, PoliticalFactors):
    input_data = pd.DataFrame({
        'MonsoonIntensity': [MonsoonIntensity],
        'TopographyDrainage': [TopographyDrainage],
        'RiverManagement': [RiverManagement],
        'Deforestation': [Deforestation],
        'Urbanization': [Urbanization],
        'ClimateChange': [ClimateChange],
        'DamsQuality': [DamsQuality],
        'Siltation': [Siltation],
        'AgriculturalPractices': [AgriculturalPractices],
        'Encroachments': [Encroachments],
        'IneffectiveDisasterPreparedness': [IneffectiveDisasterPreparedness],
        'DrainageSystems': [DrainageSystems],
        'CoastalVulnerability': [CoastalVulnerability],
        'Landslides': [Landslides],
        'Watersheds': [Watersheds],
        'DeterioratingInfrastructure': [DeterioratingInfrastructure],
        'PopulationScore': [PopulationScore],
        'WetlandLoss': [WetlandLoss],
        'InadequatePlanning': [InadequatePlanning],
        'PoliticalFactors': [PoliticalFactors]
    })
    input_data_transformed = preprocessor.fit_transform(input_data)

    model = joblib.load('Sel.pkl')

    prediction = model.predict(input_data_transformed)

    return float(prediction[0])
    
st.title("Flood Risk Regression Model")
st.write("Enter Input Data")

MonsoonIntensity = st.number_input('MonsoonIntensity', min_value=int(df['MonsoonIntensity'].min()), max_value=int(df['MonsoonIntensity'].max()), step=1)
TopographyDrainage = st.number_input('TopographyDrainage', min_value=int(df['TopographyDrainage'].min()), max_value=int(df['TopographyDrainage'].max()), step=1)
RiverManagement = st.number_input('RiverManagement', min_value=int(df['RiverManagement'].min()), max_value=int(df['RiverManagement'].max()), step=1)
Deforestation = st.number_input('Deforestation', min_value=int(df['Deforestation'].min()), max_value=int(df['Deforestation'].max()), step=1)
Urbanization = st.number_input('Urbanization', min_value=int(df['Urbanization'].min()), max_value=int(df['Urbanization'].max()), step=1)
ClimateChange = st.number_input('ClimateChange', min_value=int(df['ClimateChange'].min()), max_value=int(df['ClimateChange'].max()), step=1)
DamsQuality = st.number_input('DamsQuality', min_value=int(df['DamsQuality'].min()), max_value=int(df['DamsQuality'].max()), step=1)
Siltation = st.number_input('Siltation', min_value=int(df['Siltation'].min()), max_value=int(df['Siltation'].max()), step=1)
AgriculturalPractices = st.number_input('AgriculturalPractices', min_value=int(df['AgriculturalPractices'].min()), max_value=int(df['AgriculturalPractices'].max()), step=1)
Encroachments = st.number_input('Encroachments', min_value=int(df['Encroachments'].min()), max_value=int(df['Encroachments'].max()), step=1)
IneffectiveDisasterPreparedness = st.number_input('IneffectiveDisasterPreparedness', min_value=int(df['IneffectiveDisasterPreparedness'].min()), max_value=int(df['IneffectiveDisasterPreparedness'].max()), step=1)
DrainageSystems = st.number_input('DrainageSystems', min_value=int(df['DrainageSystems'].min()), max_value=int(df['DrainageSystems'].max()), step=1)
CoastalVulnerability = st.number_input('CoastalVulnerability', min_value=int(df['CoastalVulnerability'].min()), max_value=int(df['CoastalVulnerability'].max()), step=1)
Landslides = st.number_input('Landslides', min_value=int(df['Landslides'].min()), max_value=int(df['Landslides'].max()), step=1)
Watersheds = st.number_input('Watersheds', min_value=int(df['Watersheds'].min()), max_value=int(df['Watersheds'].max()), step=1)
DeterioratingInfrastructure = st.number_input('DeterioratingInfrastructure', min_value=int(df['DeterioratingInfrastructure'].min()), max_value=int(df['DeterioratingInfrastructure'].max()), step=1)
PopulationScore = st.number_input('PopulationScore', min_value=int(df['PopulationScore'].min()), max_value=int(df['PopulationScore'].max()), step=1)
WetlandLoss = st.number_input('WetlandLoss', min_value=int(df['WetlandLoss'].min()), max_value=int(df['WetlandLoss'].max()), step=1)
InadequatePlanning = st.number_input('InadequatePlanning', min_value=int(df['InadequatePlanning'].min()), max_value=int(df['InadequatePlanning'].max()), step=1)
PoliticalFactors = st.number_input('PoliticalFactors', min_value=int(df['PoliticalFactors'].min()), max_value=int(df['PoliticalFactors'].max()), step=1)

if st.button('Predict'):
    sel = sel_pred(MonsoonIntensity, TopographyDrainage, RiverManagement, Deforestation, Urbanization,
                   ClimateChange, DamsQuality, Siltation, AgriculturalPractices, Encroachments,
                   IneffectiveDisasterPreparedness, DrainageSystems, CoastalVulnerability, Landslides,
                   Watersheds, DeterioratingInfrastructure, PopulationScore, WetlandLoss,
                   InadequatePlanning, PoliticalFactors)
    st.write(f'The predicted flood probability is: {sel:.2f}')

