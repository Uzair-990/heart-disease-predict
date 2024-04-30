import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def get_clean_data():
    data = pd.read_csv("./data/heart.csv")
    print(data.head())
    return data

def get_scaled_values(input_dict):
    data = get_clean_data()

    x= data.drop(['target'],axis=1)
    
    scaled_dict={}
    for key ,value in input_dict.items():
                max_val = int(x[key].max())
                min_val = int(x[key].min())
                value = int(value)
                
                scaled_value = (value - min_val)/(max_val - min_val)
                scaled_dict[key] = scaled_value

    return scaled_dict

def add_sidebar():
  st.sidebar.header("Health Parameters")
  data = get_clean_data()
  
  input_dict = {}
  
  input_dict["age"] = st.sidebar.slider("Age",min_value=0,max_value=110)
  sex= st.sidebar.selectbox("Sex",["male","female"])
  if sex=="male":input_dict["sex"]=1 
  else: input_dict["sex"]=0
  cp = st.sidebar.selectbox("Chest Pain",["Mild","Moderate","Severe","Critical"])
  if cp=="Mild" : input_dict["cp"] = 1
  elif  cp=="Moderate" : input_dict["cp"]=2
  elif cp=="Severe" : input_dict["cp"]=3
  elif cp=="Critical" : input_dict["cp"]=4   
  input_dict["trestbps"] = st.sidebar.slider("Resting Blood Pressure",min_value=0,max_value=500)
  input_dict["chol"] =st.sidebar.slider("Cholestoral",min_value=0,max_value=1000)
  fbs =st.sidebar.selectbox("Is fasting blood sugar > 120 mg/dl?",["Yes","No"])
  if fbs=='Yes':input_dict["fbs"]=1
  else: input_dict["fbs"]=0
  input_dict["restecg"] =st.sidebar.selectbox("Resting Electrocardiographic results",[0,1,2])
  input_dict["thalach"] =st.sidebar.slider("maximum heart rate achieved",min_value=0,max_value=500)
  exang =st.sidebar.selectbox("Exercise induced angina",["Yes","No"])
  if exang=="Yes":
     input_dict["exang"]=1
  else:input_dict["exang"]=0    
  input_dict["oldpeak"] =st.sidebar.slider("ST depression induced by exercise relative to rest",float(0),data["oldpeak"].max())
  input_dict["slope"] =st.sidebar.selectbox("the slope of the peak exercise ST segment",[0,1,2])
  input_dict["ca"] =st.sidebar.selectbox("number of major vessels (0-3) colored by flourosopy",[0,1,2,3])
  thal =st.sidebar.selectbox("thal",["normal","fixed defect","reversable defect"])
  if thal == "normal": input_dict["thal"]=0
  elif thal == "fixed defect":input_dict["thal"]=1
  elif thal == "fixed reversable defect":input_dict["thal"]=2
  return input_dict
    
def get_radar_chart(input):

  input = get_scaled_values(input)
  categories = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope',"ca",'thal']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
      r=[
        input['age'],input['sex'],input['thal'],
      ],
      theta=categories,
      fill='toself',
      name='Patient Vitals'
   ))
  fig.add_trace(go.Scatterpolar(
      r=[
        input['cp'],input['trestbps'],input['chol'],input['fbs'],input['restecg'],input['thalach'],input['exang'],input['oldpeak'],input['slope'],input['ca']
      ],
      theta=categories,
      fill='toself',
      name='Controlable Vitals'
   ))
  

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
      visible=True,
      range=[0, 1]
     )),
     showlegend=True 
    )
  return fig

def add_predictions(input_data):
  model = pickle.load(open("model/model.pkl",'rb'))
  scaler = pickle.load(open("model/scaler.pkl",'rb'))
  input_array = np.array(list(input_data.values())).reshape(1,-1)
  input_array_scaled = scaler.transform(input_array)
  prediction = model.predict(input_array_scaled)
  st.subheader("Heart Disease Prediction")
  if prediction==1:
     st.write("<span class='diagnosis suffer'>Patient MAY BE suffering from underlying heart disease.</span>",unsafe_allow_html=True)
  else: 
     st.write("<span class='diagnosis nosuffer'>Patient MAY NOT BE suffering from underlying heart disease.</span>",unsafe_allow_html=True)
  
  st.write("Probability of NOT HAVING a heart disease is: ", model.predict_proba(input_array_scaled)[0][0])
  st.write("Probability of HAVING a heart disease is: ", model.predict_proba(input_array_scaled)[0][1])
  st.write("Disclaimer: Our heart disease prediction app provides estimates but may not be entirely accurate. Please consult a doctor for confirmation and personalized advice.")
def main():
  st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon=":heart:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  with open("assets/style.css") as f:
     st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

  input_data = add_sidebar()
   
  print(input_data)
  with st.container():
   st.title("Heart Disease Predictor")
   st.write("Welcome to our Heart Disease Prediction Website! Input your health data, including vital signs and lifestyle habits, for personalized risk assessment. Our advanced algorithms provide valuable insights and guidance for preventive measures. Stay proactive about your heart health with our user-friendly platform.")
  col1 , col2 =  st.columns([4,1])

  with col1:
   radar_chart = get_radar_chart(input_data)
   st.plotly_chart(radar_chart)
  with col2:
    add_predictions(input_data)
if __name__ == '__main__':
  main()  