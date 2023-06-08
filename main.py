import streamlit as st
from data import * 
from models import *
from metrics import *
from preprocessing import *
from styling import *

st.set_page_config(page_title="Transformers UI")

st.sidebar.header("Options")
mode_options = ["Light mode", "Dark mode"]
selected_mode = st.sidebar.selectbox("Select mode", mode_options)
if selected_mode == "Dark mode":
  set_dark_mode() 
elif selected_mode == "Light mode":
  set_light_mode()
  
st.sidebar.subheader("Color palette")  
palette_options = ["violet", "plum", "lilac"]
selected_palette = st.sidebar.selectbox("Select palette", palette_options)
set_page_bg_color(selected_palette)

models_and_data = st.container()
col1, col2 = st.columns(2)

with col1:
  st.header("Models")
  model_options = list_models()
  selected_model = st.selectbox("Select a model", model_options)
  if selected_model: 
    view_model_details(selected_model)
  
with col2:
  st.header("Datasets")
  dataset_options = list_datasets()
  selected_dataset = st.selectbox("Select a dataset", dataset_options)
  if selected_dataset:  
    st.text(get_dataset_details(selected_dataset))  
  
if selected_model and selected_dataset:
  metrics = compute_metrics(selected_model, selected_dataset)
  st.plotly_chart(metrics_chart(metrics))
  st.table(metrics_df(metrics))
  
st.sidebar.header("Upload files") 
model_file_uploader = st.sidebar.file_uploader("Select model file")  
if model_file_uploader: 
  save_model_file(model_file_uploader)
  model_options.append(model_file_uploader.name)
  
dataset_file_uploader = st.sidebar.file_uploader("Select dataset file")    
if dataset_file_uploader:  
  save_dataset_file(dataset_file_uploader)
  dataset_options.append(dataset_file_uploader.name)  

if st.sidebar.button("Clear cache"):
  clear_cache()
  model_options = []
  dataset_options = [] 
