# Color palette options 
PALETTE_OPTIONS = {
    'violet': '#9400D3', 
    'plum': '#81007F',
    'lilac': '#D8BFD8',
    'mauve': '#E0B0FF',
    'lavender': '#B57EDC', 
    'eggplant': '#311464',
    'blueberry': '#46438A',
    'navy': '#000080',
    'cobalt': '#0047AB', 
    'cerulean': '#007BA7',
    'sapphire': '#0A0792',
    'teal': '#01837B',
    'emerald': '#357C3C', 
    'olive': '#3D9970',
    'moss': '#88883D',
    'pear': '#D1E231',
    'lemon': '#FDFF38',
    'daffodil': '#FFF000',
    'gold': '#FFD700',
    'orange': '#FFA500',
    'peach': '#FFE5B4',
    'cantaloupe': '#FF8C00', 
    'pumpkin': '#FF7518',
    'ruby': '#922B3E',
    'burgundy': '#9B2335',
    'merlot': '#730050',
    'chestnut': '#954535', 
    'sepia': '#704214',
    'tan': '#D2B48C',
    'sand': '#E6FFC2',
    'ivory': '#FFFFF0'
}  

# Dark mode colors  
DARK_BG_COLOR = '#333333'
DARK_TEXT_COLOR = '#FFFFFF'

# Light mode colors    
LIGHT_BG_COLOR = '#FFFFFF'
LIGHT_TEXT_COLOR = '#333333'   

def set_page_bg_color(color):
    """Set the page background color"""
    bg_color = PALETTE_OPTIONS[color]
    style = f"""<style> 
               .stApp {{ background-color: {bg_color}; }} 
               .css-1aumxhk {{ background-color: {bg_color}; }}  
               .css-1g3nvew  {{ background-color: {bg_color}; }} 
               .css-2k4xy0 {{ background-color: {bg_color}; }} 
               .css-1nkbm6r {{ background-color: {bg_color}; }} 
               .css-z59ceb  {{ background-color: {bg_color}; }}
               </style>
               """
    st.markdown(style, unsafe_allow_html=True) 

DARK_BG_COLOR = '#333333'
DARK_TEXT_COLOR = '#FFFFFF'

def set_dark_mode(): 
    """Set dark mode styling"""
    dark_mode_style = f""" 
                 <style> 
                 .stApp {{ background-color: {DARK_BG_COLOR}; color: {DARK_TEXT_COLOR}; }} 
                 .css-17iph5v {{ color: {DARK_TEXT_COLOR}; background-color: {DARK_BG_COLOR}; }}  
                 .css-ykziq8  {{ color: {DARK_TEXT_COLOR}; background-color: {DARK_BG_COLOR}; }} 
                 .css-1g3nvew  {{ color: {DARK_TEXT_COLOR}; background-color: {DARK_BG_COLOR}; }}  
                 .css-2k4xy0 {{ color: {DARK_TEXT_COLOR}; background-color: {DARK_BG_COLOR}; }}  
                 .css-1nkbm6r {{ color: {DARK_TEXT_COLOR}; background-color: {DARK_BG_COLOR}; }}
                 .css-hp2m1f {{ color: {DARK_TEXT_COLOR}; background-color: {DARK_BG_COLOR}; }} 
                 .css-z59ceb {{ color: {DARK_TEXT_COLOR}; background-color: {DARK_BG_COLOR}; }}
                 </style>
                 """
    st.markdown(dark_mode_style, unsafe_allow_html=True) 
    
def set_light_mode():   
LIGHT_BG_COLOR = '#FFFFFF' 
LIGHT_TEXT_COLOR = '#333333'  

def set_light_mode():  
    """Set light mode styling"""
    light_mode_style = f"""
                 <style> 
                 .stApp {{ background-color: {LIGHT_BG_COLOR}; color: {LIGHT_TEXT_COLOR}; }}
                 .css-17iph5v {{ color: {LIGHT_TEXT_COLOR}; background-color: {LIGHT_BG_COLOR}; }}  
                 .css-ykziq8  {{ color: {LIGHT_TEXT_COLOR}; background-color: {LIGHT_BG_COLOR}; }} 
                 .css-1g3nvew  {{ color: {LIGHT_TEXT_COLOR}; background-color: {LIGHT_BG_COLOR}; }}  
                 .css-2k4xy0 {{ color: {LIGHT_TEXT_COLOR}; background-color: {LIGHT_BG_COLOR}; }}  
                 .css-1nkbm6r {{ color: {LIGHT_TEXT_COLOR}; background-color: {LIGHT_BG_COLOR}; }}  
                 .css-z59ceb {{ color: {LIGHT_TEXT_COLOR}; background-color: {LIGHT_BG_COLOR}; }}
                 </style>
                 """
    st.markdown(light_mode_style, unsafe_allow_html=True) 

def set_padding(padding):
    """Set padding between page elements"""
    style = f"""<style>  
               .css-1aumxhk .css-1g3nv8z {{ padding: {padding}rem 0rem; }} 
               .css-1aumxhk .css-1g3nv8z > div:first-child {{ margin: {padding}rem 0rem {padding}rem 0rem ; }} 
               .css-z59ceb  {{ padding: {padding}rem 0rem; }}
               </style>
               """ 
    st.markdown(style, unsafe_allow_html=True)   

def hide_footer():  
    """Hide the Streamlit footer"""
    hide_footer_style = f""" 
               <style>
               footer {{ 
                   visibility: hidden;
               }} 
               .css-18zq0mr {{ 
                   visibility: hidden;
                   height: 0px; 
                   margin: 0;
                   padding: 0;
               }} 
               </style>
               """
    st.markdown(hide_footer_style, unsafe_allow_html=True)  

def set_tab_style(active_tab):
    """Set the styling for selected and unselected tabs"""
    selected_tab_style = """
        div[data-testid="stSidebar"] ul li a[aria-selected="true"] {
            background-color: #FFFFFF;
            color: #333333;
            border-radius: 0.5rem 0.5rem 0 0;
        }
    """
    
    unselected_tab_style = """
        div[data-testid="stSidebar"] ul li a[aria-selected="false"] {  
            background-color: #F0F2F6; 
            color: #888888;    
        }
    """

    if active_tab == 'Training':
        st.markdown(f"<style>{selected_tab_style}</style>", unsafe_allow_html=True) 
        st.markdown(f"<style>{unselected_tab_style}</style>", unsafe_allow_html=True) 
    elif active_tab == 'Inference':
        selected_tab_style = """
            div[data-testid="stSidebar"] ul li:nth-child(2) a[aria-selected="true"] {
                background-color: #FFFFFF;
                color: #333333;
                border-radius: 0.5rem 0.5rem 0 0;         
            }
        """
        st.markdown(f"<style>{selected_tab_style}</style>", unsafe_allow_html=True)
        
    elif active_tab == 'Model Hub':
        selected_tab_style = """
            div[data-testid="stSidebar"] ul li:nth-child(3) a[aria-selected="true"] {
                background-color: #FFFFFF;
                color: #333333;
                border-radius: 0.5rem 0.5rem 0 0;         
            }
        """
        st.markdown(f"<style>{selected_tab_style}</style>", unsafe_allow_html=True) 
        
    elif active_tab == 'Datasets':
        selected_tab_style = """
            div[data-testid="stSidebar"] ul li:nth-child(4) a[aria-selected="true"] {
                background-color: #FFFFFF;
                color: #333333;
                border-radius: 0.5rem 0.5rem 0 0;         
            }
        """
        st.markdown(f"<style>{selected_tab_style}</style>", unsafe_allow_html=True) 
        
def set_sidebar_and_main_layout(num_cols):
    """Set 1 or 2 column layout with sidebar""" 
    
    # Sidebar
    sidebar_style = """
        div[data-testid="stSidebar"] {
            width: 15rem;
            position: fixed;
            z-index: 100;
        }   
    """
    
    # Add padding if nesting sidebar 
    if num_cols == 2:
        padding = '2rem'
        sidebar_style += f"""
            div[data-testid="stSidebar"] {{ padding: {padding}; }} 
        """
        
    st.markdown(f"<style>{sidebar_style}</style>", unsafe_allow_html=True)
    
    # Main page 
    content_style = """
        div[data-testid="stContainer"] {
            margin-left: 17rem; 
        }
    """
    st.markdown(f"<style>{content_style}</style>", unsafe_allow_html=True)
    
    # Tabs 
    tab_options = ['Training', 'Inference', 'Model Hub', 'Datasets']
    active_tab = st.sidebar.radio('Navigate', tab_options)
    
    # 2 column layout   
    if num_cols == 2: 
        
        left_column, right_column = st.columns(2)
        with left_column:    
            st.sidebar.header('Menus') 
            st.sidebar.radio('Select', tab_options)
            
        with right_column:
            set_tab_content(active_tab)
            
    # 1 column layout 
    elif num_cols == 1:
        st.sidebar.header('Menus')
        set_tab_content(active_tab) 

def set_tab_content(active_tab):def set_sidebar_and_main_layout(num_cols):
    """Set 1 or 2 column layout with sidebar""" 
    
    # Sidebar
    sidebar_style = """
        div[data-testid="stSidebar"] {
            width: 15rem;
            position: fixed;
            z-index: 100;
        }   
    """
    
    # Add padding if nesting sidebar 
    if num_cols == 2:
        padding = '2rem'
        sidebar_style += f"""
            div[data-testid="stSidebar"] {{ padding: {padding}; }} 
        """
        
    st.markdown(f"<style>{sidebar_style}</style>", unsafe_allow_html=True)
    
    # Main page 
    content_style = """
        div[data-testid="stContainer"] {
            margin-left: 17rem; 
        }
    """
    st.markdown(f"<style>{content_style}</style>", unsafe_allow_html=True)
    
    # Tabs 
    tab_options = ['Training', 'Inference', 'Model Hub', 'Datasets']
    active_tab = st.sidebar.radio('Navigate', tab_options)
    
    # 2 column layout   
    if num_cols == 2: 
        
        left_column, right_column = st.columns(2)
        with left_column:    
            st.sidebar.header('Menus') 
            st.sidebar.radio('Select', tab_options)
            
        with right_column:
            set_tab_content(active_tab)
            
    # 1 column layout 
    elif num_cols == 1:
        st.sidebar.header('Menus')
        set_tab_content(active_tab) 

def set_tab_content(active_tab):
    if active_tab == 'Training':
        # Load dataset and model selectboxes 
        dataset_options = list_datasets()
        selected_dataset = st.selectbox("Select a dataset", dataset_options) 
        
        model_options = list_models()
        selected_model = st.selectbox("Select a model", model_options)  
        
        # Train model on selected dataset
        if st.button("Train model"):
            model = train_model(selected_dataset, selected_model)
        
        # Show training metrics 
        show_training(model)
    
    elif active_tab == 'Inference':
        # Show text input to enter text for inference
        input_text = st.text_input("Enter some text for inference:", "Here is some sample text")
        
        # Load model selectbox
        model_options = list_models()
        selected_model = st.selectbox("Select a model", model_options)  
        
        # Show prediction with a button
        if st.button("Predict"):
            prediction = inference(input_text, selected_model)
            st.write(prediction)
            
    elif active_tab == 'Model Hub':
        # Load model selectbox with 🤗 Transformers models
        model_options = list_transformer_models()
        selected_model = st.selectbox("Select a model", model_options)  
        
        # Download selected model and save to cache with a button
        if st.button("Download model"):
            downloaded_model = download_model(selected_model)
            save_model(downloaded_model)
            
        # Once downloaded, show model details
        if selected_model in list_models():
            view_model_details(selected_model)
            
    elif active_tab == 'Datasets':
        # Load dataset selectbox
        dataset_options = list_datasets()
        selected_dataset = st.selectbox("Select a dataset", dataset_options) 
        
        # Show preview of dataset 
        if selected_dataset:
            view_dataset(selected_dataset)
padding = '2rem'
sidebar_style = sidebar_style = f"""
    <style> 
    div[data-testid="stSidebar"] {{
        width: 15rem; 
        position: fixed;
        z-index: 100; 
    }}
    div[data-testid="stSidebar"] ul {{ 
        list-style-type: none;
        margin: 0;
        padding: 1rem 0; 
    }}
    div[data-testid="stSidebar"] ul li a {{
        color: #333333;
        font-weight: 500;
        padding: 0.5rem 1rem;
        display: block;
    }}
    div[data-testid="stSidebar"] ul li a:hover {{
        background-color: #F0F2F6;
        border-radius: 0.5rem;
    }}  
    </style> 
"""
st.markdown(sidebar_style, unsafe_allow_html=True)   

st.sidebar.header('Menus')  
st.sidebar.radio('Select', tab_options)   

set_tab_style(active_tab)   

</style>
