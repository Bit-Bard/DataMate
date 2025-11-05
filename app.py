import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ================== CONFIGURATION ==================
st.set_page_config(
    page_title="DataMate – Professional Data Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark professional styling with animations
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    /* Dark theme colors */
    :root {
        --bg-primary: #0A0E27;
        --bg-secondary: #1A1F3A;
        --bg-tertiary: #252B48;
        --accent-primary: #6C63FF;
        --accent-secondary: #FF6584;
        --accent-tertiary: #4ECDC4;
        --text-primary: #E8E9ED;
        --text-secondary: #A0A3BD;
        --success: #00D9A5;
        --warning: #FFA726;
        --danger: #FF6584;
    }
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0A0E27 0%, #1A1F3A 50%, #0A0E27 100%);
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Animated header with glow effect */
    .main-header {
        background: linear-gradient(135deg, #6C63FF 0%, #FF6584 50%, #4ECDC4 100%);
        background-size: 200% 200%;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(108, 99, 255, 0.3);
        animation: headerGlow 8s ease infinite;
        position: relative;
        overflow: hidden;
    }
    
    @keyframes headerGlow {
        0%, 100% { 
            background-position: 0% 50%;
            box-shadow: 0 10px 40px rgba(108, 99, 255, 0.3);
        }
        50% { 
            background-position: 100% 50%;
            box-shadow: 0 10px 40px rgba(255, 101, 132, 0.3);
        }
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 0 30px rgba(255,255,255,0.5);
        position: relative;
        z-index: 1;
        animation: textGlow 3s ease-in-out infinite;
    }
    
    @keyframes textGlow {
        0%, 100% { text-shadow: 0 0 20px rgba(255,255,255,0.5); }
        50% { text-shadow: 0 0 40px rgba(255,255,255,0.8); }
    }
    
    .main-header p {
        color: #E8E9ED;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        position: relative;
        z-index: 1;
        font-weight: 300;
    }
    
    /* Animated stat cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(37, 43, 72, 0.9) 0%, rgba(26, 31, 58, 0.9) 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(108, 99, 255, 0.2);
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(108, 99, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .stat-card:hover::before {
        left: 100%;
    }
    
    .stat-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 48px rgba(108, 99, 255, 0.4);
        border-color: rgba(108, 99, 255, 0.5);
    }
    
    .stat-card h3 {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0;
    }
    
    .stat-card h2 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0 0 0;
        background: linear-gradient(135deg, #6C63FF, #FF6584);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Sidebar dark theme */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1F3A 0%, #0A0E27 100%);
        border-right: 1px solid rgba(108, 99, 255, 0.2);
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    /* Enhanced button styling with pulse animation */
    .stButton>button {
        background: linear-gradient(135deg, #6C63FF 0%, #FF6584 100%);
        color: white;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.2);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(108, 99, 255, 0.6);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    /* Animated tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background: rgba(37, 43, 72, 0.6);
        border-radius: 12px 12px 0 0;
        padding: 0 28px;
        font-weight: 600;
        border: 1px solid rgba(108, 99, 255, 0.2);
        transition: all 0.3s ease;
        color: var(--text-secondary);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(37, 43, 72, 0.9);
        border-color: rgba(108, 99, 255, 0.4);
        color: var(--text-primary);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6C63FF 0%, #FF6584 100%);
        color: white;
        border-color: transparent;
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.4);
        animation: tabPulse 2s ease-in-out infinite;
    }
    
    @keyframes tabPulse {
        0%, 100% { box-shadow: 0 4px 20px rgba(108, 99, 255, 0.4); }
        50% { box-shadow: 0 4px 30px rgba(108, 99, 255, 0.6); }
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(37, 43, 72, 0.6);
        border-radius: 12px;
        font-weight: 600;
        color: var(--text-primary);
        border: 1px solid rgba(108, 99, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(37, 43, 72, 0.9);
        border-color: rgba(108, 99, 255, 0.4);
    }
    
    /* Input fields */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>div,
    .stTextArea>div>div>textarea {
        background: rgba(37, 43, 72, 0.6);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 8px;
        color: var(--text-primary);
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div>div:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: #6C63FF;
        box-shadow: 0 0 20px rgba(108, 99, 255, 0.3);
    }
    
    /* Alert boxes with dark theme */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
        animation: slideIn 0.5s ease-out;
        backdrop-filter: blur(10px);
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    [data-baseweb="notification"] {
        background: rgba(37, 43, 72, 0.95);
        border: 1px solid rgba(108, 99, 255, 0.3);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: rgba(26, 31, 58, 0.6);
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(108, 99, 255, 0.2);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #FF6584);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(37, 43, 72, 0.6);
        border: 2px dashed rgba(108, 99, 255, 0.3);
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(108, 99, 255, 0.6);
        background: rgba(37, 43, 72, 0.8);
    }
    
    /* Animated footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(90deg, #1A1F3A 0%, #0A0E27 50%, #1A1F3A 100%);
        background-size: 200% 100%;
        color: white;
        text-align: center;
        padding: 1.2rem;
        font-size: 0.95rem;
        z-index: 999;
        box-shadow: 0 -4px 30px rgba(0, 0, 0, 0.5);
        border-top: 1px solid rgba(108, 99, 255, 0.3);
        animation: footerGradient 5s ease infinite;
    }
    
    @keyframes footerGradient {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .footer strong {
        background: linear-gradient(135deg, #6C63FF, #FF6584, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    
    /* Radio buttons */
    .stRadio>div {
        background: rgba(37, 43, 72, 0.4);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(108, 99, 255, 0.2);
    }
    
    /* Checkbox */
    .stCheckbox {
        background: rgba(37, 43, 72, 0.4);
        padding: 0.75rem;
        border-radius: 8px;
        border: 1px solid rgba(108, 99, 255, 0.2);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6C63FF, #FF6584);
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #6C63FF !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1A1F3A;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #6C63FF, #FF6584);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #FF6584, #4ECDC4);
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: var(--text-primary) !important;
    }
    
    /* Markdown styling */
    .stMarkdown {
        color: var(--text-primary);
    }
    
    /* Success/Error messages with icons */
    .element-container div[data-testid="stMarkdownContainer"] p {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# Add your Gemini API Key here
GEMINI_API_KEY = "AIzaSyCOBZDP4S-F3bj091NwYcW9wyb8XG8_fRM"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ================== SESSION STATE ==================
if "df" not in st.session_state:
    st.session_state.df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "original_df" not in st.session_state:
    st.session_state.original_df = None

# ================== SIDEBAR CHAT ==================
st.sidebar.markdown("### 💬 AI Data Assistant")
st.sidebar.markdown("---")

user_query = st.sidebar.text_input("Ask your question:", placeholder="e.g. Which columns may have outliers?")

if user_query and st.session_state.df is not None:
    df = st.session_state.df.copy()
    df_info = f"Columns: {list(df.columns)}\nData types:\n{df.dtypes}\nShape: {df.shape}"
    prompt = (
        f"You are a data analysis assistant. Here is dataset info:\n{df_info}\n"
        f"User asked: {user_query}\n"
        "Answer briefly in under 5 lines. If user asks about methods or recommendations, respond clearly."
    )
    response = model.generate_content(prompt)
    reply = response.text.strip()
    st.session_state.chat_history.append(("You", user_query))
    st.session_state.chat_history.append(("AI", reply))

# Display chat
st.sidebar.markdown("---")
st.sidebar.markdown("**💭 Recent Conversations**")
for role, msg in st.session_state.chat_history[-10:]:
    if role == "You":
        st.sidebar.markdown(f"🧑‍💻 **You:** {msg}")
    else:
        st.sidebar.markdown(f"🤖 **AI:** {msg}")

# ================== MAIN AREA ==================
# Animated header
st.markdown("""
    <div class="main-header">
        <h1>🧠 DataMate</h1>
        <p>Professional Data Analytics & AI-Powered Insights Platform</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload your CSV file to get started", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.session_state.original_df = df.copy()
    st.success("✅ Dataset uploaded successfully!")
    
    # Animated stat cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="stat-card">
                <h3 style="color: #6C63FF;">📊 ROWS</h3>
                <h2>{df.shape[0]:,}</h2>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="stat-card">
                <h3 style="color: #FF6584;">📋 COLUMNS</h3>
                <h2>{df.shape[1]}</h2>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="stat-card">
                <h3 style="color: #FFA726;">⚠️ MISSING</h3>
                <h2>{df.isnull().sum().sum():,}</h2>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="stat-card">
                <h3 style="color: #4ECDC4;">🔢 NUMERIC</h3>
                <h2>{len(df.select_dtypes(include=np.number).columns)}</h2>
            </div>
        """, unsafe_allow_html=True)

# ================== DATA VIEW ==================
if st.session_state.df is not None:
    with st.expander("👁️ View Current Data (First 50 rows)", expanded=False):
        st.dataframe(st.session_state.df.head(50), use_container_width=True)

# ================== MAIN TABS ==================
if st.session_state.df is not None:
    df = st.session_state.df
    tabs = st.tabs(["📊 Exploratory Analysis", "🧹 Data Cleaning", "⚠️ Outlier Detection", "🧩 Feature Engineering", "⬇️ Export Data"])

    # ---- EDA ----
    with tabs[0]:
        st.markdown("### 📊 Exploratory Data Analysis")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### 📈 Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
        with col2:
            st.markdown("#### 🏷️ Data Types")
            dtype_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Type': df.dtypes.values
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 📊 Interactive Visualization")
        
        col = st.selectbox("Select column to visualize:", df.columns)
        
        if df[col].dtype == "object":
            vc = df[col].value_counts().reset_index()
            vc.columns = [col, "count"]
            fig = px.bar(
                vc, 
                x=col, 
                y="count", 
                title=f"Distribution of {col}",
                color="count",
                color_continuous_scale=["#6C63FF", "#FF6584", "#4ECDC4"]
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12, color='#E8E9ED'),
                title_font_size=20,
                title_font_color='#E8E9ED',
                showlegend=False,
                xaxis=dict(gridcolor='rgba(108, 99, 255, 0.1)'),
                yaxis=dict(gridcolor='rgba(108, 99, 255, 0.1)')
            )
        else:
            fig = px.histogram(
                df, 
                x=col, 
                nbins=30, 
                title=f"Distribution of {col}",
                color_discrete_sequence=['#6C63FF']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12, color='#E8E9ED'),
                title_font_size=20,
                title_font_color='#E8E9ED',
                xaxis=dict(gridcolor='rgba(108, 99, 255, 0.1)'),
                yaxis=dict(gridcolor='rgba(108, 99, 255, 0.1)')
            )
            
            # Add gradient fill
            fig.update_traces(marker=dict(
                line=dict(color='#FF6584', width=1),
                opacity=0.8
            ))
        
        st.plotly_chart(fig, use_container_width=True)

        if st.button("🧠 Generate AI Insights", type="primary"):
            with st.spinner("🔮 Analyzing dataset..."):
                sample_data = df.head(10).to_csv(index=False)
                prompt = f"Given dataset sample:\n{sample_data}\nBriefly summarize key relationships or important features in 4-5 lines."
                insight = model.generate_content(prompt).text.strip()
                st.info(f"**🤖 AI Insights:** {insight}")

    # ---- CLEANING ----
    with tabs[1]:
        st.markdown("### 🧹 Data Cleaning Operations")
        
        st.markdown("#### 🔍 Missing Values Overview")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Percentage': (missing_data.values / len(df) * 100).round(2)
            })
            
            # Create animated bar chart for missing values
            fig = px.bar(
                missing_df,
                x='Column',
                y='Percentage',
                title='Missing Values by Column (%)',
                color='Percentage',
                color_continuous_scale=['#4ECDC4', '#FF6584'],
                text='Missing Count'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E8E9ED'),
                title_font_color='#E8E9ED',
                xaxis=dict(gridcolor='rgba(108, 99, 255, 0.1)'),
                yaxis=dict(gridcolor='rgba(108, 99, 255, 0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")
        
        st.markdown("---")
        st.markdown("#### ⚙️ Cleaning Configuration")
        
        handle_option = st.radio(
            "Choose a missing value strategy:",
            ["Do nothing", "Drop rows with null values", "Fill missing values (Smart)"],
            help="Smart fill uses mean for numeric and mode for categorical columns"
        )

        remove_dupes = st.checkbox("🔄 Remove duplicate rows", help="Remove exact duplicate rows from dataset")

        if st.button("🚀 Apply Cleaning Operations", type="primary"):
            with st.spinner("🔄 Processing..."):
                df_updated = st.session_state.df.copy()
                changes_made = []

                if handle_option == "Drop rows with null values":
                    before = len(df_updated)
                    df_updated = df_updated.dropna()
                    after = len(df_updated)
                    if before > after:
                        changes_made.append(f"Dropped {before - after} rows with null values")

                elif handle_option == "Fill missing values (Smart)":
                    numeric_cols = df_updated.select_dtypes(include=['number']).columns
                    cat_cols = df_updated.select_dtypes(exclude=['number']).columns

                    filled_numeric = 0
                    filled_cat = 0
                    
                    for col in numeric_cols:
                        null_count = df_updated[col].isnull().sum()
                        if null_count > 0:
                            df_updated[col] = df_updated[col].fillna(df_updated[col].mean())
                            filled_numeric += null_count
                            
                    for col in cat_cols:
                        null_count = df_updated[col].isnull().sum()
                        if null_count > 0:
                            mode_val = df_updated[col].mode()
                            if len(mode_val) > 0:
                                df_updated[col] = df_updated[col].fillna(mode_val[0])
                                filled_cat += null_count

                    if filled_numeric > 0 or filled_cat > 0:
                        changes_made.append(f"Filled {filled_numeric} numeric and {filled_cat} categorical missing values")

                if remove_dupes:
                    before = len(df_updated)
                    df_updated = df_updated.drop_duplicates()
                    after = len(df_updated)
                    if before > after:
                        changes_made.append(f"Removed {before - after} duplicate rows")

                if changes_made:
                    st.session_state.df = df_updated
                    st.success("✅ **Data cleaning completed successfully!**")
                    for change in changes_made:
                        st.success(f"✓ {change}")
                    st.dataframe(df_updated.head(20), use_container_width=True)
                    st.balloons()
                else:
                    st.info("ℹ️ No changes applied. Please select cleaning options above.")

    # ---- OUTLIERS ----
    with tabs[2]:
        st.markdown("### ⚠️ Outlier Detection & Treatment")
        
        num_cols = df.select_dtypes(include=np.number).columns
        
        if len(num_cols) == 0:
            st.warning("⚠️ No numeric columns found for outlier detection.")
        else:
            col = st.selectbox("Select numeric column for analysis:", num_cols)
            method = st.radio(
                "Choose detection method:", 
                ["IQR (Interquartile Range)", "Z-Score", "Isolation Forest", "AI Recommendation"],
                help="IQR: 1.5*IQR rule | Z-Score: >3 std dev | Isolation Forest: ML-based"
            )

            if st.button("🔍 Detect Outliers", type="primary"):
                with st.spinner("🔎 Analyzing data..."):
                    if method == "IQR (Interquartile Range)":
                        Q1, Q3 = df[col].quantile([0.25, 0.75])
                        IQR = Q3 - Q1
                        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                        outliers = df[(df[col] < lower) | (df[col] > upper)]
                        
                    elif method == "Z-Score":
                        z_scores = np.abs(stats.zscore(df[col].dropna()))
                        outliers = df.loc[z_scores > 3]
                        
                    elif method == "Isolation Forest":
                        iso = IsolationForest(contamination=0.05, random_state=42)
                        preds = iso.fit_predict(df[[col]].fillna(df[col].median()))
                        outliers = df[preds == -1]
                        
                    elif method == "AI Recommendation":
                        df_info = f"Dataset Columns: {list(df.columns)}\nSample data:\n{df.head(3)}"
                        prompt = (
                            f"You are an expert data analyst. Given dataset info:\n{df_info}\n"
                            "Recommend the best outlier detection method (IQR, Z-Score, or Isolation Forest) "
                            "for numeric column analysis. Reply only with the method name and 1-line reason."
                        )
                        suggestion = model.generate_content(prompt).text.strip()
                        st.info(f"🤖 **AI Recommendation:** {suggestion}")
                        st.stop()

                    st.markdown(f"#### 📊 Results for `{col}`")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Outliers Detected", f"{len(outliers)} rows", f"{len(outliers)/len(df)*100:.2f}%")
                    with col2:
                        st.metric("Clean Data", f"{len(df) - len(outliers)} rows", f"{(len(df) - len(outliers))/len(df)*100:.2f}%")
                    
                    # Professional box plot with dark theme
                    fig = go.Figure()
                    fig.add_trace(go.Box(
                        y=df[col],
                        name=col,
                        marker=dict(
                            color='#6C63FF',
                            line=dict(color='#FF6584', width=2)
                        ),
                        boxmean='sd',
                        fillcolor='rgba(108, 99, 255, 0.5)'
                    ))
                    fig.update_layout(
                        title=f"Outlier Visualization - {col}",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=12, color='#E8E9ED'),
                        title_font_size=20,
                        title_font_color='#E8E9ED',
                        yaxis=dict(gridcolor='rgba(108, 99, 255, 0.1)')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if len(outliers) > 0:
                        with st.expander("👁️ View Outlier Data"):
                            st.dataframe(outliers, use_container_width=True)
                        
                        if st.button("🧹 Remove Detected Outliers"):
                            if method == "IQR (Interquartile Range)":
                                df_clean = df[(df[col] >= lower) & (df[col] <= upper)]
                            elif method == "Z-Score":
                                z_scores = np.abs(stats.zscore(df[col].dropna()))
                                df_clean = df[z_scores <= 3]
                            elif method == "Isolation Forest":
                                df_clean = df[preds != -1]
                            
                            st.session_state.df = df_clean
                            st.success(f"✅ Removed {len(df) - len(df_clean)} outliers successfully!")
                            st.balloons()
                    else:
                        st.success("✅ No outliers detected!")

    # ---- FEATURE ENGINEERING ----
    with tabs[3]:
        st.markdown("### 🧩 Feature Engineering")
        
        mode = st.radio("Select mode:", ["🔧 Manual Feature Creation", "🤖 AI-Powered Suggestions"])
        
        if mode == "🔧 Manual Feature Creation":
            st.markdown("#### Create Custom Features")
            c1, c2 = st.columns(2)
            with c1:
                col1 = st.selectbox("Select first column:", df.columns, key="fe_col1")
            with c2:
                col2 = st.selectbox("Select second column:", df.columns, key="fe_col2")
            
            op = st.selectbox("Operation:", ["➕ Add", "➖ Subtract", "✖️ Multiply", "➗ Divide"])
            name = st.text_input("New feature name:", placeholder="e.g., total_score")
            
            if st.button("✨ Create Feature", type="primary"):
                if name:
                    df_updated = st.session_state.df.copy()
                    try:
                        if op == "➕ Add": 
                            df_updated[name] = df_updated[col1] + df_updated[col2]
                        elif op == "➖ Subtract": 
                            df_updated[name] = df_updated[col1] - df_updated[col2]
                        elif op == "✖️ Multiply": 
                            df_updated[name] = df_updated[col1] * df_updated[col2]
                        elif op == "➗ Divide": 
                            df_updated[name] = df_updated[col1] / (df_updated[col2] + 1e-6)
                        
                        st.session_state.df = df_updated
                        st.success(f"✅ Created new feature: **{name}**")
                        
                        # Visualize new feature
                        st.markdown("#### 📊 New Feature Preview")
                        preview_df = df_updated[[col1, col2, name]].head(10)
                        st.dataframe(preview_df, use_container_width=True)
                        
                        # Create comparison chart
                        if df_updated[name].dtype in ['int64', 'float64']:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=df_updated.index[:50],
                                y=df_updated[name].head(50),
                                mode='lines+markers',
                                name=name,
                                line=dict(color='#6C63FF', width=3),
                                marker=dict(size=8, color='#FF6584')
                            ))
                            fig.update_layout(
                                title=f"New Feature: {name}",
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#E8E9ED'),
                                title_font_color='#E8E9ED',
                                xaxis=dict(gridcolor='rgba(108, 99, 255, 0.1)'),
                                yaxis=dict(gridcolor='rgba(108, 99, 255, 0.1)')
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error creating feature: {str(e)}")
                else:
                    st.warning("⚠️ Please enter a name for the new feature.")
        
        else:
            st.markdown("#### 🤖 AI-Powered Feature Suggestions")
            
            if st.button("💡 Get AI Suggestions", type="primary"):
                with st.spinner("🧠 AI is analyzing your dataset..."):
                    sample = df.head(5).to_csv(index=False)
                    prompt = f"Suggest 3 new engineered feature ideas from this dataset:\n{sample}\nKeep response short (under 5 lines)."
                    ai_idea = model.generate_content(prompt).text.strip()
                    st.info(f"**🤖 AI Suggestions:**\n\n{ai_idea}")
            
            st.markdown("---")
            
            if st.button("✨ Auto-Create Smart Features", type="primary"):
                with st.spinner("⚡ Creating features..."):
                    df_updated = st.session_state.df.copy()
                    num_cols = df_updated.select_dtypes(include=np.number).columns
                    features_created = []
                    
                    # Create squared features
                    for c in num_cols[:3]:
                        new_col = f"{c}_squared"
                        df_updated[new_col] = df_updated[c] ** 2
                        features_created.append(new_col)
                    
                    # Create ratio features
                    if len(num_cols) >= 2:
                        new_col = f"{num_cols[0]}_to_{num_cols[1]}_ratio"
                        df_updated[new_col] = df_updated[num_cols[0]] / (df_updated[num_cols[1]] + 1e-6)
                        features_created.append(new_col)
                    
                    st.session_state.df = df_updated
                    st.success(f"✅ Created {len(features_created)} new features!")
                    st.write("**✨ New features:**", ", ".join(features_created))
                    st.dataframe(df_updated[features_created].head(10), use_container_width=True)
                    st.balloons()

    # ---- DOWNLOAD ----
    with tabs[4]:
        st.markdown("### ⬇️ Export Processed Data")
        
        st.info(f"**📊 Current dataset:** {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} columns")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            csv = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="datamate_processed.csv",
                mime="text/csv",
                type="primary"
            )
        
        with col2:
            excel_buffer = st.session_state.df.to_excel
            st.download_button(
                label="📊 Download Excel",
                data=st.session_state.df.to_csv(index=False),
                file_name="datamate_processed.xlsx",
                mime="application/vnd.ms-excel",
                type="primary"
            )
        
        with col3:
            if st.button("🔄 Reset to Original", type="primary"):
                if st.session_state.original_df is not None:
                    st.session_state.df = st.session_state.original_df.copy()
                    st.success("✅ Dataset reset to original state!")
                    st.balloons()
                    st.rerun()
        
        st.markdown("---")
        st.markdown("#### 📊 Final Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{st.session_state.df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", st.session_state.df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

else:
    st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem;'>
            <h2 style='color: #6C63FF; font-size: 2.5rem; margin-bottom: 1rem;'>
                👋 Welcome to DataMate
            </h2>
            <p style='color: #A0A3BD; font-size: 1.2rem; margin-bottom: 2rem;'>
                Upload a CSV file to unlock powerful data analytics
            </p>
            <div style='font-size: 4rem; margin: 2rem 0;'>📊</div>
        </div>
    """, unsafe_allow_html=True)

# ================== ANIMATED FOOTER ==================
st.markdown("""
    <div class="footer">
        <strong>Made with ❤️ by Dhruv Devaliya</strong> | Powered by Streamlit & Google Gemini AI | 🚀 DataMate v2.0
    </div>
""", unsafe_allow_html=True)

# Add bottom padding to prevent footer overlap
st.markdown("<div style='padding-bottom: 100px;'></div>", unsafe_allow_html=True)