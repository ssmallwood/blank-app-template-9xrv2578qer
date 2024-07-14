import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Hidden Gem", page_icon="💎")

st.markdown("#### Version 2")
st.sidebar.header("Version 2")

# Column name mappings
column_names = {
    'Admission Rate 22': 'Admit Rate',
    'FTFT Grad Rate (6 Years) 2015-2016 Cohort': 'Grad Rate',
    'Earnings 10 Years Post-Entry 2008-2009 + 2009-2010 Cohorts': 'Earnings',
    '2022 Yield Rate': 'Yield Rate',
    'Grad Rates Residual': 'Grad Rates Residual',
    'Earnings Residual': 'Earnings Residual'
}

# Preset weight configurations
preset_weights = {
    'Balanced': {
        'Admit Rate': -0.2,
        'Grad Rate': 0.2,
        'Earnings': 0.2,
        'Yield Rate': 0.1,
        'Grad Rates Residual': 0.15,
        'Earnings Residual': 0.15
    },
    'Grad Rate Focused': {
        'Admit Rate': -0.1,
        'Grad Rate': 0.4,
        'Earnings': 0.1,
        'Yield Rate': 0.1,
        'Grad Rates Residual': 0.2,
        'Earnings Residual': 0.1
    },
    'Exceeding Expectations Focused': {
        'Admit Rate': -0.1,
        'Grad Rate': 0.1,
        'Earnings': 0.1,
        'Yield Rate': 0.1,
        'Grad Rates Residual': 0.3,
        'Earnings Residual': 0.3
    }
}

@st.cache_data
def load_data():
    df = pd.read_csv('college_data.csv')
    
    for old_name, new_name in column_names.items():
        if old_name in df.columns:
            df[new_name] = pd.to_numeric(df[old_name].astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', ''), errors='coerce')
            
            # Convert percentages to decimals
            if 'Rate' in new_name:
                df[new_name] = df[new_name] / 100
    
    # Change Control values to words
    df['Control'] = df['Control'].map({1: 'Public', 2: 'Private', 3: 'For-Profit'})
    
    return df

def z_score_standardization(df, columns):
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df[f'{col}_standardized'] = (df[col] - mean) / std
    return df

def percentile_ranking(df, columns):
    for col in columns:
        df[f'{col}_percentile'] = df[col].rank(pct=True)
    return df

def calculate_score(row, weights, method):
    score = 0
    for col, weight in weights.items():
        if method == 'Z-Score':
            score += row[f'{col}_standardized'] * weight if pd.notna(row[f'{col}_standardized']) else 0
        elif method == 'Percentile':
            score += row[f'{col}_percentile'] * weight if pd.notna(row[f'{col}_percentile']) else 0
    return score

def main():
    st.title('Hidden Gems Finder')

    df = load_data()

    # Move state selector to the top of the main display
    states = sorted(df['State'].unique())
    default_states = ['CA', 'PA', 'FL', 'MA', 'VA']
    selected_states = st.multiselect('Select States to Display', states, default=default_states)

    st.sidebar.header('Adjust Parameters')

    method = st.sidebar.radio(
        "Choose standardization method:",
        ('Z-Score', 'Percentile'),
        help="Z-Score measures how many standard deviations a value is from the mean. Percentile ranks values from 0 to 1 based on their position in the dataset."
    )

    min_admission_rate, max_admission_rate = st.sidebar.slider(
        'Admission Rate Range (%)', 
        min_value=0, 
        max_value=100, 
        value=(20, 100)
    )

    st.sidebar.subheader('Adjust Weights')
    preset_choice = st.sidebar.selectbox("Choose a preset or customize weights:", 
                                         ['Balanced', 'Grad Rate Focused', 'Exceeding Expectations Focused', 'Custom'])
    
    weights = preset_weights['Balanced'].copy()
    if preset_choice != 'Custom':
        weights = preset_weights[preset_choice].copy()
    
    st.sidebar.markdown("""
    Adjust the importance of each factor. Higher absolute values mean more importance.
    Negative values for Admit Rate mean lower rates are preferred.
    """)
    
    for col in weights.keys():
        weight = st.sidebar.slider(f'{col} weight', -1.0, 1.0, weights[col], 0.1, key=col)
        weights[col] = weight
        if weight != preset_weights.get(preset_choice, {}).get(col, weight):
            preset_choice = 'Custom'
    
    # Score all colleges
    columns_to_score = list(weights.keys())
    complete_data_df = df.dropna(subset=columns_to_score)
    incomplete_data_df = df[~df.index.isin(complete_data_df.index)]

    if method == 'Z-Score':
        scored_df = z_score_standardization(complete_data_df.copy(), columns_to_score)
    else:  # Percentile
        scored_df = percentile_ranking(complete_data_df.copy(), columns_to_score)

    scored_df['Score'] = scored_df.apply(lambda row: calculate_score(row, weights, method), axis=1)

    # Filter data for display
    filtered_scored_df = scored_df[
        (scored_df['Admit Rate'] >= min_admission_rate/100) & 
        (scored_df['Admit Rate'] <= max_admission_rate/100) &
        (scored_df['State'].isin(selected_states))
    ]

    filtered_incomplete_df = incomplete_data_df[
        (incomplete_data_df['Admit Rate'] >= min_admission_rate/100) & 
        (incomplete_data_df['Admit Rate'] <= max_admission_rate/100) &
        (incomplete_data_df['State'].isin(selected_states))
    ]

    # New: Create a DataFrame for colleges filtered out by Admission Rate
    filtered_out_df = pd.concat([
        scored_df[
            ((scored_df['Admit Rate'] < min_admission_rate/100) | 
             (scored_df['Admit Rate'] > max_admission_rate/100)) &
            (scored_df['State'].isin(selected_states))
        ],
        incomplete_data_df[
            ((incomplete_data_df['Admit Rate'] < min_admission_rate/100) | 
             (incomplete_data_df['Admit Rate'] > max_admission_rate/100)) &
            (incomplete_data_df['State'].isin(selected_states))
        ]
    ])

    # Display results
    for state in selected_states:
        st.write(f"\n### {state}")
        
        # Display scored colleges
        state_df = filtered_scored_df[filtered_scored_df['State'] == state].sort_values('Score', ascending=False).head(10).reset_index(drop=True)
        if not state_df.empty:
            st.subheader("Scored Colleges")
            display_columns = ['Institution Name', 'Control', 'Score'] + columns_to_score
            
            fig = px.bar(state_df, x='Institution Name', y='Score', title=f'Top 10 Scored Colleges in {state}')
            st.plotly_chart(fig)

            st.dataframe(state_df[display_columns].style.format({
                'Score': '{:.2f}',
                'Admit Rate': '{:.0%}',
                'Grad Rate': '{:.0%}',
                'Earnings': '${:,.0f}',
                'Yield Rate': '{:.0%}',
                'Grad Rates Residual': '{:.2f}',
                'Earnings Residual': '${:,.0f}'
            }).background_gradient(cmap="RdYlGn", subset=["Score"]))

        # Display unscored colleges in an expander
        unscored_state_df = filtered_incomplete_df[filtered_incomplete_df['State'] == state]
        if not unscored_state_df.empty:
            with st.expander("View Unscored Colleges"):
                st.subheader("Unscored Colleges (Incomplete Data)")
                unscored_display_columns = ['Institution Name', 'Control', 'Admit Rate', 
                                            'Grad Rate', 'Earnings', 'Yield Rate']
                st.dataframe(unscored_state_df[unscored_display_columns].style.format({
                    'Admit Rate': '{:.0%}',
                    'Grad Rate': '{:.0%}',
                    'Earnings': '${:,.0f}',
                    'Yield Rate': '{:.0%}'
                }))

        # New: Display colleges filtered out by Admission Rate
        filtered_out_state_df = filtered_out_df[filtered_out_df['State'] == state].sort_values('Admit Rate')
        if not filtered_out_state_df.empty:
            with st.expander("View Colleges Filtered Out by Admission Rate Setting"):
                st.subheader("Colleges Outside Admission Rate Range")
                filtered_out_columns = ['Institution Name', 'Control', 'Admit Rate', 'Grad Rate', 'Earnings']
                st.dataframe(filtered_out_state_df[filtered_out_columns].style.format({
                    'Admit Rate': '{:.0%}',
                    'Grad Rate': '{:.0%}',
                    'Earnings': '${:,.0f}'
                }))


    if st.button('Download Full Results'):
        csv = pd.concat([scored_df, incomplete_data_df]).to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="hidden_gems_rankings.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()