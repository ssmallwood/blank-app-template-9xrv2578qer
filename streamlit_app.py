import streamlit as st
import pandas as pd
import numpy as np
import time

# Compatibility wrapper for st.cache_data
def compatible_cache_data(*args, **kwargs):
    try:
        return st.cache_data(*args, **kwargs)
    except TypeError:
        return st.cache_data()

# Load and preprocess data
@compatible_cache_data
def load_data():
    try:
        df = pd.read_csv('college_data.csv')
        
        def string_to_float(value):
            if isinstance(value, str):
                value = value.replace(',', '').replace('%', '').strip()
                try:
                    return float(value)
                except ValueError:
                    return np.nan
            return value

        columns_to_convert = [
            'Admission Rate 22',
            'FTFT Grad Rate (6 Years) 2015-2016 Cohort',
            'Earnings 10 Years Post-Entry 2008-2009 + 2009-2010 Cohorts',
            'Grad Rates Residual',
            'Earnings Residual',
            '2022 Yield Rate'
        ]

        for col in columns_to_convert:
            df[col] = df[col].apply(string_to_float)
            if 'Rate' in col or 'Grad' in col:
                df[col] = df[col] / 100

        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Updated scoring functions
def score_standard(row):
    return (
        row['FTFT Grad Rate (6 Years) 2015-2016 Cohort'] +
        row['Earnings 10 Years Post-Entry 2008-2009 + 2009-2010 Cohorts'] / 100000 +
        row['Grad Rates Residual'] * 10 +
        row['Earnings Residual'] / 1000
    )

def score_yield_focused(row):
    return (
        row['FTFT Grad Rate (6 Years) 2015-2016 Cohort'] * 0.2 +
        row['Earnings 10 Years Post-Entry 2008-2009 + 2009-2010 Cohorts'] / 100000 * 0.2 +
        row['Grad Rates Residual'] * 5 +
        row['Earnings Residual'] / 1000 * 0.1 +
        row['2022 Yield Rate'] * 0.5
    )

def score_grad_rate_focused(row):
    return (
        row['FTFT Grad Rate (6 Years) 2015-2016 Cohort'] * 0.3 +
        row['Earnings 10 Years Post-Entry 2008-2009 + 2009-2010 Cohorts'] / 100000 * 0.1 +
        row['Grad Rates Residual'] * 20 +
        row['Earnings Residual'] / 1000 * 0.2 +
        (1 - row['Admission Rate 22']) * 0.1
    )

# Function to display college information
def display_college_info(rank, college):
    st.markdown(f"**{rank}. {college['Inst Name']}**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Score: {college['Score']:.2f}")
        st.write(f"Admission Rate: {college['Admission Rate 22']:.1%}")
        st.write(f"Yield Rate: {college['2022 Yield Rate']:.1%}")
        st.write(f"Graduation Rate: {college['FTFT Grad Rate (6 Years) 2015-2016 Cohort']:.1%}")
    with col2:
        st.write(f"Earnings: ${college['Earnings 10 Years Post-Entry 2008-2009 + 2009-2010 Cohorts']:,.0f}")
        st.write(f"Earnings Residual: ${college['Earnings Residual']:,.0f}")
        st.write(f"Grad Rate Residual: {college['Grad Rates Residual']:.2f}")
    st.markdown("---")

# Streamlit app
def main():
    st.title('College Hidden Gems Finder')

    # Load data
    df = load_data()

    if df.empty:
        st.warning("No data available. Please check your data file.")
    else:
        # Sidebar for filters
        st.sidebar.header('Adjust Criteria')
        admission_rate_range = st.sidebar.slider(
            'Admission Rate', 
            min_value=0, 
            max_value=100, 
            value=(30, 100),  # Default range from 30% to 100%
            step=1
         )
        # Graduation Rate Range Slider
        graduation_rate_range = st.sidebar.slider(
            'Graduation Rate', 
            min_value=0, 
            max_value=100, 
            value=(60, 100),  # Default range from 60% to 100%
            step=1
        )

        # Convert percentages to decimals for filtering
        min_admission_rate = admission_rate_range[0] / 100
        max_admission_rate = admission_rate_range[1] / 100
        min_graduation_rate = graduation_rate_range[0] / 100
        max_graduation_rate = graduation_rate_range[1] / 100

        min_earnings = st.sidebar.number_input('Minimum Earnings', 0, 200000, 50000, step=1000)
        min_grad_residual = st.sidebar.number_input('Minimum Graduation Rate Residual', -0.2, 0.2, 0.0, step=0.01)
        min_earnings_residual = st.sidebar.number_input('Minimum Earnings Residual', -10000, 10000, 0, step=100)

        # Display options
        st.sidebar.header('Display Options')
        selected_states = st.sidebar.multiselect('Select States', options=['All'] + sorted(df['State'].unique().tolist()), default='All')
        display_mode = st.radio("Display Mode", ("By State", "Global Ranking"))
        
        if display_mode == "By State":
            colleges_per_state = st.sidebar.slider('Number of colleges per state', 1, 10, 5)
        else:
            num_colleges = st.sidebar.slider('Number of colleges to display', 10, 200, 50)

        scoring_method = st.selectbox(
            "Scoring Method",
            ["Standard", "Yield Focused", "Graduation Rate Focused"],
            format_func=lambda x: x.replace("_", " ").title(),
            help="""
            Standard: Balances grad rate, earnings, and residuals equally.
            Yield Focused: Emphasizes yield rate and balances other factors.
            Graduation Rate Focused: Heavily weights graduation rate and its residual, rewarding schools that outperform expectations.
            """
        )

        # Add checkbox to show/hide scoring formula
        show_formula = st.checkbox("Show Scoring Formula")

        # Display scoring formula only if checkbox is checked
        if show_formula:
            if scoring_method == "Standard":
                st.markdown("""
                **Scoring Formula:**
                ```
                Score = Graduation Rate +
                        (Earnings / 100,000) +
                        (Grad Rate Residual * 10) +
                        (Earnings Residual / 1,000)
                ```
                """)
            elif scoring_method == "Yield Focused":
                st.markdown("""
                **Scoring Formula:**
                ```
                Score = (Graduation Rate * 0.2) +
                        (Earnings / 100,000 * 0.2) +
                        (Grad Rate Residual * 5) +
                        (Earnings Residual / 1,000 * 0.1) +
                        (Yield Rate * 0.5)
                ```
                """)
            else:  # Graduation Rate Focused
                st.markdown("""
                **Scoring Formula:**
                ```
                Score = (Graduation Rate * 0.3) +
                        (Earnings / 100,000 * 0.1) +
                        (Grad Rate Residual * 20) +
                        (Earnings Residual / 1,000 * 0.2) +
                        ((1 - Admission Rate) * 0.1)
                ```
                """)

        # Filter data based on user input
        filtered_df = df[
            (df['Admission Rate 22'] >= min_admission_rate) &
            (df['Admission Rate 22'] <= max_admission_rate) &
            (df['FTFT Grad Rate (6 Years) 2015-2016 Cohort'] >= min_graduation_rate) &
            (df['FTFT Grad Rate (6 Years) 2015-2016 Cohort'] <= max_graduation_rate) &
            (df['Earnings 10 Years Post-Entry 2008-2009 + 2009-2010 Cohorts'] > min_earnings) &
            (df['Grad Rates Residual'] > min_grad_residual) &
            (df['Earnings Residual'] > min_earnings_residual)
        ].copy()

        # Apply state filter
        if 'All' not in selected_states:
            filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]

        # Calculate score based on selected method
        if scoring_method == "Standard":
            filtered_df['Score'] = filtered_df.apply(score_standard, axis=1)
        elif scoring_method == "Yield Focused":
            filtered_df['Score'] = filtered_df.apply(score_yield_focused, axis=1)
        else:
            filtered_df['Score'] = filtered_df.apply(score_grad_rate_focused, axis=1)

        # Display results
        if display_mode == "By State":
            top_colleges = (filtered_df.groupby('State', group_keys=False)
                            .apply(lambda x: x.nlargest(colleges_per_state, 'Score'))
                            .reset_index(drop=True))
            st.write(f"Found {len(top_colleges)} potential hidden gems across {len(top_colleges['State'].unique())} states")
            
            for state in top_colleges['State'].unique():
                st.subheader(f"Top hidden gems in {state}")
                state_colleges = top_colleges[top_colleges['State'] == state]
                for i, (_, college) in enumerate(state_colleges.iterrows(), 1):
                    display_college_info(i, college)
        else:
            top_colleges = filtered_df.nlargest(num_colleges, 'Score').reset_index(drop=True)
            st.write(f"Top {len(top_colleges)} potential hidden gems")
            
            for i, (_, college) in enumerate(top_colleges.iterrows(), 1):
                display_college_info(i, college)

        # Optional: Add a download button for the results
        if not top_colleges.empty:
            csv = top_colleges.to_csv(index=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name="hidden_gems.csv",
                mime="text/csv",
            )

    # Add a refresh button at the end
    if st.button('Refresh App'):
        st.rerun()

    # Display last update time
    st.sidebar.write(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()