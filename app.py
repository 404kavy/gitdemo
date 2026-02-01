import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import regex as re

# ======================
# LOAD DATA (ONCE)
# ======================
df = pd.read_csv('startup_funding.csv')
df.dropna(subset=['Amount in USD'], inplace=True)
df['Amount in USD'] = (
    df['Amount in USD']
    .astype(str)
    .str.replace(r'\D+', '', regex=True)
)

df = df[df['Amount in USD'] != '']
df['Amount in USD'] = df['Amount in USD'].astype(int)




# ======================
# CLEAN DATA
# ======================
df['Investors Name'] = df['Investors Name'].astype('string')

df['Amount in USD'] = (
    df['Amount in USD']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.replace('$', '', regex=False)
)

df['Amount in USD'] = pd.to_numeric(df['Amount in USD'], errors='coerce')

# Convert date
df['Date dd/mm/yyyy'] = pd.to_datetime(df['Date dd/mm/yyyy'], format='%d/%m/%Y', errors='coerce')
df['Year'] = df['Date dd/mm/yyyy'].dt.year
df['YearMonth'] = df['Date dd/mm/yyyy'].dt.to_period('M')

#Load investor Detail function
def load_investor_details(investor):
    investor_df = df[df['Investors Name'].astype(str).str.contains(investor)][['Date dd/mm/yyyy', 'Startup Name', 'Industry Vertical', 'SubVertical', 'City  Location', 'Amount in USD']]
    
    st.header(investor)
    
    # Format table
    display_df = investor_df.copy()
    display_df['Amount in USD'] = display_df['Amount in USD'].apply(lambda x: f"$ {x:,.0f}" if pd.notna(x) else "N/A")
    display_df.columns = ['Date', 'Startup', 'Industry', 'Sub-Industry', 'Location', 'Investment']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)    
    
    total = investor_df['Amount in USD'].sum() / 1_000_000
    st.metric("Total Investment", f"$ {total:,.1f}M")
    
    biggest_invested = df[df['Investors Name'].str.contains(investor,na =False)].groupby('Startup Name')['Amount in USD'].sum().sort_values(ascending=False).head(5)
    
    st.subheader("Biggest Investments")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(biggest_invested.index, biggest_invested.values, color=sns.color_palette("husl", len(biggest_invested)))
        ax.set_xlabel("Amount in USD")
        ax.set_title("Top 5 Startups by Investment")
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Sector-wise pie chart
        sector_invested = df[df['Investors Name'].str.contains(investor, na=False)].groupby('Industry Vertical')['Amount in USD'].sum().sort_values(ascending=False).head(8)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(sector_invested.values, labels=sector_invested.index, autopct='%1.1f%%', startangle=90)
        ax.set_title("Investment Distribution by Sector")
        st.pyplot(fig)
    
    # City-wise analysis
    st.subheader("City-wise Investment")
    city_invested = df[df['Investors Name'].str.contains(investor, na=False)].groupby('City  Location')['Amount in USD'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(city_invested.index, city_invested.values, color=sns.color_palette("coolwarm", len(city_invested)))
    ax.set_xlabel("City")
    ax.set_ylabel("Amount in USD")
    ax.set_title("Top 10 Cities by Investment")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Similar Investors
    st.subheader("Similar Investors")
    investor_startups = set(df[df['Investors Name'].str.contains(investor, na=False)]['Startup Name'].unique())
    similar_investors = df[df['Startup Name'].isin(investor_startups)]['Investors Name'].value_counts().head(10)
    st.bar_chart(similar_investors)

# Load startup Detail function
def load_startup_details(startup):
    startup_df = df[df['Startup Name'].astype(str).str.contains(startup)]
    
    st.header(startup)
    
    # Display full table
    display_df = startup_df.copy()
    display_df['Amount in USD'] = display_df['Amount in USD'].apply(lambda x: f"$ {x:,.0f}" if pd.notna(x) else "N/A")
    
    st.subheader("All Investment Rounds")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_funding = startup_df['Amount in USD'].sum() / 1_000_000
    num_rounds = len(startup_df)
    avg_funding = startup_df['Amount in USD'].mean() / 1_000_000
    num_investors = startup_df['Investors Name'].nunique()
    
    with col1:
        st.metric("Total Funding", f"$ {total_funding:,.1f}M")
    with col2:
        st.metric("Funding Rounds", num_rounds)
    with col3:
        st.metric("Avg per Round", f"$ {avg_funding:,.1f}M")
    with col4:
        st.metric("Total Investors", num_investors)
    
    # Funding Timeline
    st.subheader("Funding Timeline")
    timeline_data = startup_df.groupby('Date dd/mm/yyyy')['Amount in USD'].sum().sort_index()
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(timeline_data.index, timeline_data.values, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.fill_between(range(len(timeline_data)), timeline_data.values, alpha=0.3, color='steelblue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Amount in USD")
    ax.set_title(f"{startup} - Funding Over Time")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Funding per Round
    st.subheader("Funding by Round")
    col1, col2 = st.columns(2)
    
    with col1:
        round_data = startup_df.groupby('Date dd/mm/yyyy')['Amount in USD'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(round_data.index.astype(str), round_data.values, color=sns.color_palette("plasma", len(round_data)))
        ax.set_xlabel("Amount in USD")
        ax.set_title("Funding Amount per Round")
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Industry Info
        industry = startup_df['Industry Vertical'].iloc[0] if len(startup_df) > 0 else "N/A"
        sub_vertical = startup_df['SubVertical'].iloc[0] if len(startup_df) > 0 else "N/A"
        location = startup_df['City  Location'].iloc[0] if len(startup_df) > 0 else "N/A"
        
        st.metric("Industry", industry)
        st.metric("Sub-Vertical", sub_vertical)
        st.metric("Location", location)
    
    # Investors
    st.subheader("All Investors")
    investor_data = startup_df.groupby('Investors Name')['Amount in USD'].sum().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(investor_data.index, investor_data.values, color=sns.color_palette("viridis", len(investor_data)))
    ax.set_xlabel("Amount in USD")
    ax.set_title("Funding by Investor")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Investment Type Breakdown
    st.subheader("Investment Type Distribution")
    if 'InvestmentType' in startup_df.columns:
        inv_type = startup_df.groupby('InvestmentType')['Amount in USD'].sum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.pie(inv_type.values, labels=inv_type.index, autopct='%1.1f%%', startangle=90)
            ax.set_title("Investment Type Distribution")
            st.pyplot(fig)
        
        with col2:
            st.subheader("Investment Type Summary")
            for inv_t, amount in inv_type.items():
                st.metric(inv_t, f"$ {amount:,.0f}")
    
    # Similar Startups (same industry)
    st.subheader("Similar Startups in Same Industry")
    similar_startups = df[
        (df['Industry Vertical'] == startup_df['Industry Vertical'].iloc[0]) & 
        (df['Startup Name'] != startup)
    ]['Startup Name'].unique()[:10]
    
    if len(similar_startups) > 0:
        similar_funding = df[df['Startup Name'].isin(similar_startups)].groupby('Startup Name')['Amount in USD'].sum().sort_values(ascending=False)
        st.bar_chart(similar_funding)
    else:
        st.info("No similar startups found in this industry")

# ======================
# SIDEBAR
# ======================
st.sidebar.title('Startup Funding Analysis')

option = st.sidebar.selectbox(
    'Select Analysis Type',
    ['Overall Analysis', 'Startup-wise Analysis', 'Investor-wise Analysis']
)

# ======================
# OVERALL ANALYSIS
# ======================
if option == 'Overall Analysis':
    st.title('Overall Analysis')
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_funding = df['Amount in USD'].sum() / 1_000_000_000
        st.metric("Total", f"$ {total_funding:,.2f}B")
    
    with col2:
        max_funding = df['Amount in USD'].max() / 1_000_000
        st.metric("Max", f"$ {max_funding:,.1f}M")
    
    with col3:
        avg_funding = df['Amount in USD'].mean() / 1_000_000
        st.metric("Avg", f"$ {avg_funding:,.1f}M")
    
    with col4:
        funded_startups = df['Startup Name'].nunique()
        st.metric("Funded Startups", f"{funded_startups}")
    
    # MoM graph
    st.subheader("MoM graph")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        mom_type = st.selectbox("Select Type", ["Total", "Count", "Average"], key="mom_select")
    
    with col1:
        if mom_type == "Total":
            mom_data = df.groupby('YearMonth')['Amount in USD'].sum().sort_index() / 1_000_000
        elif mom_type == "Count":
            mom_data = df.groupby('YearMonth')['Amount in USD'].count().sort_index()
        else:
            mom_data = df.groupby('YearMonth')['Amount in USD'].mean().sort_index() / 1_000_000
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(mom_data.index.astype(str), mom_data.values, color='steelblue')
        ax.set_xlabel("Month")
        ax.set_ylabel("Amount (M)" if mom_type != "Count" else "Number of Deals")
        ax.set_title(f"Month-on-Month {mom_type}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Cards - Total + Max + Avg
    st.subheader("Cards")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_b = df['Amount in USD'].sum() / 1_000_000_000
        st.metric("Total Funded Startups", f"$ {total_b:,.2f}B")
    with col2:
        max_m = df['Amount in USD'].max() / 1_000_000
        st.metric("Top Investment", f"$ {max_m:,.1f}M")
    with col3:
        avg_m = df['Amount in USD'].mean() / 1_000_000
        st.metric("Average Funding", f"$ {avg_m:,.1f}M")
    
    # Sector Analysis
    #
    st.subheader("Sector Analysis")
    sector_data = df.groupby('Industry Vertical')['Amount in USD'].sum().sort_values(ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(sector_data.index, sector_data.values, color=sns.color_palette("viridis", len(sector_data)))
    ax.set_xlabel("Amount in USD")
    ax.set_title("Top 10 Sectors by Funding")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Type of Funding
    st.subheader("Type of Funding")
    if 'InvestmentType' in df.columns:
        funding_type = df.groupby('InvestmentType')['Amount in USD'].sum().sort_values(ascending=False)
    else:
        funding_type = df.groupby('Industry Vertical')['Amount in USD'].sum().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.pie(funding_type.values, labels=funding_type.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Funding Distribution by Type")
    st.pyplot(fig)
    
    # City wise funding
    st.subheader("City Wise Funding")
    city_data = df.groupby('City  Location')['Amount in USD'].sum().sort_values(ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(city_data.index, city_data.values, color=sns.color_palette("coolwarm", len(city_data)))
    ax.set_xlabel("City")
    ax.set_ylabel("Amount in USD")
    ax.set_title("Top 15 Cities by Funding")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Top Startups
    st.subheader("Top Startups")
    top_startups = df.groupby('Startup Name')['Amount in USD'].sum().sort_values(ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(top_startups.index, top_startups.values, color=sns.color_palette("rocket", len(top_startups)))
    ax.set_xlabel("Amount in USD")
    ax.set_title("Top 10 Startups by Funding")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Top Investors
    st.subheader("Top Investors")
    top_investors = df.groupby('Investors Name')['Amount in USD'].sum().sort_values(ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(top_investors.index, top_investors.values, color=sns.color_palette("mako", len(top_investors)))
    ax.set_xlabel("Amount in USD")
    ax.set_title("Top 10 Investors by Funding")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Funding Heatmap
    df.columns = df.columns.str.strip()
    
    heatmap_data = (
        df.dropna(subset=['Industry Vertical', 'Year', 'Amount in USD'])
        .pivot_table(
            values='Amount in USD',
            index='Industry Vertical',
            columns='Year',
            aggfunc='sum'
        )
        .head(10)
    )

    st.subheader("Funding Heatmap: Industry vs Year")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax, annot=True, fmt='.0f')

    ax.set_xlabel("Year")
    ax.set_ylabel("Industry")
    plt.tight_layout()
    st.pyplot(fig)

    
# ======================
# STARTUP-WISE ANALYSIS
# ======================
elif option == 'Startup-wise Analysis':
    st.title('Startup-wise Analysis')

    startup_names = sorted(
        df['Startup Name']
        .dropna()
        .astype(str)
        .replace(r'[^A-Za-z0-9 ]+', '', regex=True)
        .unique()
        .tolist()
    )

    startup = st.sidebar.selectbox('Select Startup', startup_names)
    btn1 = st.sidebar.button('Find Startup Details')

    if btn1:
        load_startup_details(startup)

    st.caption(f'{len(startup_names)} startups in database')

# ======================
# INVESTOR-WISE ANALYSIS
# ======================
else:
    st.title('Investor-wise Analysis')

    investors = sorted(
        df['Investors Name']
        .dropna()
        .astype(str)
        .replace(r'[^A-Za-z0-9 ]+', '', regex=True)
        .unique()
        .tolist()
    )

    investor = st.sidebar.selectbox('Select Investor', investors)
    btn2 = st.sidebar.button('Find Investor Details')

    if btn2:
        load_investor_details(investor)
    
    st.caption(f'{len(investors)} investors in database')