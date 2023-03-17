import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

### Config
st.set_page_config(
    page_title="Getaround Analysis",
    page_icon="🚗",
    layout="wide"
)

DATA_URL = ('https://getaround-bucket-27-12-2022.s3.eu-west-3.amazonaws.com/getaround_df.csv')
DATA_URL_2 = ('https://getaround-bucket-27-12-2022.s3.eu-west-3.amazonaws.com/getaround_deltas_df.csv')

### App
st.title("Getaround Analysis")

st.markdown("""
    **Welcome to the analysis of rental delays for Getaround!**
""")

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    return data

st.markdown("---")
st.subheader(""" **Load part of initial enriched dataset** """)

data_load_state = st.text('Loading data...')
data = load_data(3000)
data_load_state.text("") # change text from "Loading data..." to "" once the the load_data function has run

## Run the below code if the check is checked ✅
if st.checkbox('Show raw data', key="global_df"):
    st.subheader('Raw data')
    st.write(data)

st.markdown("---")
st.subheader(""" **Load dataset enriched with information on previous clients' delays** """)

data_load_state = st.text('Loading data...')
delta_df = pd.read_csv(DATA_URL_2)
data_load_state.text("") # change text from "Loading data..." to "" once the the load_data function has run

st.markdown("---")

## Run the below code if the check is checked ✅
if st.checkbox('Show raw data', key="delta_df"):
    st.subheader('Raw data')
    st.write(delta_df)   


#### Defining charts


def get_chart_rentals_by_check_in_type(df):
    colors = ['#8d1586', '#eec186']
    fig = px.pie(df, names='checkin_type', color_discrete_sequence=colors, hole=0.55)
    fig.update_layout(title='Car rentals by check-in type')
    st.plotly_chart(fig, theme="streamlit")


def get_chart_cancellations_by_contract_type(df):
    # Compute the percentage of rentals that ended or were canceled for each contract type
    contract_ended = df[df['state'] == 'ended']
    contract_canceled = df[df['state'] == 'canceled']
    mobile_ended_pct = round(100 * len(contract_ended[contract_ended['checkin_type'] == 'mobile']) / len(df[df['checkin_type'] == 'mobile']), 2)
    mobile_canceled_pct = round(100 * len(contract_canceled[contract_canceled['checkin_type'] == 'mobile']) / len(df[df['checkin_type'] == 'mobile']), 2)
    connect_ended_pct = round(100 * len(contract_ended[contract_ended['checkin_type'] == 'connect']) / len(df[df['checkin_type'] == 'connect']), 2)
    connect_canceled_pct = round(100 * len(contract_canceled[contract_canceled['checkin_type'] == 'connect']) / len(df[df['checkin_type'] == 'connect']), 2)
    total_ended_pct = round(100 * len(contract_ended) / len(df), 2)
    total_canceled_pct = round(100 * len(contract_canceled) / len(df), 2)
    # Create a new dataframe with the percentages computed above
    df_pct = pd.DataFrame({
        'Contract Type': ['Mobile', 'Connect', 'Overall'],
        'Ended': [mobile_ended_pct, connect_ended_pct, total_ended_pct],
        'Canceled': [mobile_canceled_pct, connect_canceled_pct, total_canceled_pct]
    })
    # Use Plotly to create a horizontal stacked bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_pct['Contract Type'], x=df_pct['Canceled'], orientation='h', name='Canceled', marker=dict(color='#c71414'), text=[f"{mobile_canceled_pct}%", f"{connect_canceled_pct}%", f"{total_canceled_pct}%"],
                        textposition='auto', textfont=dict(color='white')))
    fig.add_trace(go.Bar(y=df_pct['Contract Type'], x=df_pct['Ended'], orientation='h', name='Ended', marker=dict(color='#7b728e'), text=[f"{mobile_ended_pct}%", f"{connect_ended_pct}%", f"{total_ended_pct}%"],
                        textposition='auto', textfont=dict(color='white')))
    fig.update_layout(barmode='stack', title='How many contracts were canceled?', yaxis={'categoryorder':'total ascending'})
    # Show the chart
    st.plotly_chart(fig, theme="streamlit")



def get_chart_clients_late_to_return_car(df):
    # Compute the percentage of rentals were delayed for each contract type
    checkout_delayed = df[df['delay_at_checkout'] == 'checkout delayed']
    checkout_not_delayed = df[df['delay_at_checkout'] == 'checkout not delayed']
    mobile_delayed_pct = round(100 * len(checkout_delayed[checkout_delayed['checkin_type'] == 'mobile']) / len(df[df['checkin_type'] == 'mobile']), 2)
    mobile_not_delayed_pct = round(100 * len(checkout_not_delayed[checkout_not_delayed['checkin_type'] == 'mobile']) / len(df[df['checkin_type'] == 'mobile']), 2)
    connect_delayed_pct = round(100 * len(checkout_delayed[checkout_delayed['checkin_type'] == 'connect']) / len(df[df['checkin_type'] == 'connect']), 2)
    connect_not_delayed_pct = round(100 * len(checkout_not_delayed[checkout_not_delayed['checkin_type'] == 'connect']) / len(df[df['checkin_type'] == 'connect']), 2)
    total_delayed_pct = round(100 * len(checkout_delayed) / len(df), 2)
    total_not_delayed_pct = round(100 * len(checkout_not_delayed) / len(df), 2)
    # Create a new dataframe with the percentages computed above
    df_pct = pd.DataFrame({
        'Contract Type': ['Mobile', 'Connect', 'Overall'],
        'Delayed': [mobile_delayed_pct, connect_delayed_pct, total_delayed_pct],
        'Not Delayed': [mobile_not_delayed_pct, connect_not_delayed_pct, total_not_delayed_pct]
    })
    # Use Plotly to create a horizontal stacked bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_pct['Contract Type'], x=df_pct['Delayed'], orientation='h', name='Delayed', marker=dict(color='#c71414'), text=[f"{mobile_delayed_pct}%", f"{connect_delayed_pct}%", f"{total_delayed_pct}%"],
                        textposition='auto', textfont=dict(color='white')))
    fig.add_trace(go.Bar(y=df_pct['Contract Type'], x=df_pct['Not Delayed'], orientation='h', name='Not Delayed', marker=dict(color='#7b728e'), text=[f"{mobile_not_delayed_pct}%", f"{connect_not_delayed_pct}%", f"{total_not_delayed_pct}%"],
                        textposition='auto', textfont=dict(color='white')))
    fig.update_layout(barmode='stack', title='How many clients were late to return the car?', yaxis={'categoryorder':'total ascending'})
    # Show the chart
    st.plotly_chart(fig, theme="streamlit")


def get_chart_existing_deltas(df):
    df_grouped = df.groupby(by="time_delta_with_previous_rental_in_minutes").size().reset_index()
    df_grouped.rename(columns={0:'count'}, inplace=True)
    df_grouped["percentage"] = df_grouped["count"]/df_grouped["count"].sum() * 100
    df_grouped["cumulated_percentage"] = 0
    df_grouped.loc[0, "cumulated_percentage"] = df_grouped.loc[0, "percentage"]
    for i in range(1, len(df_grouped)):
        df_grouped.loc[i, "cumulated_percentage"] = df_grouped.loc[i, "percentage"] + df_grouped.loc[i-1, "cumulated_percentage"]
    fig = px.bar(df_grouped, x="time_delta_with_previous_rental_in_minutes", y="count", title="Currently Existing Deltas")
    fig.update_traces(marker_color="#d076ca")
    st.plotly_chart(fig, theme="streamlit")


def get_chart_percentage_of_contracts_impacted_by_deltas(df):
    df_grouped = df.groupby(by="time_delta_with_previous_rental_in_minutes").size().reset_index()
    df_grouped.rename(columns={0:'count'}, inplace=True)
    df_grouped["percentage"] = df_grouped["count"]/df_grouped["count"].sum() * 100
    df_grouped["cumulated_percentage"] = 0
    df_grouped.loc[0, "cumulated_percentage"] = df_grouped.loc[0, "percentage"]
    for i in range(1, len(df_grouped)):
        df_grouped.loc[i, "cumulated_percentage"] = df_grouped.loc[i, "percentage"] + df_grouped.loc[i-1, "cumulated_percentage"]
    fig = px.bar(df_grouped, x='time_delta_with_previous_rental_in_minutes', y='cumulated_percentage')
    fig.update_traces(marker_color="#5ecbdd")
    fig.update_layout(title='What percentage of contracts will be impacted by a given minimum delta?', xaxis=dict(title='Delta'), yaxis=dict(title='Percent'), bargap=0.10)
    st.plotly_chart(fig, theme="streamlit")


def get_chart_distribution_of_delays(df):
    delays_df = df[df['delay_at_checkout_in_minutes']>0]
    trace = go.Histogram(x=delays_df['delay_at_checkout_in_minutes'], nbinsx=90, marker=dict(color='blue'))
    layout = go.Layout(title="How are clients' delays distributed?", xaxis=dict(title='Delay at Checkout in Minutes'), yaxis=dict(title='Count'))
    fig = go.Figure(data=[trace], layout=layout)
    fig.update_layout(bargap=0.15)
    fig.update_traces(marker_color="#e39735")
    st.plotly_chart(fig, theme="streamlit")

def get_chart_distribution_of_delays_less_than_1000_min(df):
    delays_df = df[df['delay_at_checkout_in_minutes']>0]
    less_than_1000_min_delays = delays_df[delays_df['delay_at_checkout_in_minutes']<=1000]
    trace = go.Histogram(x=less_than_1000_min_delays['delay_at_checkout_in_minutes'], nbinsx=100, marker=dict(color='blue'))
    layout = go.Layout(title="How are clients' delays distributed? - A Closer Look", xaxis=dict(title='Delay at Checkout in Minutes'), yaxis=dict(title='Count'))
    fig = go.Figure(data=[trace], layout=layout)
    fig.update_layout(bargap=0.15)
    fig.update_traces(marker_color="#e39735")
    st.plotly_chart(fig, theme="streamlit")


def get_chart_cumulated_delays(df):
    thresholds = range(0, 1000, 10)
    bins = pd.cut(df['delay_at_checkout_in_minutes'], bins=thresholds, include_lowest=True)
    bin_counts = bins.value_counts(normalize=True, sort=False) * 100
    delay_percentages = bin_counts.to_frame().reset_index()
    delay_percentages.rename(columns={"delay_at_checkout_in_minutes":"percentage", "index":"interval"}, inplace=True)
    delay_percentages["cumulated_percentage"]=0
    delay_percentages["delta_threshold"]=10
    delay_percentages.loc[0, "cumulated_percentage"] = delay_percentages.loc[0, "percentage"]
    for i in range(1, len(delay_percentages)):
        delay_percentages.loc[i, "cumulated_percentage"] = delay_percentages.loc[i, "percentage"] + delay_percentages.loc[i-1, "cumulated_percentage"]
        delay_percentages.loc[i, "delta_threshold"] = delay_percentages.loc[i-1, "delta_threshold"] + 10
    fig = px.bar(delay_percentages, x='delta_threshold', y='cumulated_percentage')
    fig.update_layout(title='What percentage of delays can be offset by a given delta?', xaxis=dict(title='Delta'), yaxis=dict(title='Percent'), bargap=0.10)
    fig.update_traces(marker_color="#2ea2ea")
    st.plotly_chart(fig, theme="streamlit")


def get_chart_impacted_clients(prev_df):
    delayed_df=prev_df[prev_df["previous_client_late"]=="yes"]
    # Compute the percentage of rentals were impacted by delay for each contract type
    impacted = delayed_df[delayed_df['client_rental_start_impacted'] == 'yes']
    not_impacted = delayed_df[delayed_df['client_rental_start_impacted'] == 'no']
    mobile_impacted_pct = round(100 * len(impacted[impacted['checkin_type'] == 'mobile']) / len(delayed_df[delayed_df['checkin_type'] == 'mobile']), 2)
    mobile_not_impacted_pct = round(100 * len(not_impacted[not_impacted['checkin_type'] == 'mobile']) / len(delayed_df[delayed_df['checkin_type'] == 'mobile']), 2)
    connect_impacted_pct = round(100 * len(impacted[impacted['checkin_type'] == 'connect']) / len(delayed_df[delayed_df['checkin_type'] == 'connect']), 2)
    connect_not_impacted_pct = round(100 * len(not_impacted[not_impacted['checkin_type'] == 'connect']) / len(delayed_df[delayed_df['checkin_type'] == 'connect']), 2)
    total_impacted_pct = round(100 * len(impacted) / len(delayed_df), 2)
    total_not_impacted_pct = round(100 * len(not_impacted) / len(delayed_df), 2)
    # Create a new dataframe with the percentages computed above
    df_pct = pd.DataFrame({
        'Contract Type': ['Mobile', 'Connect', 'Overall'],
        'Impacted': [mobile_impacted_pct, connect_impacted_pct, total_impacted_pct],
        'Not Impacted': [mobile_not_impacted_pct, connect_not_impacted_pct, total_not_impacted_pct]
    })
    # Use Plotly to create a horizontal stacked bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_pct['Contract Type'], x=df_pct['Impacted'], orientation='h', name='Impacted', marker=dict(color='#c71414'), text=[f"{mobile_impacted_pct}%", f"{connect_impacted_pct}%", f"{total_impacted_pct}%"],
                        textposition='auto', textfont=dict(color='white')))
    fig.add_trace(go.Bar(y=df_pct['Contract Type'], x=df_pct['Not Impacted'], orientation='h', name='Not Impacted', marker=dict(color='#7b728e'), text=[f"{mobile_not_impacted_pct}%", f"{connect_not_impacted_pct}%", f"{total_not_impacted_pct}%"],
                        textposition='auto', textfont=dict(color='white')))
    fig.update_layout(barmode='stack', title='In cases where the previous client was late, how many clients were impacted by the delay?', yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, theme="streamlit")

def get_chart_impact_on_next_client(prev_df):
    impacted_df = prev_df[prev_df['client_rental_start_impacted']=="yes"]
    impact_less_900_min = impacted_df[impacted_df['delay_of_rental_start_in_minutes']<=900]
    # Create a histogram trace
    trace = go.Histogram(x=impact_less_900_min['delay_of_rental_start_in_minutes'], nbinsx=90, marker=dict(color='blue'))
    layout = go.Layout(title='How long did the clients have to wait before they could start their rental?', xaxis=dict(title='Delay of Rental Start in Minutes'), yaxis=dict(title='Count'))
    fig = go.Figure(data=[trace], layout=layout)
    fig.update_layout(bargap=0.15)
    fig.update_traces(marker_color="#ed4545")
    st.plotly_chart(fig, theme="streamlit")


def get_chart_cancellations_due_to_impact(prev_df):
    impacted_df = prev_df[prev_df['client_rental_start_impacted']=="yes"]
    impact_less_900_min = impacted_df[impacted_df['delay_of_rental_start_in_minutes']<=900]
    canceled_rental_start_delays = impact_less_900_min['delay_of_rental_start_in_minutes'][impact_less_900_min['state']=='canceled']
    not_canceled_rental_start_delays = impact_less_900_min['delay_of_rental_start_in_minutes'][impact_less_900_min['state']=='ended']
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=canceled_rental_start_delays, nbinsx=90, marker=dict(color='#c71414'), name='canceled rentals'))
    fig.add_trace(go.Histogram(x=not_canceled_rental_start_delays, nbinsx=90, marker=dict(color='#7b728e'), opacity=0.7, name='not canceled rentals'))
    fig.update_layout(barmode='stack', bargap=0.15, title='How many rentals were canceled depending on the delay of rental start?', xaxis=dict(title='Delay of Rental Start in Minutes'), yaxis=dict(title='Count'))
    st.plotly_chart(fig, theme="streamlit")


def get_chart_cancellations_by_impacted_clients(prev_df):
    impacted_df = prev_df[prev_df['client_rental_start_impacted']=="yes"]
    canceled = impacted_df[impacted_df['state'] == 'canceled']
    not_canceled = impacted_df[impacted_df['state'] == 'ended']
    mobile_canceled_pct = round(100 * len(canceled[canceled['checkin_type'] == 'mobile']) / len(impacted_df[impacted_df['checkin_type'] == 'mobile']), 2)
    mobile_not_canceled_pct = round(100 * len(not_canceled[not_canceled['checkin_type'] == 'mobile']) / len(impacted_df[impacted_df['checkin_type'] == 'mobile']), 2)
    connect_canceled_pct = round(100 * len(canceled[canceled['checkin_type'] == 'connect']) / len(impacted_df[impacted_df['checkin_type'] == 'connect']), 2)
    connect_not_canceled_pct = round(100 * len(not_canceled[not_canceled['checkin_type'] == 'connect']) / len(impacted_df[impacted_df['checkin_type'] == 'connect']), 2)
    total_canceled_pct = round(100 * len(canceled) / len(impacted_df), 2)
    total_not_canceled_pct = round(100 * len(not_canceled) / len(impacted_df), 2)
    # Creating a temporary dataframe
    df_pct = pd.DataFrame({
        'Contract Type': ['Mobile', 'Connect', 'Overall'],
        'Canceled': [mobile_canceled_pct, connect_canceled_pct, total_canceled_pct],
        'Not Canceled': [mobile_not_canceled_pct, connect_not_canceled_pct, total_not_canceled_pct]
    })
    # Creating a horizontal stacked bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_pct['Contract Type'], x=df_pct['Canceled'], orientation='h', name='Canceled', marker=dict(color='#c71414'), text=[f"{mobile_canceled_pct}%", f"{connect_canceled_pct}%", f"{total_canceled_pct}%"],
                        textposition='auto', textfont=dict(color='white')))
    fig.add_trace(go.Bar(y=df_pct['Contract Type'], x=df_pct['Not Canceled'], orientation='h', name='Not Canceled', marker=dict(color='#7b728e'), text=[f"{mobile_not_canceled_pct}%", f"{connect_not_canceled_pct}%", f"{total_not_canceled_pct}%"],
                        textposition='auto', textfont=dict(color='white')))
    fig.update_layout(barmode='stack', title='How many of the impacted clients canceled their rentals?', yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, theme="streamlit")


def funnel_chart_overall_cancel_due_to_delays(prev_df):
    # Counting the number of cases where the driver was late at checkout
    total_delays = len(prev_df[prev_df["previous_client_late"]=="yes"])
    connect_late = round(len(prev_df[(prev_df["checkin_type"]=="connect") 
                            & (prev_df["previous_client_late"]=="yes")]) / total_delays * 100, 2)
    mobile_late = round(len(prev_df[(prev_df["checkin_type"]=="mobile") 
                            & (prev_df["previous_client_late"]=="yes")]) / total_delays * 100, 2)
    mobile_late_impacted = round(len(prev_df[(prev_df["checkin_type"]=="mobile") 
                                    & (prev_df["previous_client_late"]=="yes") 
                                    & (prev_df["client_rental_start_impacted"]=="yes")]) / total_delays * 100, 2)
    connect_late_impacted = round(len(prev_df[(prev_df["checkin_type"]=="connect") 
                                    & (prev_df["previous_client_late"]=="yes") 
                                    & (prev_df["client_rental_start_impacted"]=="yes")]) / total_delays * 100, 2)
    mobile_late_impacted_canceled = round(len(prev_df[(prev_df["checkin_type"]=="mobile") 
                                    & (prev_df["previous_client_late"]=="yes") 
                                    & (prev_df["client_rental_start_impacted"]=="yes")
                                    & (prev_df["state"]=="canceled")]) / total_delays * 100)
    connect_late_impacted_canceled = round(len(prev_df[(prev_df["checkin_type"]=="connect") 
                                    & (prev_df["previous_client_late"]=="yes") 
                                    & (prev_df["client_rental_start_impacted"]=="yes")
                                    & (prev_df["state"]=="canceled")]) / total_delays * 100, 2)
    stages = ['canceled','next client impacted','previous client late']
    sr1 = [mobile_late_impacted_canceled, mobile_late_impacted, mobile_late] # values for mobile 
    sr2= [connect_late_impacted_canceled, connect_late_impacted, connect_late] # values for connect
    #convert sr1
    def convert(lst):
        return [ -i for i in lst ]
    sr3 = convert(sr2)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=stages, x=sr1,
                    base=0,
                    marker_color='#8d1586',
                    name='Mobile',
                    orientation='h',
                    text = sr1,
                    textposition='inside',
                    texttemplate = "%{x} %"
    ))
    fig.add_trace(go.Bar(y=stages, x=sr2,
                    base=sr3,
                    marker_color='#eec186',
                    name='Connect',
                    orientation='h',
                    text = sr2,
                    textposition='inside',
                    texttemplate = "%{x} %"
    ))
    fig.update_layout(barmode='overlay', xaxis_tickangle=45, title_text="For both check-in types, how many cancellations are attributable to delays?")
    st.plotly_chart(fig, theme="streamlit")


###### Beginning of script
st.markdown("---")
st.subheader("Contracts by check-in types")
st.markdown("""Getaround proposes two major types of check-in:
* Mobile rental agreement on native apps: driver and owner meet and both sign the rental agreement on the owner’s smartphone
* Connect: the driver doesn’t meet the owner and opens the car with his smartphone""")
get_chart_rentals_by_check_in_type(data)

st.markdown("---")
st.subheader("Cancellations by check-in types")
get_chart_cancellations_by_contract_type(data)
st.markdown("""It looks like 'connect' type rentals are more frequently cancelled than 'mobile' type rentals.""")

get_chart_clients_late_to_return_car(data)
st.markdown("""Overall, more than 40 percent of clients are late at the checkout, 
and clients for 'mobile' type rentals are more frequently late than 'connect' type rentals.""")

get_chart_impacted_clients(delta_df)
st.markdown("""Fortunately, existing deltas between rentals set by the car owners 
compensate for three-quarters of the delays. We consider that the client was not impacted by a previous's clients delay
 if the delay wasn't greater than the delta between rentals, so the client was able to start their rental on time""")

get_chart_cancellations_by_impacted_clients(delta_df)
st.markdown("""Still, a significant part of clients who are impacted by delays choose to cancel their rental. 
'Connect' type rentals are particularly sensitive to the delays. 
Supposedly, a client who chooses this contract type expects to be able to begin their rental without having to wait, 
while clients who choose 'mobile' type rental check-in expect that some time will be spent for meeting the owner and signing the contract.""")

funnel_chart_overall_cancel_due_to_delays(delta_df)
st.markdown("""At the same time, it still should be noted that majority of cancellations do not seem to be attributable to delays at checkout.""")

st.markdown("---")
st.subheader("Currently, how are the clients' delays distributed?")
get_chart_distribution_of_delays(data)
st.markdown("""It looks like the majority of delays are shorter than 1000 minutes. 
Let us have a look at their distribution.""")
get_chart_distribution_of_delays_less_than_1000_min(data)

st.markdown("---")
get_chart_impact_on_next_client(delta_df)

st.markdown("---")
get_chart_cancellations_due_to_impact(delta_df)

st.markdown("---")
st.subheader("Compensating delays: deltas between rentals")
st.markdown("""Let us look at the current deltas between rentals chosen by car owners.""")
get_chart_existing_deltas(data)

st.markdown("""Let us look at the thresholds necessary to compensate for clients' delays.""")
get_chart_cumulated_delays(data)

st.markdown("""Let us look how many deltas are there that are shorter than a given threshold.
If we impose a threshold now, it might mean that rentals where the owner proposes a smaller delta will not have been contracted.""")
get_chart_percentage_of_contracts_impacted_by_deltas(data)


### Footer 
empty_space, footer = st.columns([1, 2])

with empty_space:
    st.write("")

with footer:
    st.markdown("""
        If you want to learn more, check out:
        * [Getaround Rental Price API: predictions based on state-of-the-art machine learning model](https://getaround-api-15032023.herokuapp.com/docs)
        * [Getaround Rental Price MLFlow Server: comparison of machine learning models](https://getaround.herokuapp.com/)
    """)