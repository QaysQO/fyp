
$pip install yfinance

import datetime
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import streamlit as st

# Streamlit page configuration
st.set_page_config(page_title="Bank Stocks Market", page_icon=":bar_chart:", layout="wide")

# Define stock tickers and mapping
stocks = ('1155.KL', '5819.KL', '1066.KL', '1295.KL')
stock_mapping = {
    '1155.KL': 'Malayan Bank', 
    '5819.KL': 'Hong Leong Bank', 
    '1066.KL': 'RHB Bank', 
    '1295.KL': 'Public Bank'
}
stocks = tuple(stock_mapping[code] for code in stocks)

# Define a consistent color palette
base_color_palette = {
    'Malayan Bank': ['red', '#ff7f0e', '#2ca02c', '#d62728'],  # Red, Orange, Green, Purple
    'Hong Leong Bank': ['blue', '#89cff0', '#e377c2', '#7f7f7f'],  # Blue, Baby Blue, Pink, Gray
    'RHB Bank': ['green', '#17becf', '#aec7e8', '#ffbb78'],  # Green, Teal, Light Blue, Light Orange
    'Public Bank': ['orange', '#ff9896', '#c5b0d5', '#c49c94']  # Orange, Light Red, Light Purple, Light Brown
}

START = "2019-01-01"
TODAY = datetime.date.today().strftime("%Y-%m-%d")

# Function to load stock data with caching
@st.cache_data
def load_data(ticker):
    reverse_mapping = {v: k for k, v in stock_mapping.items()}
    data = yf.download(reverse_mapping[ticker], START, TODAY)
    data.reset_index(inplace=True)
    return data

# Function to plot metrics
def plot_metric(label, value, prefix="", suffix="", comparison_value=None):
    text_color = "black"
    if comparison_value is not None:
        text_color = "blue" if value > comparison_value else "red"
    
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        value=value,
        gauge={"axis": {"visible": False}},
        number={"prefix": prefix, "suffix": suffix, "font": {"size": 28, "color": text_color}},
        title={"text": label, "font": {"size": 24, "color": text_color}}
    ))
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        margin=dict(t=50, b=50, l=50, r=50),
        showlegend=False,
        plot_bgcolor="grey",
        height=150,
        autosize=True
    )
    return fig

# Function to plot raw data
def plot_raw_data(data_list, tag_options, selected_stocks, base_color_palette):
    fig = go.Figure()
    for i, (data, stock) in enumerate(zip(data_list, selected_stocks)):
        colors = base_color_palette[stock]
        color_dict = {tag: colors[j] for j, tag in enumerate(tag_options)}
        if 'Open' in tag_options:
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name=f"{stock} Open", line=dict(color=color_dict['Open'])))
        if 'Close' in tag_options:
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name=f"{stock} Close", line=dict(color=color_dict['Close'])))
        if 'High' in tag_options:
            fig.add_trace(go.Scatter(x=data['Date'], y=data['High'], name=f"{stock} High", line=dict(color=color_dict['High'])))
        if 'Low' in tag_options:
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Low'], name=f"{stock} Low", line=dict(color=color_dict['Low'])))
    
    fig.update_layout(
        title="Banks Stock Market Close Price",
        xaxis_rangeslider_visible=True,
        template='plotly_white',
        height=400
    )
    return fig

# Function to apply event impact on stock data
def apply_event_impact(data, event_start_date, event_end_date, impact_percentage):
    data['Date'] = pd.to_datetime(data['Date'])
    impact_mask = (data['Date'] >= pd.to_datetime(event_start_date)) & (data['Date'] <= pd.to_datetime(event_end_date))
    data.loc[impact_mask, ['Open', 'Close', 'High', 'Low']] *= (1 + impact_percentage / 100)
    return data

# Function to prepare data for forecasting
def prepare_data(stock_data, train_percentage):
    df_train = stock_data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    df_train['ds'] = pd.to_datetime(df_train['ds'])
    df_train = df_train[df_train['ds'].dt.weekday < 5]
    train_size = int(len(df_train) * (train_percentage / 100))
    df_train = df_train.sort_values(by='ds', ascending=False).reset_index(drop=True)[:train_size]
    return df_train, stock_data

# Home page function
def home_page():
    st.title("Welcome to the Banks Stocks Market Prediction Dashboard")
    st.write("""
        This dashboard allows you to select a bank stock dataset and predict future stock prices.
        Please select a dataset from the sidebar to get started.
    """)

    with st.sidebar.expander("Data/Prediction Settings"):
        selected_stocks = st.multiselect('Select datasets for prediction', stocks, default=stocks)
        forecast_date = st.date_input('Select a prediction end date:', value=datetime.date.today() + datetime.timedelta(days=365))
        train_percentage = st.slider('Training data percentage:', 80, 100, step=5)
        tag_options = st.multiselect('Select tags for the dataset', ['Open', 'Close', 'High', 'Low'], default=['Close'])
        
        event_start_date = st.date_input('Select an event start date:(To manipulate data)', value=datetime.date.today() - datetime.timedelta(days=365))
        event_end_date = st.date_input('Select an event end date:', value=datetime.date.today())
        impact_percentage = st.slider('Event impact percentage:', -20, 20, step=1, value=0)

    if not selected_stocks:
        st.warning("Please select a dataset for prediction.")
    else:
        data_list = [load_data(stock) for stock in selected_stocks]
        original_data_list = [data.copy() for data in data_list]
        data_list = [apply_event_impact(data, event_start_date, event_end_date, impact_percentage) for data in data_list]
        recent_open_prices = {stock: data_list[i][data_list[i]['Date'] == data_list[i]['Date'].max()]['Open'].iloc[0] for i, stock in enumerate(selected_stocks)}

        st.title("Bank Stocks Market Prediction for Selected Stocks")

        row1_1, row1_2, row1_3, row1_4 = st.columns((1, 1, 1, 1))
        metrics_open = [plot_metric(f"{stock} Current Price", recent_open_prices[stock]) for stock in selected_stocks]

        with row1_1:
            st.plotly_chart(metrics_open[0], use_container_width=True)
        with row1_2:
            if len(metrics_open) > 1:
                st.plotly_chart(metrics_open[1], use_container_width=True)
        with row1_3:
            if len(metrics_open) > 2:
                st.plotly_chart(metrics_open[2], use_container_width=True)
        with row1_4:
            if len(metrics_open) > 3:
                st.plotly_chart(metrics_open[3], use_container_width=True)

        fig = plot_raw_data(data_list, tag_options, selected_stocks, base_color_palette=base_color_palette)
        row2_1, row2_2, row2_3 = st.columns((3.3, 0.3, 2))

        with row2_1:
            st.plotly_chart(fig, use_container_width=True)

        with row2_3:
            st.subheader('Raw Data')
            combined_data = pd.concat(data_list, keys=selected_stocks, names=['Bank', 'Index']).reset_index(level=0)
            combined_data = combined_data.sort_values(by='Date', ascending=False)
            st.dataframe(combined_data)

        forecast_fig = go.Figure()
        predicted_prices = []

        for stock, stock_data in zip(selected_stocks, data_list):
            df_train, df_test = prepare_data(stock_data, train_percentage)
            
            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=(forecast_date - datetime.date.today()).days)
            future = future[future['ds'].dt.weekday < 5]
            forecast = m.predict(future)

            actual_data = df_test.set_index('Date')
            forecast_fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat'], 
                mode='lines', 
                name=f"{stock} Prediction",
                line=dict(color=base_color_palette[stock][1], dash='dash')
            ))

            forecast_fig.add_trace(go.Scatter(
                x=actual_data.index, y=actual_data['Close'], 
                mode='markers', 
                name=f"{stock} Actual",
                marker=dict(color=base_color_palette[stock][0], size=3)
            ))

            nearest_date = forecast['ds'].dt.date.iloc[-1]
            predicted_price = forecast.loc[forecast['ds'].dt.date == nearest_date, 'yhat'].values[0]
            predicted_prices.append((stock, predicted_price))

        row3_1, row3_2, row3_3, row3_4 = st.columns((1, 1, 1, 1))
        predicted_metrics = [plot_metric(f"{stock} Predicted Price", predicted_price, comparison_value=recent_open_prices[stock]) for stock, predicted_price in predicted_prices]

        with row3_1:
            st.plotly_chart(predicted_metrics[0], use_container_width=True)
        with row3_2:
            if len(predicted_metrics) > 1:
                st.plotly_chart(predicted_metrics[1], use_container_width=True)
        with row3_3:
            if len(predicted_metrics) > 2:
                st.plotly_chart(predicted_metrics[2], use_container_width=True)
        with row3_4:
            if len(predicted_metrics) > 3:
                st.plotly_chart(predicted_metrics[3], use_container_width=True)

        row4_1 = st.columns((1,))
        row4_1[0].subheader(f'Prediction Chart for Selected Stocks')

        forecast_fig.update_layout(
            title="Prediction & Actual Line Chart for All Selected Stocks",
            xaxis_title="Date",
            yaxis_title="Stock Price",
            xaxis_range=['2021-10-01', forecast_date.strftime("%Y-%m-%d")],
            height=600,
            template='plotly_white'
        )

        row4_1[0].plotly_chart(forecast_fig, use_container_width=True)

# Function for individual dataset page
def dataset_page(selected_stock):
    data = load_data(selected_stock)

    st.title(f"{selected_stock} Stock Analysis and Prediction")

    with st.sidebar.expander("Data/Prediction Settings"):
        forecast_date = st.date_input('Select a prediction end date:', value=datetime.date.today() + datetime.timedelta(days=365), key=f'forecast_date_{selected_stock}')
        train_percentage = st.slider('Training data percentage:', 80, 100, step=5, key=f'train_percentage_{selected_stock}')
        tag_options = st.multiselect('Select tags for the dataset', ['Open', 'Close', 'High', 'Low'], default=['Open'], key=f'tags_{selected_stock}')
        show_forecast_components = st.checkbox("Prediction Summary Components", key=f'components_{selected_stock}')

        event_start_date = st.date_input('Select an event start date:(To Manipulate Data)', value=datetime.date.today() - datetime.timedelta(days=365), key=f'event_start_date_{selected_stock}')
        event_end_date = st.date_input('Select an event end date:', value=datetime.date.today(), key=f'event_end_date_{selected_stock}')
        impact_percentage = st.slider('Event impact percentage:', -20, 20, step=1, value=0, key=f'impact_percentage_{selected_stock}')

    original_data = data.copy()
    data = apply_event_impact(data, event_start_date, event_end_date, impact_percentage)
    post_event_mask = data['Date'] > pd.to_datetime(event_end_date)
    data.loc[post_event_mask, ['Open', 'Close', 'High', 'Low']] = original_data.loc[post_event_mask, ['Open', 'Close', 'High', 'Low']]

    df_train, df_test = prepare_data(data, train_percentage)
            
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=(forecast_date - datetime.date.today()).days)
    future = future[future['ds'].dt.weekday < 5]
    forecast = m.predict(future)

    nearest_date = forecast['ds'].dt.date.iloc[-1]
    predicted_price = forecast.loc[forecast['ds'].dt.date == nearest_date, 'yhat'].values[0]
    metric_predicted_price = plot_metric("Prediction Price", predicted_price, comparison_value=data[data['Date'] == data['Date'].max()]['Close'].iloc[0])

    recent_prices = {
        "Close": data[data['Date'] == data['Date'].max()]['Close'].iloc[0],
        "High": data[data['Date'] == data['Date'].max()]['High'].iloc[0],
        "Low": data[data['Date'] == data['Date'].max()]['Low'].iloc[0]
    }

    row1_1, row1_2, row1_3, row1_4 = st.columns(4)

    with row1_1:
        st.plotly_chart(plot_metric("Current Close Price", recent_prices["Close"]), use_container_width=True)
    with row1_2:
        st.plotly_chart(plot_metric("Current High Price", recent_prices["High"]), use_container_width=True)
    with row1_3:
        st.plotly_chart(plot_metric("Current Low Price", recent_prices["Low"]), use_container_width=True)
    with row1_4:
        st.plotly_chart(metric_predicted_price, use_container_width=True)

    row2_1, row2_2, row2_3 = st.columns((3.3, 0.3, 2))
    fig = plot_raw_data([data], tag_options, [selected_stock], base_color_palette)

    with row2_1:
        st.plotly_chart(fig, use_container_width=True)

    with row2_3:
        st.subheader('Raw Data')
        st.dataframe(data)

    row3_1 = st.columns((1,))
    row3_1[0].subheader(f'Prediction Chart')
    forecast_filtered = forecast[forecast['ds'] >= '2020-01-01']

    fig1 = plot_plotly(m, forecast_filtered)
    fig1.update_layout(showlegend=True, height=400, autosize=True, template='plotly_white',
                       xaxis_range=['2020-01-01', forecast_date.strftime("%Y-%m-%d")],
                       xaxis_title="Date",
                       yaxis_title="Stock Price")  # Update axis titles here
    row3_1[0].plotly_chart(fig1, use_container_width=True)

    st.subheader('Prediction data')
    st.write(""" Crucial highlight to:
         ||ds: Upcoming Date || yhat: Predicted price || yhat_lower: lowest predicted price|| yhat_upper: highest predicted price ||
    """)
    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper', 'weekly', 'weekly_lower', 'weekly_upper', 'yearly', 'yearly_lower', 'yearly_upper']].tail(50)
    st.write(forecast_display)

    fig2 = m.plot_components(forecast)
    if show_forecast_components:
        st.write("Prediction Summary components")
        st.write(fig2)

# Sidebar menu
st.sidebar.title("Menu")

# Dropdown menu for selecting datasets
selected_page = st.sidebar.selectbox("Choose a dataset", ["Home", *stocks])

# Page routing based on selection
if selected_page == "Home":
    home_page()
elif selected_page != "Select a dataset":
    dataset_page(selected_page)
