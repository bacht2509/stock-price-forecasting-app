import streamlit as st 
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import datetime
import pydtmc as mc
from sklearn.metrics import accuracy_score, precision_score, recall_score, 	root_mean_squared_error, mean_absolute_percentage_error
from collections import Counter
import random
import statsmodels.api as sm
from statsmodels.tsa.api import Holt, SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller 
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import pydot
import graphviz
from prophet import Prophet
from prophet.plot import plot_plotly

def main():
    #Page config and title
    st.set_page_config(page_title="Stock Price Forecast", page_icon=":chart_with_upwards_trend:", layout="wide")
    st.title("Stock Price Forecasting App :desktop_computer:")

    #Sidebar
    st.sidebar.header("Select :pencil2:")
    group_list = ["Banking", "Securities", "Electronics", "Petroleum", "Public Investment", "Steel"]
    ticker_list = {"Banking": ["BID.VN", "VCB.VN", "TCB.VN", "CTG.VN"],
                   "Securities": ["AGR.VN", "APG.VN", "BSI.VN"],
                   "Electronics": ["BTP.VN", "CHP.VN", "DRL.VN", "CAV.VN"],
                   "Petroleum": ["ASP.VN", "CNG.VN"],
                   "Public Investment": ["C47.VN", "CII.VN", "CTD.VN", "CTI.VN", "CTR.VN"],
                   "Steel": ["DTL.VN", "HMC.VN", "HPG.VN", "HSG.VN"]}
    group = st.sidebar.selectbox("Industry Group :hammer_and_wrench:", group_list)
    ticker = st.sidebar.selectbox("Stock Ticker :ticket:", ticker_list[group])
    model_list = ["Markov Chain", "ARIMA", "Facebook Prophet", "Markov Regime-Switching"]
    model_choice = st.sidebar.selectbox("Forecasting Model :slot_machine:", model_list)

    start_date = date(2022, 1, 1)
    end_date = date.today()

    data = get_data(ticker, start_date, end_date)

    tab1, tab2 = st.tabs(["Data", "Forecast"])

    with tab1: 
        #View data frame
        st.header("Stock Data")

        price_difference, percentage_difference = calculate_price_difference(data)
        latest_close_price = data.iloc[-1]["Close"]
        max_52_week_high = data["High"].tail(252).max() if len(data) >= 252 else data["High"].max()
        min_52_week_low = data["Low"].tail(252).min() if len(data) >= 252 else data["Low"].min()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Close Price", f"{latest_close_price:.2f}")
        with col2:
            st.metric("Price Difference (YoY)", f"{price_difference:.2f}", f"{percentage_difference:+.2f}%")
        with col3:
            st.metric("52-Week High", f"{max_52_week_high:.2f}")
        with col4:
            st.metric("52-Week Low", f"{min_52_week_low:.2f}")

        st.dataframe(data[::-1], hide_index=True, use_container_width=True)
        

        #Plot the data
        st.header("Data Visualization :chart:")
        st.subheader("Close price with moving averages")
        #Calculate moving averages
        ma = st.slider("Choose period of moving average", 1, 30, 7)
        data_view = data.copy(deep=True)
        data_view["MA"] = calculate_moving_average(data_view["Close"], ma)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_view["Date"], y=data_view['Close'], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=data_view["Date"], y=data_view['MA'], mode='lines', line = {'color': '#fc5a03'}, name=f'MA{ma}'))
        fig.update_layout(xaxis_title="Date",yaxis_title="Close",legend=dict(x=0,y=1,traceorder="normal"),font=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Candlestick Chart')
        candlestick = go.Candlestick(x=data_view["Date"],
                                    open=data_view['Open'],
                                    high=data_view['High'],
                                    low=data_view['Low'],
                                    close=data_view['Close'])
        candlestick_fig = go.Figure(data=candlestick)
        candlestick_fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(candlestick_fig, use_container_width=True)    

        # Volume plot
        st.subheader("Volume Plot")
        volume_fig = go.Figure()
        volume_fig.add_trace(go.Bar(x=data_view["Date"], y=data_view['Volume'], name='Volume'))
        st.plotly_chart(volume_fig, use_container_width=True)

    with tab2:
        #Forecasting
        data1 = data[["Date", "Close"]]

        #Markov chain
        if model_choice == "Markov Chain":
            # Markov chain 1: Trend states
            mc_data = data1.copy(deep=True)
            mc_data["Close_diff"] = mc_data["Close"].diff()
            mc_data.loc[mc_data["Close_diff"] > 0, "Trend_state"] = "up"
            mc_data.loc[mc_data["Close_diff"] == 0, "Trend_state"] = "flat"
            mc_data.loc[mc_data["Close_diff"] < 0, "Trend_state"] = "down"

            mc_train = mc_data[:int(len(mc_data)*0.8)]
            mc_test = mc_data[int(len(mc_data)*0.8):]

            trend_mc = mc.MarkovChain.fit_sequence("mle", ["up", "flat", "down"], 
                                            mc_train["Trend_state"].tolist()[1:])
        
        
            st.subheader("Markov Chain model ")
            st.subheader(r"$\textsf{\small Markov Chain of Stock Trends}$")
        
            plt.ioff()
            fig, ax = mc.plot_graph(trend_mc, dpi = 100)
            fig.set_size_inches(10, 4)
            st.pyplot(use_container_width = False)

        
            st.markdown("**Markov chain properties check**")
            prop_list = [[trend_mc.is_irreducible, trend_mc.is_aperiodic, trend_mc.recurrent_states, trend_mc.transient_states]]
            prop_df = pd.DataFrame(prop_list, columns=["Irreducible", "Aperiodic", "Recurrent states", "Transient states"])
            st.dataframe(prop_df, hide_index=True)

            st.subheader(r"$\textsf{\footnotesize Trend forecast}$")
            steps = st.slider("Choose number of days to forecast", 1, 20, 5)
            pred = make_pred_df(trend_mc, ["up", "flat", "down"], [0,1,0], steps)
            actual = mc_test[:steps][["Date", "Trend_state"]].reset_index(drop=True)
            compare = pd.concat([pred, actual], axis=1)
            compare = compare[['Date', "up_prob", "flat_prob", "down_prob", "Prediction", "Trend_state"]]
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                st.dataframe(compare, hide_index=True, use_container_width=True)
            with col2:
                st.metric(":orange[Accuracy]", f"{accuracy_score(compare['Trend_state'], compare['Prediction'])*100:.2f}%")
            
            if trend_mc.is_ergodic:
                st.subheader(r"$\textsf{\footnotesize Long-run proportion forecast}$")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Forecast**")
                    pi_df = pd.DataFrame(trend_mc.pi, columns=["up", "flat", "down"])
                    st.dataframe(pi_df, hide_index=True)
                with col2:
                    st.markdown(f"**Actual from {mc_test.iloc[0,0].date()} to {mc_test.iloc[-1,0].date()} data**")
                    actual_proportion = cal_proportion(mc_test, "Trend_state", ["up", "flat", "down"])
                    st.dataframe(actual_proportion, hide_index=True)

                pi_df_long = pi_df.melt(var_name='state', value_name='prop')
                actual_prop_long = actual_proportion.melt(var_name='state', value_name='prop')

                pi_df_long['name'] = 'Forecast'
                actual_prop_long['name'] = 'Actual'

                combined = pd.concat([pi_df_long, actual_prop_long], ignore_index=True)

                fig = px.bar(combined, x='state', y='prop', color='name', barmode='group', 
                    title='Comparison of long-run proportion between forecast and actual',
                    labels={'state': 'State', 'prop': 'Proportion of time', 'name': 'Name'})
                st.plotly_chart(fig)
            
            st.subheader(r"$\textsf{\footnotesize Some useful insights}$")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Expected return time**")
                st.dataframe(pd.DataFrame(trend_mc.mrt().reshape(1,3), columns=['up', 'flat', 'down']), hide_index=True)
            with col2:
                st.markdown("**Expected hitting times**")
                efpt = pd.DataFrame(trend_mc.mfptt(), columns=['up', 'flat', 'down'])
                efpt = efpt.rename(index={0: 'up', 1: 'flat', 2: 'down'})
                st.dataframe(efpt)
            with col3:
                st.markdown("**Expected numbers of visits**")
                steps = st.number_input("Choose number of steps", 1, 10, 5)
                evn = pd.DataFrame(exp_visits_num(steps, trend_mc), columns=['up', 'flat', 'down'])
                evn = evn.rename(index={0: 'up', 1: 'flat', 2: 'down'})
                st.dataframe(evn)
            
            
            
            # Markov chain 2: Daily return states
            st.subheader(r"$\textsf{\small Predict stock closing price}$")
            #Calculate daily return
            mc_data2 = data1.copy(deep=True)
            mc_data2["Daily return"] = mc_data2["Close"].diff()/mc_data2["Close"]
            #Get min and max daily return in train set
            max_dr = mc_data2["Daily return"][:int(len(mc_data2)*0.8)].max()
            min_dr = mc_data2["Daily return"][:int(len(mc_data2)*0.8)].min()
            #Define the states by splitting the interval [minDR, maxDR]
            up_states = []
            down_states = []
            #Split [0,max] into 3 parts with a ratio of 4:3:3
            up_states = split_range(0, max_dr, 3, [0.4, 0.3, 0.3])
            #Split [min,0) into 3 parts with a ratio of 3:3:4
            down_states = split_range(min_dr, 0, 3, [0.3, 0.3, 0.4])
            states = [down_states[0], down_states[1], down_states[2], up_states[0], up_states[1], up_states[2]]
            #Assign state to each data point
            mc_data2["DR State"] = mc_data2["Daily return"].apply(lambda x: get_state(x, states))
            #Split into train and test set
            mc_train2 = mc_data2[1:int(len(mc_data2)*0.8)].reset_index (drop=True)
            mc_test2 = mc_data2[int(len(mc_data2)*0.8):]
            # mc_train2["MA"] = calculate_moving_average(mc_train2["Close"], 15)
            # mc_train2["Daily return"] = mc_train2["MA"].diff()/mc_train2["MA"]

            #Window sizes from 5 to 30 are validated 
            window_size_list = range(5, 31, 1)
            #List to record error (RMSE) of each window size
            list_error = []
            for window_size in window_size_list:
                start_index = 0
                prediction = []
                true_value = []
                error = 0
                while(start_index + window_size < len(mc_train2)):
                    #Get current state and closing price
                    cur_state = mc_train2["DR State"][start_index+window_size-1]
                    cur_price = mc_train2["Close"][start_index+window_size-1]
                    #Predict price for the next day
                    price = predict_price(cur_state, cur_price, ['0', '1', '2', '3', '4', '5'], 
                                states, mc_train2["DR State"][start_index:start_index+window_size].tolist())
                    prediction.append(price)
                    true_value.append(mc_train2["Close"][start_index+window_size])
                    #Shift the window
                    start_index = start_index+window_size+1
                #Get the error
                error = root_mean_squared_error(true_value, prediction)
                list_error.append(error)
            #Choose the optimal window size
            best_ws = window_size_list[np.argmin(list_error)]

            begin = mc_test2.index[0]
            end = mc_test2.index[len(mc_test2)-1]
            cur_id = begin
            cur_state = mc_data2["DR State"][begin-1]
            cur_price = mc_data2["Close"][begin-1]

            pred_price = []
            while cur_id <= end:
                price = predict_price(cur_state, cur_price, ['0', '1', '2', '3', '4', '5'], 
                                states, mc_data2["DR State"][cur_id-best_ws:cur_id].tolist())
                pred_price.append(price)
                cur_state = mc_data2["DR State"][cur_id]
                cur_price = mc_data2["Close"][cur_id]
                cur_id += 1
            mc_test2["Prediction"] = pred_price
            st.subheader(r"$\textsf{\footnotesize Performance}$")

            rmse = root_mean_squared_error(mc_test2["Close"], mc_test2["Prediction"])
            mape = mean_absolute_percentage_error(mc_test2["Close"], mc_test2["Prediction"])
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=mc_test2["Date"], y=mc_test2['Close'], mode='lines', line = {'color': 'blue'}, name='Test'))
                fig2.add_trace(go.Scatter(x=mc_test2["Date"], y=mc_test2['Prediction'], mode='lines', line = {'color': '#fc5a03'}, name='Prediction'))
                fig2.update_layout(xaxis_title="Date",yaxis_title="Close Price",legend=dict(x=0,y=1,traceorder="normal"),font=dict(size=12))
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                st.metric(":green[Best window size]", best_ws)
                st.metric(":orange[Root Mean Squared Error]", f'{rmse:.2f}')
                st.metric(":orange[Mean Absolute Percentage Error]", f'{mape*100:.2f}%')

            st.subheader(r"$\textsf{\footnotesize Prediction for next day}$")
            new_date = mc_data2["Date"][len(mc_data2)-1] + datetime.timedelta(days=1)
            if new_date.strftime("%A") == "Saturday":
                new_date = new_date + datetime.timedelta(days=2)
            if new_date.strftime("%A") == "Sunday":
                new_date = new_date + datetime.timedelta(days=1)
            new_price = predict_price(mc_data2["DR State"][len(mc_data2)-1], mc_data2["Close"][len(mc_data2)-1], 
                                               ['0', '1', '2', '3', '4', '5'], states, mc_data2["DR State"][len(mc_data2)-best_ws:].tolist())
            new_prediction = pd.DataFrame({"Date": new_date, "Prediction": new_price}, index=[0])
            st.dataframe(new_prediction, hide_index=True)

        if model_choice == "ARIMA":
            arima_data = data1.copy(deep=True)
            arima_data2 = arima_data.set_index("Date")

            st.subheader("ARIMA model")

            st.subheader(r"$\textsf{\footnotesize Test for stationarity}$")
            test_stationarity(arima_data2["Close"])

            #st.subheader("Trend and Seasonality")
            #decomposition = seasonal_decompose(arima_data2["Close"], period=30)
            #fig = plt.figure()
            #fig = decomposition.plot()
            #fig.set_size_inches(16, 9)
            #st.pyplot(fig, use_container_width=True)

            arima_train = arima_data.iloc[:int(arima_data.shape[0]*0.8)]
            arima_test = arima_data.iloc[int(arima_data.shape[0]*0.8):]
            y_pred = arima_test.copy(deep=True)

            model_arima = auto_arima(arima_train["Close"], start_p=0, start_q=0,
                                    test='adf',       # use adftest to find optimal 'd'
                                    max_p=3, max_q=3, # maximum p and q
                                    m=1,              # frequency of series
                                    d=None,           # let model determine 'd'
                                    seasonal=False,   # no seasonality
                                    start_P=0, 
                                    D=0, 
                                    trace=True,
                                    error_action='ignore',  
                                    suppress_warnings=True, 
                                    stepwise=True)
            model_arima.fit(arima_train["Close"])

            prediction_arima, confint = model_arima.predict(len(arima_test), alpha = 0.05, return_conf_int=True) 
            y_pred["Prediction"] = prediction_arima
            y_pred["Lower"] = confint[:, 0]
            y_pred["Upper"] = confint[:, 1]

            

            st.subheader(r"$\textsf{\footnotesize Performance}$")

            rmse_arima = root_mean_squared_error(y_pred["Close"], y_pred["Prediction"])
            mape_arima = mean_absolute_percentage_error(y_pred["Close"], y_pred["Prediction"])
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                plt.figure(figsize=(10,5))
                plt.plot(arima_train["Date"], arima_train["Close"], label="Training")
                plt.plot(arima_test["Date"], arima_test["Close"], color = 'blue', label="Test")
                plt.plot(y_pred["Date"], y_pred["Prediction"], color = 'orange', label="Prediction")
                plt.fill_between(y_pred["Date"], y_pred["Lower"], y_pred["Upper"], color='k', alpha=.10)
                plt.legend(loc='upper left', fontsize=8)
                plt.xlabel('Date')
                plt.ylabel('Close Price')
                st.pyplot(use_container_width=True)
            with col2:
                st.metric(":orange[Root Mean Squared Error]", f"{rmse_arima:.2f}")
                st.metric(":orange[Mean Absolute Percentage Error]", f'{mape_arima*100:.2f}%')

            st.subheader(r"$\textsf{\footnotesize Prediction for future}$")
            num_days = st.slider("Choose number of days to predict", 1, 15, 5)
            arima_new_date=[]
            arima_new_prediction=[]
            new_date = arima_data["Date"][len(arima_data)-1]
            for i in range(1, num_days+1):
                new_date = new_date + datetime.timedelta(days=1)
                if new_date.strftime("%A") == "Saturday":
                    new_date = new_date + datetime.timedelta(days=2)
                if new_date.strftime("%A") == "Sunday":
                    new_date = new_date + datetime.timedelta(days=1)
                arima_new_date.append(new_date)
            predictions, confint = model_arima.predict(len(arima_test)+num_days, alpha = 0.05, return_conf_int=True)
            predictions = predictions.reset_index(drop=True)
            lower = confint[:, 0]
            lower = lower[len(lower)-num_days:]
            upper = confint[:, 1]
            upper = upper[len(upper)-num_days:]
            arima_new_prediction = predictions[len(predictions)-num_days:]
    
            pd.set_option('display.float_format', lambda x: '%.2f' % x)
            model_predictions=pd.DataFrame(zip(arima_new_date, arima_new_prediction, lower, upper), columns=["Date","Prediction", "Lower", "Upper"])
            st.dataframe(model_predictions, hide_index=True)
        
        if model_choice == "Facebook Prophet":
            data3 = data.copy(deep=True)
            data3 = data3[["Date", "Close"]]
            data3.columns = ['ds', 'y']
            data3_train = data3.iloc[:int(data3.shape[0]*0.8)]
            data3_test = data3.iloc[int(data3.shape[0]*0.8):]

            prophet=Prophet(daily_seasonality=True)
            prophet.fit(data3_train)

            future_dates = prophet.make_future_dataframe(periods=365)
            predictions= prophet.predict(future_dates)
            pred=predictions[predictions['ds'].isin(data3_test['ds'])]

            st.subheader("Facebook Prophet model")

            st.subheader(r"$\textsf{\footnotesize Performance}$")

            plot = plot_plotly(prophet, predictions)
            st.plotly_chart(plot, use_container_width=True)

            rmse = root_mean_squared_error(data3_test["y"], pred["yhat"])
            mape = mean_absolute_percentage_error(data3_test["y"], pred["yhat"])
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                plt.figure(figsize=(10,5))
                plt.plot(pd.to_datetime(data3_train['ds']), data3_train['y'], label='Trainng')
                plt.plot(pd.to_datetime(data3_test['ds']), data3_test['y'], color='blue', label='Test')
                plt.plot(pd.to_datetime(data3_test['ds']), pred['yhat'], color = 'orange', label='Prediction')
                plt.fill_between(pred["ds"], pred["yhat_lower"], pred["yhat_upper"], color='k', alpha=.10)
                plt.legend()
                st.pyplot(use_container_width=True)
            with col2:
                st.metric(":orange[Root Mean Squared Error]", f"{rmse:.2f}")
                st.metric(":orange[Mean Absolute Percentage Error]", f'{mape*100:.2f}%')

            st.subheader(r"$\textsf{\footnotesize Prediction for future}$")
            num_days = st.slider("Choose number of days to predict", 1, 15, 5)
            dates=[]
            new_date = data3["ds"][len(data3)-1]

            for i in range(1, num_days+1):
                new_date = new_date + datetime.timedelta(days=1)
                if new_date.strftime("%A") == "Saturday":
                    new_date = new_date + datetime.timedelta(days=2)
                if new_date.strftime("%A") == "Sunday":
                    new_date = new_date + datetime.timedelta(days=1)
                dates.append(new_date)
            new_prediction = predictions[predictions['ds'].isin(dates)]
            yhat = new_prediction['yhat']
            lower = new_prediction['yhat_lower']
            upper = new_prediction['yhat_upper']
    
            pd.set_option('display.float_format', lambda x: '%.2f' % x)
            model_predictions=pd.DataFrame(zip(dates, yhat, lower, upper), columns=["Date","Prediction", "Lower", "Upper"])
            st.dataframe(model_predictions, hide_index=True)

        if model_choice == "Markov Regime-Switching":
            st.subheader("Markov Regime-Switching model")
            data4 = data1.copy(deep=True)
            data4['Returns'] = np.log(data4['Close'] / data4['Close'].shift())
            data4.dropna(inplace=True)
            data4_train = data4.iloc[:int(data4.shape[0]*0.8)]
            data4_test = data4.iloc[int(data4.shape[0]*0.8):]

            train = data4_train['Returns']
            test = data4_test["Returns"]
            model = sm.tsa.MarkovRegression(train[1:], k_regimes=2, exog=train[:-1])
            res = model.fit(search_reps=50)

            fig, axes = plt.subplots(2, figsize=(12,4))
            axes[0].plot(data4_train["Date"][1:], res.smoothed_marginal_probabilities[0])
            axes[0].set(title="Probability of being in high regime of stock returns")
            axes[1].plot(data4_train["Date"][1:], res.smoothed_marginal_probabilities[1])
            axes[1].set(title="Probability of being in low regime of stock returns")
            fig.tight_layout()
            st.pyplot(use_container_width=True)

            predictions = []

            params = res.params
            coefs = np.array([params[[2,4]], params[[3,5]]])
            mat = np.array([[params[0], 1-params[0]], [params[1], 1-params[1]]])
            mat_base = mat
            probs = np.array(res.smoothed_marginal_probabilities.iloc[-1])
            current_regime = np.argmax(probs)

            for i in range(len(data4_test)):
                new_probs = mat[current_regime]
                values = np.dot(coefs, np.array([1, train[len(train)-1]]))
                weighted_val = np.dot(values, new_probs)
                mat = np.matmul(mat, mat_base)
                predictions = np.append(predictions, weighted_val)
                train = np.append(train, weighted_val)
                
            predicted_price = []
            current_price = data4_train["Close"].iloc[-1]
            for i in range(len(data4_test)):
                current_price = np.exp(predictions[i]) * current_price
                predicted_price = np.append(predicted_price, current_price)

            rmse = root_mean_squared_error(data4_test["Close"], predicted_price)
            mape = mean_absolute_percentage_error(data4_test["Close"], predicted_price)

            st.subheader(r"$\textsf{\footnotesize Performance}$")

            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                plt.figure(figsize=(10,5))
                plt.plot(data4_train["Date"], data4_train["Close"], label="Training")
                plt.plot(data4_test["Date"], data4_test["Close"], color = 'blue', label="Test")
                plt.plot(data4_test["Date"], predicted_price, color = 'orange', label="Prediction")
                plt.xlabel('Date')
                plt.ylabel('Close Price')
                plt.legend()
                st.pyplot(use_container_width=True)
            with col2:
                st.metric(":orange[Root Mean Squared Error]", f"{rmse:.2f}")
                st.metric(":orange[Mean Absolute Percentage Error]", f'{mape*100:.2f}%')

@st.cache_data
def get_data(ticker, start, end):
    data = yf.download(ticker, start = start, end = end)
    data.insert(0, "Date", data.index, True)
    data.reset_index(drop=True, inplace=True)
    return data

def calculate_moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

def calculate_price_difference(stock_data):
    latest_price = stock_data.iloc[-1]["Close"]
    previous_year_price = stock_data.iloc[-252]["Close"] if len(stock_data) >= 252 else stock_data.iloc[0]["Close"]
    price_difference = latest_price - previous_year_price
    percentage_difference = (price_difference / previous_year_price) * 100
    return price_difference, percentage_difference

def cal_proportion(data, column, order):
    total = len(data[column])
    occurrences = data[column].value_counts()
    result = {}
    for value in order:
        if value in occurrences:
            result[value] = occurrences[value] / total
        else:
            result[value] = 0.0
    result_df = pd.DataFrame([result], columns=order)
    return result_df

def make_pred_df(mc, states, init_state, steps):
    trans_mat = mc.to_matrix()
    res = []
    max_state = []
    for i in range(steps):
        prob = np.matmul(init_state, trans_mat)
        res.append(prob)
        max_state.append(states[np.argmax(prob)])
        init_state = prob
    columns = []
    for x in states:
        columns.append(f"{x}_prob")
    res_df = pd.DataFrame(res, columns=columns)
    res_df["Prediction"] = max_state
    return res_df

def exp_visits_num(n, mc):
    p = mc.to_matrix()
    result = np.zeros_like(p)  
    for i in range(1, n + 1):
        result += np.linalg.matrix_power(p, i)
    return result

def split_range(a, b, num_chunks, len_chunks):
    chunk_size = []
    for len in len_chunks:
        chunk_size.append((b-a)*len)
    result = []
    current = a
    for i in range(num_chunks):
        result.append((current, current + chunk_size[i]))
        current += chunk_size[i]
    return result

def get_state(value, ranges):
    for index in range(len(ranges)):
        if ranges[index][0] <= value < ranges[index][1]:
            return str(index)
    if value > 0:
        return str(len(ranges)-1)
    if value < 0:
        return str(0)

def center_value(value, range):
    upper = value + value * range[1]
    lower = value + value * range[0]
    return (upper+lower)/2

def predict_price(cur_state, cur_price, state_list, state_ranges, data):
    #Fit a Markov chain with data
    chain2 = mc.MarkovChain.fit_sequence("mle", state_list, data)
    #Get initial state
    init = np.zeros((len(state_list),))
    init[int(cur_state)] = 1
    #Multiply matrices to get prediction
    probs = np.matmul(init, chain2.p)
    max_prob = np.max(probs)
    max_states = []
    for i in range(len(probs)):
        if probs[i] == max_prob:
            max_states.append(i)
    if len(max_states) == 1:
        state = max_states[0]
    else:
        #If there are multiple states with same probability, choose a random one
        state = random.choice(max_states)
    #Calculate the price according to that state
    price = center_value(cur_price, state_ranges[state])
    return price

def test_stationarity(timeseries):
    #Determining rolling statistics
    rolmean = timeseries.rolling(10).mean() # around 4 weeks on each month
    rolstd = timeseries.rolling(10).std()
    
    #Plot rolling statistics:
    plt.figure(figsize=(16,9))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    st.pyplot(use_container_width=True)
    
    #Perform Dickey-Fuller test:
    st.write('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    
    st.dataframe(dfoutput, column_config=None)

    if dfoutput['p-value'] < 0.05:
        st.markdown('**Result: Close price is stationary**')
    else: 
        st.markdown('**Result: Close price is not stationary**')

if __name__ == '__main__':
    main()

    
