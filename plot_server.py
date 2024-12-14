from flask import Flask, render_template_string, request
import matplotlib.pyplot as plt
import mplfinance as mpf
import FinanceDataReader as fdr

import os
import io
import base64
import json
import copy

import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

symbol_list = ["301300", "085310", "024800", "019540", "094840", "053060", 
           "154040", "088910", "210980", "005860", "092460", 
           "036220", "007860", "016710", "000540", "001230", 
           "092790", "306200"]

KOSPI_SYMBOL = "KS11"  # 코스피 지수의 심볼

def plot_segment(seg, ref_idx=None, symbol=None, kospi_seg=None):
    mc = mpf.make_marketcolors(up='red', down='blue', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc)

    segment = seg['segment']
    dates = seg['date']

    df = pd.DataFrame(segment, columns=['Open', 'High', 'Low', 'Close'])
    df.index = pd.to_datetime(dates)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1]})
    fig.subplots_adjust(wspace=0.4)

    mpf.plot(df, type='candle', ax=axes[0], style=s)
    date_str = str(seg['date'][0]).split(' ')[0]
    title = f"{symbol}_{date_str}" if symbol else "Test"
    axes[0].set_title(title, fontsize=16, loc='center')

    if ref_idx is not None:
        ref_close = df.iloc[ref_idx]['Close']
        subsequent_high = df.iloc[ref_idx+1:]['High'].max()
        subsequent_low = df.iloc[ref_idx+1:]['Low'].min()

        percentage_increase = ((subsequent_high - ref_close) / ref_close) * 100
        percentage_decrease = ((subsequent_low - ref_close) / ref_close) * 100
        
        axes[0].hlines(y=ref_close, xmin=0, xmax=len(df), colors='green', linestyles='dashed', label='Ref Close')
        axes[0].hlines(y=subsequent_high, xmin=0, xmax=len(df), colors='red', linestyles='dashed', label='Max High After Ref')
        axes[0].hlines(y=subsequent_low, xmin=0, xmax=len(df), colors='blue', linestyles='dashed', label='Min Low After Ref')

        axes[0].text(len(df) * 1.08, subsequent_high - (subsequent_high * 0.002), f"+{percentage_increase:.2f}%",
                     color='red', fontsize=12, ha='center', va='top')
        axes[0].text(len(df) * 1.08, subsequent_low + (subsequent_low * 0.002), f"{percentage_decrease:.2f}%",
                     color='blue', fontsize=12, ha='center', va='bottom')

    if kospi_seg:
        kospi_df = pd.DataFrame(kospi_seg['segment'], columns=['Open', 'High', 'Low', 'Close'])
        kospi_df.index = pd.to_datetime(kospi_seg['date'])

        mpf.plot(kospi_df, type='candle', ax=axes[1], style=s)
        axes[1].set_title(f"KOSPI_{date_str}", fontsize=14, loc='center')

        if ref_idx is not None:
            kospi_ref_close = kospi_df.iloc[ref_idx]['Close']
            kospi_subsequent_high = kospi_df.iloc[ref_idx+1:]['High'].max()
            kospi_subsequent_low = kospi_df.iloc[ref_idx+1:]['Low'].min()

            kospi_percentage_increase = ((kospi_subsequent_high - kospi_ref_close) / kospi_ref_close) * 100
            kospi_percentage_decrease = ((kospi_subsequent_low - kospi_ref_close) / kospi_ref_close) * 100

            axes[1].hlines(y=kospi_ref_close, xmin=0, xmax=len(kospi_df), colors='green', linestyles='dashed', label='Ref Close')
            axes[1].hlines(y=kospi_subsequent_high, xmin=0, xmax=len(kospi_df), colors='red', linestyles='dashed', label='Max High After Ref')
            axes[1].hlines(y=kospi_subsequent_low, xmin=0, xmax=len(kospi_df), colors='blue', linestyles='dashed', label='Min Low After Ref')

            axes[1].text(len(kospi_df) * 1.08, kospi_subsequent_high - (kospi_subsequent_high * 0.002), f"+{kospi_percentage_increase:.2f}%",
                        color='red', fontsize=12, ha='center', va='top')
            axes[1].text(len(kospi_df) * 1.08, kospi_subsequent_low + (kospi_subsequent_low * 0.002), f"{kospi_percentage_decrease:.2f}%",
                        color='blue', fontsize=12, ha='center', va='bottom')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=50)
    buf.seek(0)
    plt.close(fig)

    return buf

def load_similar(symbol, date, seq_len):
    load_path = f"/data2/konanbot/GPT_train/preprocess/ipynb/q_test/sim_seg/{date}/{symbol}_{date}_{seq_len}.jsonl"
    data = []
    with open(load_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_price(start, end, symbol=None):
    data = {}
    if symbol:
        stock_data = fdr.DataReader(symbol, start=start, end=end)[['Open', 'High', 'Low', 'Close']]
        data[symbol] = stock_data
    return data

def plot_source(symbol, year, month, day, seq_len, l1_dist=None):
    start_date = datetime(year, month, day)
    end_date = start_date + timedelta(days=seq_len * 3)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    data = get_price(start_date, end_date, symbol)
    kospi_data = get_price(start_date, end_date, KOSPI_SYMBOL)

    segment = {'segment': data[symbol].values[:seq_len], 'date': [pd.Timestamp(date) for date in data[symbol].index[:seq_len]]}
    kospi_seg = {'segment': kospi_data[KOSPI_SYMBOL].values[:seq_len], 'date': [pd.Timestamp(date) for date in kospi_data[KOSPI_SYMBOL].index[:seq_len]]}

    if l1_dist:
        symbol = f"{symbol}_{l1_dist:.4f}"
        buf = plot_segment(segment, ref_idx=4, symbol=symbol, kospi_seg=kospi_seg)
    else:
        buf = plot_segment(segment, symbol=symbol, kospi_seg=kospi_seg)

    img_data = base64.b64encode(buf.getvalue()).decode()
    buf.close()

    return f"data:image/png;base64,{img_data}"

def plot_similar(symbol, date, seq_len, idx):
    data = load_similar(symbol, date, seq_len)
    plot_dict = copy.deepcopy(data[idx])
    start_ts = pd.Timestamp(plot_dict['date'][0])
    return plot_source(plot_dict['symbol'], start_ts.year, start_ts.month, start_ts.day, seq_len + 10, l1_dist=plot_dict['l1_dist'])

def get_date_list():
    dataset_path = "/data2/konanbot/GPT_train/preprocess/ipynb/q_test/sim_seg"
    try:
        return sorted([name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))])
    except FileNotFoundError:
        return []

@app.route("/", methods=["GET", "POST"])
def home():
    date_list = get_date_list()
    print(date_list)
    selected_date = date_list[0] if date_list else ""
    selected_symbol = symbol_list[0]

    if request.method == "POST":
        selected_symbol = request.form.get("symbol")
        selected_date = request.form.get("date")

    print(selected_date)
    y = int(selected_date.split("-")[0])
    m = int(selected_date.split("-")[1])
    d = int(selected_date.split("-")[2])

    plot1 = plot_source(selected_symbol, y, m, d, 5)
    plot2 = plot_similar(selected_symbol, selected_date, 5, 0)
    plot3 = plot_similar(selected_symbol, selected_date, 5, 1)
    plot4 = plot_similar(selected_symbol, selected_date, 5, 2)
    plot5 = plot_similar(selected_symbol, selected_date, 5, 3)
    plot6 = plot_similar(selected_symbol, selected_date, 5, 4)
    plot7 = plot_similar(selected_symbol, selected_date, 5, 5)
    plot8 = plot_similar(selected_symbol, selected_date, 5, 6)
    plot9 = plot_similar(selected_symbol, selected_date, 5, 7)

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Plots Comparison</title>
        <style>
            body {
                display: flex;
                flex-direction: column;
                align-items: center;
                margin: 0;
                padding: 0;
                overflow-y: auto;
            }

            .form-container {
                margin: 20px 0;
                text-align: center;
            }

            .main-container {
                display: flex;
                flex-direction: row;
                justify-content: center;
                gap: 20px;
                max-width: 1600px;
                margin: 0 auto;
            }

            .left-container {
                flex: 1;
                display: flex;
                justify-content: center;
                align-items: center;
            }

            .left-container .plot {
                max-width: 700px;
                height: auto;
            }

            .right-container {
                flex: 2;
                display: grid;
                grid-template-columns: repeat(1, 1fr);
                gap: 20px;
            }

            .right-container .plot {
                width: 100%;
                height: auto;
            }
        </style>
    </head>
    <body>
        <div class="form-container">
            <form method="POST">
                <label for="symbol">Select Symbol:</label>
                <select name="symbol" id="symbol">
                    {% for symbol in symbol_list %}
                    <option value="{{ symbol }}" {% if symbol == selected_symbol %}selected{% endif %}>
                        {{ symbol }}
                    </option>
                    {% endfor %}
                </select>

                <label for="date">Select Date:</label>
                <select name="date" id="date">
                    {% for date in date_list %}
                    <option value="{{ date }}" {% if date == selected_date %}selected{% endif %}>
                        {{ date }}
                    </option>
                    {% endfor %}
                </select>

                <button type="submit">Update</button>
            </form>
        </div>

        <div class="main-container">
            <div class="left-container">
                <img src="{{ plot1 }}" alt="Main Plot" class="plot">
            </div>

            <div class="right-container">
                <img src="{{ plot2 }}" alt="Plot 2" class="plot">
                <img src="{{ plot3 }}" alt="Plot 3" class="plot">
                <img src="{{ plot4 }}" alt="Plot 4" class="plot">
                <img src="{{ plot5 }}" alt="Plot 5" class="plot">
                <img src="{{ plot6 }}" alt="Plot 6" class="plot">
                <img src="{{ plot7 }}" alt="Plot 7" class="plot">
                <img src="{{ plot8 }}" alt="Plot 8" class="plot">
                <img src="{{ plot9 }}" alt="Plot 9" class="plot">
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, 
                                  symbol_list=symbol_list, 
                                  date_list=date_list,
                                  selected_symbol=selected_symbol, 
                                  selected_date=selected_date,
                                  plot1=plot1, 
                                  plot2=plot2, 
                                  plot3=plot3,
                                  plot4=plot4,
                                  plot5=plot5,
                                  plot6=plot6,
                                  plot7=plot7,
                                  plot8=plot8,
                                  plot9=plot9)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
