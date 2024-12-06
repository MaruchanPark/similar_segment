from flask import Flask, render_template_string, request
import matplotlib.pyplot as plt
import mplfinance as mpf
import FinanceDataReader as fdr

import os
import random

import io
import base64
import json
import copy

import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

symbol_list = ["000540", "240810"]

def plot_segment(seg, symbol=None):
    mc = mpf.make_marketcolors(up='red', down='blue', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc)
    
    segment = seg['segment']
    dates = seg['date']
    df = pd.DataFrame(segment, columns=['Open', 'High', 'Low', 'Close'])
    df.index = pd.to_datetime(dates)
    
    fig, axlist = mpf.plot(df, type='candle', style=s, returnfig=True)
    
    date_str = str(seg['date'][0]).split(' ')[0]
    title = f"{symbol}_{date_str}" if symbol else "Test"
    axlist[0].set_title(title, fontsize=16, loc='center')
    axlist[0].set_ylabel("")
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=50)
    buf.seek(0)
    plt.close(fig)
    
    return buf


def load_similar(symbol, date, seq_len):
    load_path = f"/data2/konanbot/GPT_train/preprocess/ipynb/q_test/sim_seg/{symbol}_{date}_{seq_len}.jsonl"
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
    segment = {'segment':data[symbol].values[:seq_len], 'date':[pd.Timestamp(date) for date in data[symbol].index[:seq_len]]}
    
    if l1_dist:
        symbol = f"{symbol}_{l1_dist:.4f}"

    buf = plot_segment(segment, symbol)
    
    img_data = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    
    return f"data:image/png;base64,{img_data}"


def plot_similar(symbol, date, seq_len, idx):
    data = load_similar(symbol, date, seq_len)
    plot_dict = copy.deepcopy(data[idx])
    start_ts = pd.Timestamp(plot_dict['date'][0])
    return plot_source(plot_dict['symbol'], start_ts.year, start_ts.month, start_ts.day, seq_len + 10, l1_dist=plot_dict['l1_dist'])


@app.route("/", methods=["GET", "POST"])
def home():
    # symbol = "000540"
    # files = os.listdir("/data2/konanbot/GPT_train/preprocess/ipynb/q_test/sim_seg/")
    # symbol = random.choice(files).split('_')[0]
    # symbol = "240810"
    
    selected_symbol = symbol_list[0]
    if request.method == "POST":
        selected_symbol = request.form.get("symbol")
    
    plot1 = plot_source(selected_symbol, 2024, 11, 29, 5)
    plot2 = plot_similar(selected_symbol, "2024-11-29", 5, 0)
    plot3 = plot_similar(selected_symbol, "2024-11-29", 5, 1)
    
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Plots</title>
        <style>
            body {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
            }
            .container {
                display: flex;
                position: relative;
            }
            .plot {
                position: absolute;
            }
            .plot-left {
                left: -400px;
                top: 100%;
                transform: translateY(-50%);
            }
            .plot-top-right {
                right: -400px;
                top: 50%;
                transform: translateY(-100%);
            }
            .plot-bottom-right {
                right: -400px;
                top: 50%;
                transform: translateY(-0%);
            }
            .form-container {
                margin-bottom: 20px;
                position: absolute;
                top: 10%;
                left: 50%;
                transform: translateX(-50%);
                z-index:10;
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
                <button type="submit">Update</button>
            </form>
        </div>
        <div class="container">
            <img src="{{ plot1 }}" alt="Plot 1" class="plot plot-left">
            <img src="{{ plot2 }}" alt="Plot 2" class="plot plot-top-right">
            <img src="{{ plot3 }}" alt="Plot 3" class="plot plot-bottom-right">
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, 
                                  symbol_list=["000540", "240810"], 
                                  selected_symbol=selected_symbol, 
                                  plot1=plot1, 
                                  plot2=plot2, 
                                  plot3=plot3)

    # return render_template_string("""
    # <!DOCTYPE html>
    # <html>
    # <body>
    #     <form method="POST">
    #         <select name="symbol">
    #             <option value="000540">000540</option>
    #             <option value="240810">240810</option>
    #         </select>
    #         <button type="submit">Submit</button>
    #     </form>
    # </body>
    # </html>
    # """)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)