import os
import time
import json
import copy
from multiprocessing import Pool
from scipy.spatial.distance import cityblock
import FinanceDataReader as fdr

def get_price(start, end, symbol=None):
    stock_list = fdr.StockListing('KRX')
    symbols = stock_list['Code'].tolist()
    
    data = {}
    start_t = time.time()
    if symbol:
        stock_data = fdr.DataReader(symbol, start=start, end=end)[['Open', 'High', 'Low', 'Close', 'Volume']]
        data[symbol] = stock_data
        return data
        
    for i, symbol in enumerate(symbols):
        
        if (i+1) % 500 == 0:
            print(f"Collecting... {i+1}/{len(symbols)}, Elapsed:{time.time()-start_t:.2f}")
            start_t = time.time()
            
        stock_data = fdr.DataReader(symbol, start=start, end=end)[['Open', 'High', 'Low', 'Close', 'Volume']]
        data[symbol] = stock_data
    return data

def normalization(data):
    for symbol in data:
        max_close = data[symbol]['Close'].max()
        data[symbol]['Close'] = data[symbol]['Close'] / max_close
        data[symbol]['Open'] = data[symbol]['Open'] / max_close
        data[symbol]['High'] = data[symbol]['High'] / max_close
        data[symbol]['Low'] = data[symbol]['Low'] / max_close
    return data

def l1_dist(seg1, seg2):
    open_seg1, high_seg1, low_seg1, close_seg1, _ = seg1['segment'].T
    open_seg2, high_seg2, low_seg2, close_seg2, _ = seg2['segment'].T
    
    open_distance = cityblock(open_seg1, open_seg2)
    high_distance = cityblock(high_seg1, high_seg2)
    low_distance = cityblock(low_seg1, low_seg2)
    close_distance = cityblock(close_seg1, close_seg2)
    
    avg_distance = (open_distance + high_distance + low_distance + close_distance) / 4
    return avg_distance

def segmentation(normalized_data, seq_len=10):
    
    segments = {symbol: [] for symbol in normalized_data}
    start_t = time.time()
    for i, symbol in enumerate(normalized_data):
        if (i+1) % 500 == 0:
            print(f"{i+1}/{len(normalized_data)}, Elapsed:{time.time()-start_t:.2f}")
            start_t = time.time()
        norm_prices = normalized_data[symbol]
        
        for k in range(len(norm_prices) - seq_len + 1):
            dates = norm_prices.index[k:k+seq_len]
            segment = norm_prices.iloc[k:k+seq_len].values
            date = norm_prices.iloc[k:k+seq_len].index
            segments[symbol].append({"segment": segment, "date": date})
            
    return segments

def search(sym1, idx, sym2, threshold):
    result = []
    seg1 = segments[sym1][idx]
    dist_list = []
    
    for i, seg2 in enumerate(global_segments[sym2]):
        avg_distance = l1_dist(seg1, seg2)
        dist_list.append((i, float(avg_distance)))

    if dist_list:
        sorted_dist = sorted(dist_list, key=lambda x: x[1])
        for i in range(3):
            idx = sorted_dist[i][0]
            seg2 = global_segments[sym2][idx]
            seg2['segment'] = seg2['segment'].tolist()
            seg2['date'] = [str(date).split(' ')[0] for date in seg2['date']]
            seg2['symbol'] = sym2
            seg2['l1_dist'] = sorted_dist[i][1]
            result.append(seg2)
            
    return result

def search_parallel(tasks):
    sym1, idx, sym2 = tasks
    result = []
    seg1 = global_segments[sym1][idx]
    dist_list = []
    for i, seg2 in enumerate(global_segments[sym2]):
        avg_distance = l1_dist(seg1, seg2)
        dist_list.append((i, float(avg_distance)))
            
    if len(dist_list) > 3:
        sorted_dist = sorted(dist_list, key=lambda x: x[1])
        for i in range(3):
            idx = sorted_dist[i][0]
            seg2 = global_segments[sym2][idx]
            seg2['segment'] = seg2['segment'].tolist()
            seg2['date'] = [str(date).split(' ')[0] for date in seg2['date']]
            seg2['symbol'] = sym2
            seg2['l1_dist'] = sorted_dist[i][1]
            result.append(seg2)
            
    return result

def initialize_segments(segments):
    global global_segments
    global_segments = segments


start = '2000-01-01'
end = '2024-12-13'
seq_len = 5
idx = -1
target_symbol = ["301300", "085310", "024800", "019540", "094840", "053060", 
           "154040", "088910", "210980", "005860", "092460", 
           "036220", "007860", "016710", "000540", "001230", 
           "092790", "306200"]

data = get_price(start, end)
normalized_data = normalization(data)
segments = segmentation(normalized_data, seq_len=seq_len)

for step in range(30):
    seg_slice = {}
    start_t = time.time()
    
    for i, symbol in enumerate(segments):
        
        if (i+1) % 100 == 0:
            print(f"Copy segments... {i+1}/{len(segments.keys())}, Elapsed:{time.time()-start_t:.2f}")
            start_t = time.time()
        seg_slice[symbol] = copy.deepcopy(segments[symbol])[:len(segments[symbol]) - step]

    date = str(seg_slice[symbol][-1]['date']).split(' ')[0]
    
    for i, sym1 in enumerate(target_symbol):
        print(f"{i+1}/{len(target_symbol)}")
        symbols = list(data.keys())
        symbols.remove(sym1)
        tasks = [(sym1, idx, sym2) for sym2 in symbols]
        
        num_workers=20
        with Pool(processes=num_workers, initializer=initialize_segments, initargs=(seg_slice,)) as pool:
            results = pool.map(search_parallel, tasks)
        
        flattened = [result for result_list in results for result in result_list if result]
        flattened = sorted(flattened, key=lambda x: x['l1_dist'])[:10]
    
        ## 저장
        if flattened:
            date = str(seg_slice[sym1][idx]['date'][0]).split(' ')[0]
            os.makedirs(f"./sim_seg/{date}", exist_ok=True)
            with open(f"./sim_seg/{date}/{sym1}_{date}_{seq_len}.jsonl", "w") as f:
                for result in flattened:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')