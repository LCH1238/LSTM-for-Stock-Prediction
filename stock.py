from pytdx.exhq import *
from pytdx.hq import *
import numpy as np

api_hq = TdxHq_API()
api_hq = api_hq.connect('119.147.212.81', 7709)

data_all = []
label_all = []
days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
mon = 1
day = 1
while mon <= 12:
    while day <= days[mon - 1]:
        date = "2021{:02}{:02}".format(mon, day)
        day_data = api_hq.get_history_minute_time_data(TDXParams.MARKET_SH, "603456", int(date))
        if len(day_data) == 0:
            day += 1
            continue
        data_all.append(day_data[59]['price'])
        data_all.append(day_data[119]['price'])
        data_all.append(day_data[179]['price'])
        data_all.append(day_data[239]['price'])
        
        label_all.append(int(date + "1"))
        label_all.append(int(date + "2"))
        label_all.append(int(date + "3"))
        label_all.append(int(date + "4"))

        day += 1
    mon += 1 
    day = 1
    
mon = 1
day = 1
while mon <= 12:
    while day <= days[mon - 1]:
        date = "2022{:02}{:02}".format(mon, day)
        day_data = api_hq.get_history_minute_time_data(TDXParams.MARKET_SH, "603456", int(date))
        if len(day_data) == 0:
            day += 1
            continue
        data_all.append(day_data[59]['price'])
        data_all.append(day_data[119]['price'])
        data_all.append(day_data[179]['price'])
        data_all.append(day_data[239]['price'])
        
        label_all.append(int(date + "1"))
        label_all.append(int(date + "2"))
        label_all.append(int(date + "3"))
        label_all.append(int(date + "4"))

        day += 1
    mon += 1 
    day = 1

print(len(data_all))
print(len(label_all))

stock = np.array(data_all)
label = np.array(label_all)

train_stock = stock[:-100]
train_label = label[:-100]

test_stock = stock[-100:]
test_label = label[-100:]

np.savetxt("train_stock.txt", train_stock, fmt="%.3f")
np.savetxt("train_label.txt", train_label, fmt="%d")
np.savetxt("test_stock.txt", test_stock, fmt="%.3f")
np.savetxt("test_label.txt", test_label, fmt="%d")

