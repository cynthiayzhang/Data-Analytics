import numpy as np
import math
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression

def sigmoid(x):
    y = 0.5 - 1/(1+math.exp(-(x-5)))
    return y*2

def main():
    data = pd.read_csv('Revenue2000_2021.csv')  # load data set
    X = data.iloc[:, 0].values.reshape(-1, 1)   # whole data
    Y = data.iloc[:, 1].values.reshape(-1, 1)   
        
    linear_regressor = LinearRegression()  
    linear_regressor.fit(X, Y)                  # linear regression of all data
    Y_pred = linear_regressor.predict(X)        # make predictions

    fig, ax1 = plt.subplots()                   # for plot
    fig.suptitle('Real vs Predicted', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.xaxis.set_label_coords(1.05, -0.05)
    ax1.set_ylabel('Revenue in billion dollars')
    ax1.set_xlim([1999, 2023])
    ax1.set_xticks(np.arange(1999, 2023, 1))
    ax1.set_xticklabels(range(1999, 2023), rotation=45)
    
    ax1.scatter(X, Y, label='Real data', color='blue') # blue dots as real values
    ax1.plot(X, Y_pred, label='Predict data', color='blue', linestyle='dashed')
    
    Xp = []         # output X
    Yp = []         # output Y
    err_up = 0      # error upper bound
    err_down = 0    # error lower bound
    start = 0       # start point of local linear segment
    end = 2 - len(X)

    while end <= 0:
        if end < 0:
            X1 = data.iloc[start:end, 0].values.reshape(-1, 1)  # for local linear segment
            Y1 = data.iloc[start:end, 1].values.reshape(-1, 1)  
            X9 = data.iloc[:end, 0].values.reshape(-1, 1)       # for global linear regression
            Y9 = data.iloc[:end, 1].values.reshape(-1, 1)  
        else:
            X1 = data.iloc[start:, 0].values.reshape(-1, 1)  
            Y1 = data.iloc[start:, 1].values.reshape(-1, 1)  
            X9 = X
            Y9 = Y

        if X1[-1] > 2002:   # collect error range since 2003
            delta = Y1[-1] - Yp[-1]
            if err_up < delta:
                err_up = delta
            if err_down > delta:
                err_down = delta

        i = int(X1[-1]) + 1                 # year to predict
        linear_regressor.fit(X9, Y9)        # global linear regression
        coef9 = linear_regressor.coef_[[0]]  
        val9 = linear_regressor.predict(i)  

        if Y1[-1] > (Y1[-2]*0.98):          # in local linear segment
            linear_regressor.fit(X1, Y1)    # local linear regression
            coef1 = linear_regressor.coef_

            val = linear_regressor.predict(i) # local linear result

            if coef9 > 3:   # when model is stable with sufficient points
                val = (val+val9)/2          # averaging global and local output

            k = end - start + 18            # year from local minima
            if k > 4:   # after year 5, adjust value with sigmoid 
                diff = coef1 * sigmoid(k) 
                val = Y1[-1]+diff           # adjusted predicted value
        else:   # at turning points or going downwards
            val = (Y1[-1] + Y1[-2]) / 2
            if coef9 > 3:
                val = (Y1[-1] + Y1[-2] + val9) / 3

            if i > 2014:
                err_down = 0

            start = len(X) + end - 1    # reset start point of local linear segment
        
        ax1.scatter(i, val, s=120, marker='*', color='red') # red stars as predicted values

        if i > 2014:    # plot with predicted error range 
            y1 = int(val + err_up)
            y2 = int(val + err_down)
            ax1.plot([i, i], [y1, y2], color='red', linewidth=3.0, linestyle='solid', alpha=0.5) 

            print(i)
            print(val)
            print(val + err_down)
            print(val + err_up)

        Xp.append(i)
        Yp.append(val)
        end += 1

    Xp = np.array(Xp).reshape(-1, 1)
    Yp = np.array(Yp).reshape(-1, 1)

    linear_regressor.fit(Xp, Yp)            # linear regression with predicted points
    Y_predp = linear_regressor.predict(Xp) 

    ax1.plot(Xp, Y_predp, label='Predict data', color='tab:red', linestyle='dashed')

    print('+++++++++++')
    print(Xp[-1])
    print(Yp[-1])   # the next year predicted value

    fig.savefig('RevenuePred2020.png') 

    err = 0
    count = 0
    for i in range(0, 18):
        e1 = abs(Yp[i] - Y[i+2])
        e1 /= Y[i+2]
        err += e1
        count += 1

        #print(X[i+2])

        #print(e1)

    print('xxxxxxx')
    print(count)
    print(err)

if __name__ == "__main__":
    main()