import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression

def main():
    data0 = pd.read_csv('f2022_datachallenge.csv')  # load data set

    # different column can be loaded as the following way
    price = data0['price']
    accommodates = data0['accommodates']
    bedrooms = data0['bedrooms']
    beds = data0['beds']
    bathrooms_text = data0['bathrooms_text']
    latitude = data0['latitude']
    longitude = data0['longitude']
    first_review = data0['first_review']
    last_review = data0['last_review']

    # statistics about a column can be checked by print out describe()
    # this will print statistics of price
    #print('price statistics - original:')
    #print(price.describe())

    # this will print statistics of accommodates
    #print('accommodates statistics - original:')
    #print(accommodates.describe())

    # clean up original data
    # remove any data with accommodates < 1
    data = data0[data0['accommodates'] > 0]
    # then remove any data with price < 1
    data = data[data['price'] > 0]
    # then remove any data with bedrooms < 1
    data = data[data['bedrooms'] > 0]
    # then remove any data with beds < 1
    data = data[data['beds'] > 0]

    # this will print statistics of price
    #print('price statistics - clean:')
    #print(data['price'].describe())

    # this will print statistics of accommodates
    #print('accommodates statistics - clean:')
    #print(data['accommodates'].describe())

    # after cleaning up we can see the data point drops from 26345 to 22596
    # further clean up or data trimming can be done in such way
    
    # here is the box plot of price vs accommodates
    # first to plot all 16 bins - accommodates: 1 to 16
    fig1, ax1 = plt.subplots() # for plotting
    ax1.set_xlabel('accommodates')
    ax1.xaxis.set_label_position('top')
    ax1.set_ylabel('price')    
    prices1 = []
    labels1 = []
    cap = 3000 # cap price to 3000
    for i in range (1, 17):
        data_i = data[data['accommodates'] == i] # data with accommodates = 1, 2, ... 16 only
        a = []
        for p in data_i['price']:
            if p > cap:
                a.append(cap)
            else:
                a.append(p)

        prices1.append(a)
        labels1.append(i) # for x axis labels

    bplot1 = ax1.boxplot(prices1,
                     vert = True,  # vertical box alignment
                     patch_artist = True,  # fill with color
                     labels = labels1) 

    fig1.savefig('price_vs_accommodates_all.png')

    # now to plot first 8 bins - accommodates: 1 to 8 for better visualization
    fig2, ax2 = plt.subplots() # for plotting
    ax2.set_xlabel('accommodates')
    ax2.xaxis.set_label_position('top')
    ax2.set_ylabel('price')
    prices2 = []
    labels2 = []
    cap = 800 # cap price to 800
    for i in range (1, 9):
        data_i = data[data['accommodates'] == i] # data with accommodates = 1, 2, ... 8 only
        a = []
        for p in data_i['price']:
            if p > cap:
                a.append(cap)
            else:
                a.append(p)

        prices2.append(a)
        labels2.append(i) # for x axis labels

    bplot2 = ax2.boxplot(prices2,
                     vert = True,  # vertical box alignment
                     patch_artist = True,  # fill with color
                     labels = labels2) 

    fig2.savefig('price_vs_accommodates_8bins.png')

    # now let's change price to price/accommodates, we're expecting to see flat pattern
    fig3, ax3 = plt.subplots() # for plotting
    ax3.set_xlabel('accommodates')
    ax3.xaxis.set_label_position('top')
    ax3.set_ylabel('price')

    # first to plot all 16 bins - accommodates: 1 to 16
    prices3 = []
    labels3 = []
    cap = 500 # cap price to 500
    for i in range (1, 17):
        data_i = data[data['accommodates'] == i] # data with accommodates = 1, 2, ... 16 only
        a = []
        for p in data_i['price']:
            if p/i > cap:
                a.append(cap)
            else:
                a.append(p/i)

        prices3.append(a)
        labels3.append(i) # for x axis labels

    bplot3 = ax3.boxplot(prices3,
                     vert = True,  # vertical box alignment
                     patch_artist = True,  # fill with color
                     labels = labels3) 

    fig3.savefig('price_per_accommodates_all.png')

    # now to plot first 8 bins - accommodates: 1 to 8 for better visualization
    fig4, ax4 = plt.subplots() # for plotting
    ax4.set_xlabel('accommodates')
    ax4.xaxis.set_label_position('top')
    ax4.set_ylabel('price')
    prices4 = []
    labels4 = []
    cap = 200 # cap price to 200
    for i in range (1, 9):
        data_i = data[data['accommodates'] == i] # data with accommodates = 1, 2, ... 8 only
        a = []
        for p in data_i['price']:
            if p/i > cap:
                a.append(cap)
            else:
                a.append(p/i)

        prices4.append(a)
        labels4.append(i) # for x axis labels

    bplot4 = ax4.boxplot(prices4,
                     vert = True,  # vertical box alignment
                     patch_artist = True,  # fill with color
                     labels = labels4) 

    fig4.savefig('price_per_accommodates_8bins.png')

    # plot geographic distribution of price/accommodates
    # we're using price/accommodates, data is cleaned data
    data['price'] = data['price'] / data['accommodates']

    # separate data into different bin for different color codes later
    data100 = data[data['price'] <= 100]
 
    data200 = data[data['price'] <= 200]
    data200 = data200[data200['price'] > 100]
   
    data300 = data[data['price'] <= 300]
    data300 = data300[data300['price'] > 200]

    data400 = data[data['price'] <= 400]
    data400 = data400[data400['price'] > 300]

    data500 = data[data['price'] <= 500]
    data500 = data500[data500['price'] > 400]
    
    data600 = data[data['price'] > 500]
    
    fig5, ax5 = plt.subplots(figsize=(20, 20)) # set a bigger fig size to have a more detailed view of geographic data

    ax5.set_xlabel('longitude', fontsize=24)
    ax5.set_ylabel('latitude', fontsize=24)
    # use data['longitude'].describe() and data['latitude'].describe() to get xlim and ylim
    ax5.set_xlim([-160, -154.5])
    ax5.set_xticks(np.arange(-160, -154.5, 0.2))
    ax5.set_ylim([18.8, 22.4])
    ax5.set_yticks(np.arange(18.8, 22.4, 0.2))

    latitude100 = data100['latitude']
    longitude100 = data100['longitude']
    X100 = longitude100.values.reshape(-1, 1)  
    Y100 = latitude100.values.reshape(-1, 1) 
    ax5.scatter(X100, Y100, color='xkcd:lightblue', marker='+')

    latitude200 = data200['latitude']
    longitude200 = data200['longitude']
    X200 = longitude200.values.reshape(-1, 1)  
    Y200 = latitude200.values.reshape(-1, 1) 
    ax5.scatter(X200, Y200, color='tab:blue', marker='+')
    
    latitude300 = data300['latitude']
    longitude300 = data300['longitude']
    X300 = longitude300.values.reshape(-1, 1)  
    Y300 = latitude300.values.reshape(-1, 1) 
    ax5.scatter(X300, Y300, color='tab:purple', marker='+')
    
    latitude400 = data400['latitude']
    longitude400 = data400['longitude']
    X400 = longitude400.values.reshape(-1, 1)  
    Y400 = latitude400.values.reshape(-1, 1) 
    ax5.scatter(X400, Y400, color='tab:pink', marker='+')

    latitude500 = data500['latitude']
    longitude500 = data500['longitude']
    X500 = longitude500.values.reshape(-1, 1)  
    Y500 = latitude500.values.reshape(-1, 1) 
    ax5.scatter(X500, Y500, color='tab:red', marker='+')
    
    latitude600 = data600['latitude']
    longitude600 = data600['longitude']
    X600 = longitude600.values.reshape(-1, 1)  
    Y600 = latitude600.values.reshape(-1, 1) 
    ax5.scatter(X600, Y600, color='xkcd:dark red', marker='+')

    fig5.savefig('locations_price_per_accommodates.png')  
    
    # plot geographic distribution of accommodates
    # we're using price/accommodates, data is cleaned data
    # separate data into different bin for different color codes later
    data100 = data[data['accommodates'] <= 2]
 
    data200 = data[data['accommodates'] <= 4]
    data200 = data200[data200['accommodates'] > 2]
   
    data300 = data[data['accommodates'] <= 6]
    data300 = data300[data300['accommodates'] > 4]

    data400 = data[data['accommodates'] <= 8]
    data400 = data400[data400['accommodates'] > 6]

    data500 = data[data['accommodates'] <= 10]
    data500 = data500[data500['accommodates'] > 8]
    
    data600 = data[data['accommodates'] > 10]
    
    fig6, ax6 = plt.subplots(figsize=(20, 20)) # set a bigger fig size to have a more detailed view of geographic data

    ax6.set_xlabel('longitude', fontsize=24)
    ax6.set_ylabel('latitude', fontsize=24)
    # use data['longitude'].describe() and data['latitude'].describe() to get xlim and ylim
    ax6.set_xlim([-160, -154.5])
    ax6.set_xticks(np.arange(-160, -154.5, 0.2))
    ax6.set_ylim([18.8, 22.4])
    ax6.set_yticks(np.arange(18.8, 22.4, 0.2))

    latitude100 = data100['latitude']
    longitude100 = data100['longitude']
    X100 = longitude100.values.reshape(-1, 1)  
    Y100 = latitude100.values.reshape(-1, 1) 
    ax6.scatter(X100, Y100, color='xkcd:lightblue', marker='+')

    latitude200 = data200['latitude']
    longitude200 = data200['longitude']
    X200 = longitude200.values.reshape(-1, 1)  
    Y200 = latitude200.values.reshape(-1, 1) 
    ax6.scatter(X200, Y200, color='tab:blue', marker='+')
    
    latitude300 = data300['latitude']
    longitude300 = data300['longitude']
    X300 = longitude300.values.reshape(-1, 1)  
    Y300 = latitude300.values.reshape(-1, 1) 
    ax6.scatter(X300, Y300, color='tab:purple', marker='+')
    
    latitude400 = data400['latitude']
    longitude400 = data400['longitude']
    X400 = longitude400.values.reshape(-1, 1)  
    Y400 = latitude400.values.reshape(-1, 1) 
    ax6.scatter(X400, Y400, color='tab:pink', marker='+')

    latitude500 = data500['latitude']
    longitude500 = data500['longitude']
    X500 = longitude500.values.reshape(-1, 1)  
    Y500 = latitude500.values.reshape(-1, 1) 
    ax6.scatter(X500, Y500, color='tab:red', marker='+')
    
    latitude600 = data600['latitude']
    longitude600 = data600['longitude']
    X600 = longitude600.values.reshape(-1, 1)  
    Y600 = latitude600.values.reshape(-1, 1) 
    ax6.scatter(X600, Y600, color='xkcd:dark red', marker='+')

    fig6.savefig('locations_accommodates.png') 
    
    # analyze when accommodates == 6 the number of baths impact on price    
    fig7, ax7 = plt.subplots() # for plotting
    ax7.set_xlabel('baths')
    ax7.xaxis.set_label_position('top')
    ax7.set_ylabel('price')
    
    labels6 = [1, 2, 3, 4]
    prices6 = []
    p1 = []
    p2 = []
    p3 = []
    p4 = []

    cap = 1000 # cap price to 1000

    data6 = data[data['accommodates'] == 6]
    i = 0
    while i < len(data6['bathrooms_text']):
        br = data6['bathrooms_text'].values[i]  # this is how to get record value from dataframe
        if type(br) is float: #if the record is a nan, which means it's a float type, not a string, we have to skip this record
            i = i+1
            continue

        pp = data6['price'].values[i]
        if pp > cap:
            pp = cap

        if "2" in br:
            p2.append(pp)
        elif "3" in br:
            p3.append(pp)
        elif "4" in br:
            p4.append(pp)
        else:
            p1.append(pp)

        i = i+1

    prices6.append(p1)
    prices6.append(p2)
    prices6.append(p3)
    prices6.append(p4)

    bplot7 = ax7.boxplot(prices6,
                     vert = True,  # vertical box alignment
                     patch_artist = True,  # fill with color
                     labels = labels6) 

    fig7.savefig('price_6_accommodates_baths.png')
    

    X = []
    Y = []
    
    cap = 1000 # cap price to 1000

    data8 = data[data['accommodates'] < 13] # trend upwards to 12 is relatively linear
    data8 = data8[data8['price'] <= 2000]   # filter out too expensive outliers
    i = 0
    while i < len(data8['price']):
        ac = data8['accommodates'].values[i]  # this is how to get record value from dataframe
        br = data8['bathrooms_text'].values[i] 
        pp = data8['price'].values[i]

        if type(br) is float: # if the record is a nan, which means it's a float type, not a string, we have to skip this record
            i = i+1
            continue

        if "2" in br:
            X.append([ac, 2]) # x_1 = accomodates, x_2 = # of baths, y = price
            Y.append(pp)
        elif "3" in br:
            X.append([ac, 3])
            Y.append(pp)
        elif "4" in br:
            X.append([ac, 4])
            Y.append(pp)
        elif "6" in br:
            X.append([ac, 6])
            Y.append(pp)
        elif "5.5" in br:
            X.append([ac, 5])
            Y.append(pp)
        elif "5" in br:
            if ".5" not in br:
                X.append([ac, 5])
                Y.append(pp)
        else:
            X.append([ac, 1])
            Y.append(pp)

        i = i+1

    linear_regressor = LinearRegression()  
    linear_regressor.fit(X, Y)                  # linear regression of all data
    Y_pred = linear_regressor.predict(X)        # make predictions
    print(linear_regressor.coef_)
    print(linear_regressor.intercept_)


    fig8, ax8 = plt.subplots() # for plotting
    ax8.set_xlabel('accommodates')
    ax8.xaxis.set_label_position('top')
    ax8.set_ylabel('price')    
    prices8 = []
    labels8 = []
        
    for i in range (1, 13): # creates 12 empty list in a list
        a = []
        prices8.append(a)
        labels8.append(i) # for x axis labels

    i = 0
    while i < len(Y_pred): # fills in the list
        ac = X[i][0]
        prices8[ac-1].append(Y_pred[i])

        i = i+1

    bplot8 = ax8.boxplot(prices8,
                     vert = True,  # vertical box alignment
                     patch_artist = True,  # fill with color
                     labels = labels8) 

    fig8.savefig('price_vs_accommodates_predict.png')

    fig9, ax9 = plt.subplots() # for plotting
    ax9.set_xlabel('accommodates')
    ax9.xaxis.set_label_position('top')
    ax9.set_ylabel('price')    
    prices9 = []
    labels9 = []
        
    for i in range (1, 13): 
        a = []
        prices9.append(a)
        labels9.append(i) # for x axis labels

    i = 0
    while i < len(Y): 
        ac = X[i][0]
        prices9[ac-1].append(Y[i])

        i = i+1

    bplot9 = ax9.boxplot(prices9,
                     vert = True,  # vertical box alignment
                     patch_artist = True,  # fill with color
                     labels = labels9) 

    fig9.savefig('price_vs_accommodates_orignal.png')

    X = []
    Y = []
    
    cap = 1000 # cap price to 1000

    data18 = data[data['accommodates'] < 13]
    data18 = data18[data18['price'] <= 2000]
    i = 0
    while i < len(data18['price']):
        ac = data18['accommodates'].values[i]  # this is how to get record value from dataframe
        br = data18['bathrooms_text'].values[i] 
        bm = data18['bedrooms'].values[i] 
        bd = data18['beds'].values[i] 
        pp = data18['price'].values[i]

        if type(br) is float:
            i = i+1
            continue

        if "2" in br:
            X.append([ac, 2, bm, bd])
            Y.append(pp)
        elif "3" in br:
            X.append([ac, 3, bm, bd])
            Y.append(pp)
        elif "4" in br:
            X.append([ac, 4, bm, bd])
            Y.append(pp)
        elif "6" in br:
            X.append([ac, 6, bm, bd])
            Y.append(pp)
        elif "5.5" in br:
            X.append([ac, 5, bm, bd])
            Y.append(pp)
        elif "5" in br:
            if ".5" not in br:
                X.append([ac, 5, bm, bd])
                Y.append(pp)
        else:
            X.append([ac, 1, bm, bd])
            Y.append(pp)

        i = i+1

    linear_regressor2 = LinearRegression()  
    linear_regressor2.fit(X, Y)                  # linear regression of all data
    Y_pred2 = linear_regressor2.predict(X)        # make predictions
    print(linear_regressor2.coef_)
    print(linear_regressor2.intercept_)


    fig18, ax18 = plt.subplots() # for plotting
    ax18.set_xlabel('accommodates')
    ax18.xaxis.set_label_position('top')
    ax18.set_ylabel('price')    
    prices18 = []
    labels18 = []
        
    for i in range (1, 13):
        a = []
        prices18.append(a)
        labels18.append(i) # for x axis labels

    i = 0
    while i < len(Y_pred2):
        ac = X[i][0]
        prices18[ac-1].append(Y_pred2[i])

        i = i+1

    bplot18 = ax18.boxplot(prices18,
                     vert = True,  # vertical box alignment
                     patch_artist = True,  # fill with color
                     labels = labels18) 

    fig18.savefig('price_vs_accommodates_predict2.png')

if __name__ == "__main__":
    main()
