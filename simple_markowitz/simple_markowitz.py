import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize

def read_data(file):
    stocks=pd.read_excel(file,index_col=0)
    return stocks

def download_data(symbols,start_date,final_date,frec="1mo"): #Example: "SPCE MSFT F TSLA PFE ^GSPC"
    stocks=yf.download(symbols, start=start_date, end=final_date, interval=frec)
    stocks=stocks["Adj Close"]
    stocks=stocks.dropna()
    return stocks

def log_data(stocks):
    return stocks.apply(np.log)

def performance(stocks): #Calculate perform of stocks
    print('Calculating returns')
    symbols_list=list(stocks.columns.values)
    symbols_len=len(symbols_list)
    stocks_perf=pd.DataFrame()

    for i in range(symbols_len):
        s=stocks[[symbols_list[i]]]
        s_plus=stocks[[symbols_list[i]]].shift(periods=1)
        r=(s-s_plus)/s_plus
        stocks_perf.insert(i,symbols_list[i]+'_perf',r)
    return stocks_perf

def performance_log(stocks): #Calculate perform of stocks
    print('Calculating returns')
    symbols_list=list(stocks.columns.values)
    symbols_len=len(symbols_list)
    stocks_perf=pd.DataFrame()

    for i in range(symbols_len):
        s=stocks[[symbols_list[i]]]
        s_plus=stocks[[symbols_list[i]]].shift(periods=1)
        r=(s-s_plus)/s
        stocks_perf.insert(i,symbols_list[i]+'_perf',r)
    return stocks_perf

def log_performance(stocks): #Calculate log perform of stocks
    print('Calculating returns')
    symbols_list=list(stocks.columns.values)
    symbols_len=len(symbols_list)
    stocks_perf=pd.DataFrame()

    for i in range(symbols_len):
        s=stocks[[symbols_list[i]]]
        s_plus=stocks[[symbols_list[i]]].shift(periods=1)
        r=np.log(s/s_plus)
        stocks_perf.insert(i,symbols_list[i]+'_perf',r)
    return stocks_perf

def measure(data):
    print('Calculating measurements')
    data_mean=data.mean(skipna=True)
    data_std=data.std(skipna=True)
    data_var=data.var(skipna=True)
    m=pd.DataFrame({'mean':data_mean,'std':data_std,'var':data_var})
    return m

def cov_matrix(data):
    return data.cov()

def var_matrix(data):
    return data.var()

def corr_matrix(data):
    return data.corr()




def weights(N): 
    #N->Assets number
    def matti_virkkunen(N): #Matti distribution
        data=np.zeros(N)
        v=np.random.rand(N-1)
        v=np.append(v,np.array([1,0]))
        v.sort()
        v=np.diff(v)
        data=v
        return pd.DataFrame(data)

    def truncated(N): #Truncated distribution
        data=np.zeros(N)
        index=np.arange(data.size)
        np.random.shuffle(index)
        for j in index:
            data[j]=np.random.rand()
            if np.sum(data)>1 or (j==index[-1] and np.sum(data)<1):
                data[j]=0
                data[j]=1-np.sum(data)
        return pd.DataFrame(data)

    selector=np.random.rand()
    if selector < 0.7:
        return matti_virkkunen(N)
    else:
        return truncated(N)

def random_portfolio(perf,analysis_matrix,cov_matrix,k):
    print('Calculating random portfolio')
    tag=list(perf.columns.values)
    mean_serie=analysis_matrix['mean']
    N=len(tag)
    random_set=pd.DataFrame(index=range(k),columns=tag+['risk','return'],data=np.zeros((k,2+N)))
    
    for c in random_set.index:
        w_set=weights(N)
        random_set.iloc[c,0:N]=w_set.transpose()
        for i in range(N):
            random_set['return'][c]+=w_set.iloc[i]*mean_serie.iloc[i]
            for j in range(N):
                random_set['risk'][c]+=cov_matrix.iloc[i,j]*w_set.iloc[i]*w_set.iloc[j]
    print(random_set)    
    return random_set

def optimal_portfolio(analysis,cov):
    print('Calculating optimal portfolio')  
    mean_R=analysis['mean']
    N=mean_R.shape[0]

    class obj:
        def r_p(w,*arg): #Return
            cum_R=0
            mean_R=arg[0][0]
            N=mean_R.shape[0]
            for i in range(N):
                cum_R+=w[i]*mean_R.iloc[i]
            return -1*cum_R
    
        def sigma_p(w,*arg): #Risk
            cum_sigma=0
            cov=arg[0][1]
            N=cov.shape[0]
            for i in range(N):
                for j in range(N):
                    cum_sigma+=w[i]*w[j]*cov.iloc[i,j]
            return cum_sigma
        
        def rest_sum(w):
            return w.sum()-1
        
        def rest_sigma(w, *arg):
            cum_sigma = 0
            cov = arg[0][1]
            N = cov.shape[0]
            for i in range(N):
                for j in range(N):
                    cum_sigma += w[i] * w[j] * cov.iloc[i, j]
            portfolio_sigma = np.sqrt(cum_sigma)
            return portfolio_sigma - np.sqrt(arg[0][2])
    
    #Resolve edge portfolios
    lim=tuple((0,1) for _ in range(N))
    rest1={'type':'eq','fun':obj.rest_sum}
    w0=tuple(weights(N).to_numpy().transpose()[0])
    edge_arg=[mean_R,cov,0]
    max_r=optimize.minimize(obj.r_p,x0=w0,bounds=lim,constraints=rest1,args=edge_arg)
    min_sigma=optimize.minimize(obj.sigma_p,x0=w0,bounds=lim,constraints=rest1,args=edge_arg)

    #Dataframe structure
    mid_port=20
    tag=mean_R.index.tolist()+['risk','return']
    index=['min_port','max_port']+['dist_'+str(i) for i in range(mid_port)]
    optimal_set=pd.DataFrame(columns=tag,index=index)

    #Replacing known values   
    optimal_set.iloc[0,0:N]=min_sigma.x
    optimal_set.iloc[1,0:N]=max_r.x
    optimal_set.iloc[0,N]=obj.sigma_p(min_sigma.x,edge_arg)
    optimal_set.iloc[0,N+1]=-1*obj.r_p(min_sigma.x,edge_arg)
    optimal_set.iloc[1,N]=obj.sigma_p(max_r.x,edge_arg)
    optimal_set.iloc[1,N+1]=-1*obj.r_p(max_r.x,edge_arg)

    #resolve middle portfolios
    min_var=optimal_set.iloc[0,N]
    max_var=optimal_set.iloc[1,N]
    r_obj=np.linspace(min_var,(max_var+min_var)/2,mid_port)
    for i in range(len(r_obj)):
        var=r_obj[i]
        mid_arg=([mean_R,cov,var],)
        rest2={'type':'eq', 'fun':obj.rest_sigma, 'args':mid_arg}
        print(var)
        mid_r = optimize.minimize(obj.r_p, x0=w0, bounds=lim, constraints=[rest1, rest2], args=mid_arg)
        optimal_set.iloc[i+2,0:N]=mid_r.x
        optimal_set.iloc[i+2,N]=obj.sigma_p(mid_r.x,edge_arg)
        optimal_set.iloc[i+2,N+1]=-1*obj.r_p(mid_r.x,edge_arg)

    return optimal_set




def bullet_plot(random_set,optimal_set):
    print('Ploting Markowitz')
    random_points=(random_set.loc[:,'risk':'return']).to_numpy(dtype=np.float64)
    optimal_points=(optimal_set.loc[:,'risk':'return']).to_numpy(dtype=np.float64)
    print(type(optimal_points[:,0]))

    d=optimal_set.shape[0]-2
    optimal_poly=np.polyfit(optimal_points[:,0],optimal_points[:,1],d)
    optimal_domain=np.linspace(optimal_points[0,0],optimal_points[-1,0],100)
    optimal_frontier=np.polyval(optimal_poly,optimal_domain)

    fig = plt.figure()
    plt.scatter(random_set['risk'],random_set['return'],s=5,alpha=0.7)
    plt.scatter(optimal_points[:,0],optimal_points[:,1],s=5,c='k')
    plt.plot(optimal_domain,optimal_frontier,'--k')
    plt.show()
    return

def write_data(file,sheet_name,dim,*args):
    frame=args[0]
    frame=[frame] if dim==[1,1] else frame
    print(frame)
    df=0
    point=[0,0]
    memo=0
    writer = pd.ExcelWriter(file, engine='openpyxl')
    workbook = writer.book
    for i in range(dim[1]):
        point[0]=0
        for j in range(dim[0]):
                frame[df].to_excel(writer,sheet_name=sheet_name,startrow=point[0],startcol=point[1])
                point[0]=frame[df].shape[0]+2
                memo=frame[df].shape[1] if memo < frame[df].shape[1] else memo
                if dim[0]*dim[1] > len(frame): break 
                df+=1
        point[1]=memo+2
    writer.close()
    workbook.close()
    return        

def quick_backtest(symbols,k):
    tag=symbols.split()
    stocks=download_data(symbols)
    logs=download_data(symbols)
    perf=performance(stocks)
    perf_analysis=measure(perf)
    perf_cov=cov_matrix(perf)
    perf_corr=corr_matrix(perf)
    random_set=random_portfolio(perf,perf_analysis,perf_cov,k)
    optimal_set=optimal_portfolio(perf_analysis,perf_cov)
    bullet_plot(random_set,optimal_set)

    file='solution.xlsx'
    write_data(file,'stocks',[2,1],(stocks))
    return


#weights, returns, risks = optimal_portfolio(return_vec)

# fig = plt.figure()
# plt.plot(stds, means, 'o')
# plt.ylabel('mean')
# plt.xlabel('std')
# plt.plot(risks, returns, 'y-o')
# py.iplot_mpl(fig, filename='efficient_frontier', strip_style=True)



