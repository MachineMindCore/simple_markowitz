import simple_markowitz.simple_markowitz as mark

data='data/datos_p2.xlsx'
simple_data=mark.read_data(data)
simple_data=simple_data.sort_index()
log_perf=mark.log_performance(simple_data)
log_analysis=mark.measure(log_perf)
cov_m=mark.cov_matrix(log_perf)
corr_m=mark.corr_matrix(log_perf)
random_set=mark.random_portfolio(log_perf,log_analysis,cov_m,1000)
optimal_set=mark.optimal_portfolio(log_analysis,cov_m)
mark.bullet_plot(random_set,optimal_set)

#Write data
file='data/stocks.xlsx'
mark.write_data(file,'stocks',[1,1],(simple_data))
file='data/performance.xlsx'
mark.write_data(file,'performance',[1,1],(log_perf))
file='data/analysis.xlsx'
mark.write_data(file,'analysis',[1,1],(log_analysis))
file='data/matrix.xlsx'
mark.write_data(file,'matrix',[2,1],(cov_m,corr_m))
file='data/portfolios.xlsx'
mark.write_data(file,'portfolios',[1,2],(random_set,optimal_set))

