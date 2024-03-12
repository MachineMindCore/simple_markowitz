import simple_markowitz.simple_markowitz as mark

symbols="AAPL C CVS CVX HD MSFT ^GSPC ^TNX"

stock=mark.download_data(symbols,start_date="2019-01-01",final_date="2022-01-01")
perf=mark.performance(stock)
cov=mark.cov_matrix(perf)
var=mark.corr_matrix(perf)

pack=(stock,perf)
mark.write_data('data.xlsx','stock',[1,2],pack)

pack=(cov,var)
mark.write_data('data2.xlsx','measure',[1,2],pack)

