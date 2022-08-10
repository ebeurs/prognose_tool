import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('final_list.csv')
df = df.drop(df.columns[[0]], axis=1)
df.columns = ['final_price']
quantile25 = df.quantile(0.25).iloc[0]
quantile75 = df.quantile(0.75).iloc[0]
mean = df.mean().iloc[0]
maximum = df.max().iloc[0]
minimum = df.min().iloc[0]
std = df.std().iloc[0]
print(quantile75)
plt.figure(figsize=(10,5))
plt.hist(df['final_price'], bins=50, color='teal');
plt.axvline(x=mean, color='red', ls='--')
plt.axvline(x=quantile25, color='purple', ls=':')
plt.axvline(x=quantile75, color='lime', ls=':')
plt.axvline(x=minimum, color='blue', ls='-.')
plt.axvline(x=maximum, color='navy', ls='-.')
montecarlo_title = str('Prognose simulation: ') + str(len(df))
plt.title(montecarlo_title, size=16, color='blue', pad=20)
plt.xlabel('Profit after four years', color='blue')
plt.ylabel('Estimation amount', color='blue')
mean_text = str('Mean ') + str(int(mean))
quantile25_text = str('25th Percentile ') + str(int(quantile25))
quantile75_text = str('75th Percentile ') + str(int(quantile75))
min_text = str('Miniumum value ') + str(int(minimum))
max_text = str('Miniumum value ') + str(int(maximum))
plt.legend([mean_text,quantile25_text,quantile75_text,min_text,max_text])
savefig_text = str('Prognose_simulation_') + str(int(len(df))) +str('.jpg')
plt.savefig(savefig_text, dpi=200)
plt.tight_layout()
plt.show()