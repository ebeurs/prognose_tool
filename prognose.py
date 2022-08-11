import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t
import sklearn
plt.close("all")




# short analysis of bitcoin
df_btc = pd.read_csv(r'./btc-usd-max.csv')
print(df_btc['price'].iloc[-1440:])
df_btc_1440 = df_btc['price'].iloc[-1440:].copy()

monthly_btc = []
for i in range((48)):
    for j in range(30):
        try:
            if j != 29:
                pass
            else:
                monthly_btc.append(df_btc_1440.iloc[j * (i + 1)])
        except:
            pass

daily_log_btc = np.diff(np.log(df_btc_1440))
monthly_log_btc = np.diff(np.log(monthly_btc))

monthly_mean_btc = np.mean(monthly_log_btc)
daily_mean_btc = np.mean(daily_log_btc)

daily_std_btc = np.std(daily_log_btc)
montly_std_btc = np.std(monthly_log_btc)



# create_csv = False

def calculate_write_csv(startvaluefund,monthlyaverageinflow,perfomancefee,year1input,year2input,year3input,year4input,stddeviateinput,hwmstartvalue,realmonths):

    start_value_fund = startvaluefund
    monthly_average_inflow = monthlyaverageinflow
    performance_fee = perfomancefee
    year1_input = year1input
    year2_input = year2input
    year3_input = year3input
    year4_input = year4input
    std_input = stddeviateinput
    hwm_start_value = hwmstartvalue
    realisation_months = realmonths

    # input
    # start_value_fund = 40000000
    # monthly_average_inflow = 500000
    # performance_fee = 0.2
    # year1_input = 0.20
    # year2_input = 0.06
    # year3_input = -0.12
    # year4_input = 0.18
    # std_input = 4
    # hwm_start_value = 45000000

    input_string_list = ['year1 %','year2 %','year3 %','year4 %','std dev','hwm lead','real months']

    yearly_input_list = [year1_input,year2_input,year3_input,year4_input]
    percentage_list = []
    inflow_list = []
    zero_list = []
    temp_value = start_value_fund
    perf_fee_list = []

    for i in range(49):
        zero_list.append(0)
        percentage_list.append(0)
        inflow_list.append(0)
        perf_fee_list.append(performance_fee)

    for j in range(4):
        found = 0
        input_year = yearly_input_list[j]
        while found == 0:
            temp_value = 1
            for i in range(12):

                if i == 0:
                    print(i)

                temp_mean = yearly_input_list[j]/12
                random_percentage = (np.random.normal(temp_mean, daily_std_btc * std_input, 1))[0]
                if j == 0 :
                    if j == 0 and i == 0:
                        percentage_list[0] = random_percentage
                        inflow_list[0] = monthly_average_inflow
                    else:
                        percentage_list[i] = random_percentage
                        inflow_list[i] = monthly_average_inflow
                elif j == 1:
                    percentage_list[12 + i] = random_percentage
                    inflow_list[12 + i] = monthly_average_inflow
                elif j == 2:
                    percentage_list[24 + i] = random_percentage
                    inflow_list[24 + i] = monthly_average_inflow
                elif j == 3:
                    percentage_list[36 + i] = random_percentage
                    inflow_list[36 + i] = monthly_average_inflow

                temp_value = temp_value * (1+random_percentage)
                print(1 + random_percentage)
                print(temp_value)
                if i == 11 and ((input_year - 0.001 ) <= (temp_value - 1) <= (input_year + 0.001)):
                    found = 1



    data = {'Percentage monthly': percentage_list[:],
            'Inflow': inflow_list[:],
            'Performance fee input': perf_fee_list[:],
            'Performance fee': zero_list[:],
            'Cum Performance fee': zero_list[:],
            'Fund value': zero_list[:],
            'Lead serie': zero_list[:],
            }

    df_prognose = pd.DataFrame(data)

    for i in range(48):
        if i < len(input_string_list):
            df_prognose[str('Serie') + str(i + 1) + str(', ') + str(input_string_list[i])] = zero_list
        else:
            df_prognose[str('Serie') + str(i+1)] = zero_list

    start_column = [0,0,0,0,0,start_value_fund,start_value_fund,yearly_input_list[0],yearly_input_list[1],yearly_input_list[2],yearly_input_list[3],std_input,hwm_start_value]

    for i in range(42):
        start_column.append(0)

    df_prognose = df_prognose.T
    df_prognose.insert(loc=0, column='Start values', value=start_column)


    # that standard dataframe is finished here



    for i in range(49):
        zero_list.append(0)


    individual_hwm_list = zero_list.copy()
    individual_per_fee_list = zero_list.copy()
    new_serie_list = zero_list.copy()

    individual_hwm_list[0] = hwm_start_value

    # add the inflow to the highwatermark value and to the fund value
    for i in range(48):

        if i == 0 :
            temp_df = df_prognose.iloc[7:13,i].copy()
            df_prognose.iloc[7:13,i] = 0
        if i == 1:
            df_prognose.iloc[7:13, 0] = temp_df

        for j in range(52):
            if j < 48:

                # the inflow
                if j == i:
                    df_prognose.iloc[i + 7, j + 1] = df_prognose.iloc[1, j + 1]
                else:
                    pass

            if df_prognose.iloc[0, i+1] > 0:

                if j < 48:
                    df_prognose.iloc[j + 6, i + 1] = df_prognose.iloc[j + 6, i + 1] + df_prognose.iloc[j + 6, i] * (1+df_prognose.iloc[0, i + 1])

                if j == 48:
                    # initiate new lists. The individual high water mark. individual performance fee
                    fund_value = 0
                    if i == 0:
                        # individual_hwm_list[0] = hwm_start_value
                        for k in range(i+2):
                            if k <= 48:
                                if individual_hwm_list[k] < df_prognose.iloc[k + 6, i + 1]:
                                    if k == (i+1):
                                        individual_hwm_list[k] = df_prognose.iloc[k + 6, i + 1]
                                        fund_value += df_prognose.iloc[k + 6, i + 1]
                                        df_prognose.iloc[5, i + 1] = fund_value
                                        df_prognose.iloc[4, i + 1] = df_prognose.iloc[3, i + 1]
                                    else:
                                        input_value = individual_hwm_list[k]
                                        output_value = df_prognose.iloc[k + 6,i + 1]
                                        profit = (output_value - input_value)
                                        performance_fee_fund = profit * df_prognose.iloc[2, i + 1]
                                        individual_per_fee_list[k] = performance_fee_fund
                                        individual_hwm_list[k] = df_prognose.iloc[k + 6, i + 1]
                                        df_prognose.iloc[k + 6, i + 1] = individual_hwm_list[k] - individual_per_fee_list[k]
                                        individual_hwm_list[k] = individual_hwm_list[k] - individual_per_fee_list[k]
                                        df_prognose.iloc[3, i + 1] += individual_per_fee_list[k]

                                        if df_prognose.iloc[k + 6,i + 1] < 0:
                                            pass
                                        else:
                                            fund_value += df_prognose.iloc[k + 6,i + 1]

                                        # print('nice')
                                else:
                                    if df_prognose.iloc[k + 6, i + 1] < 0:
                                        pass
                                    else:
                                        fund_value += df_prognose.iloc[k + 6, i + 1]
                    else:
                        for k in range(i+2):
                            if k <= 48:
                                if individual_hwm_list[k] < df_prognose.iloc[k + 6, i + 1]:
                                    if k == (i+1) :
                                        individual_hwm_list[k] = df_prognose.iloc[k + 6, i + 1]
                                        fund_value += df_prognose.iloc[k + 6, i + 1]
                                        df_prognose.iloc[5, i + 1] = fund_value
                                        df_prognose.iloc[4, i + 1] = df_prognose.iloc[3, i + 1] + df_prognose.iloc[4, i]
                                    else:
                                        # individual_hwm_list[k] = df_prognose.iloc[k + 6, i + 1]
                                        # input_value = df_prognose.iloc[k + 6,i]
                                        # output_value = df_prognose.iloc[k + 6,i + 1]
                                        # profit = (output_value - input_value)
                                        # performance_fee_fund = profit * df_prognose.iloc[2, i]
                                        input_value = individual_hwm_list[k]
                                        output_value = df_prognose.iloc[k + 6,i + 1]
                                        profit = (output_value - input_value)
                                        performance_fee_fund = profit * df_prognose.iloc[2, i + 1]
                                        individual_per_fee_list[k] = performance_fee_fund
                                        individual_hwm_list[k] = df_prognose.iloc[k + 6, i + 1]
                                        df_prognose.iloc[k + 6, i + 1] = individual_hwm_list[k] - individual_per_fee_list[k]
                                        individual_hwm_list[k] = individual_hwm_list[k] - individual_per_fee_list[k]
                                        df_prognose.iloc[3, i + 1] += individual_per_fee_list[k]
                                        if df_prognose.iloc[k + 6,i + 1] < 0:
                                            pass
                                        else:
                                            fund_value += df_prognose.iloc[k + 6,i + 1]
                                else:
                                    if df_prognose.iloc[k + 6, i + 1] < 0:
                                        pass
                                    else:
                                        fund_value += df_prognose.iloc[k + 6, i + 1]
            else:
                if j < 48:
                    df_prognose.iloc[j + 6, i + 1] = df_prognose.iloc[j + 6, i + 1] + df_prognose.iloc[j + 6, i] * (
                                1 + df_prognose.iloc[0, i + 1])

                if j == 48:
                    # initiate new lists. The individual high water mark. individual performance fee
                    fund_value = 0
                    if i == 0:
                        # individual_hwm_list[0] = start_value_fund
                        for k in range(i + 2):
                            if k <= 48:
                                if individual_hwm_list[k] < df_prognose.iloc[k + 6, i + 1]:
                                    if k == (i + 1):
                                        individual_hwm_list[k] = df_prognose.iloc[k + 6, i + 1]
                                        fund_value += df_prognose.iloc[k + 6, i + 1]
                                        df_prognose.iloc[5, i + 1] = fund_value
                                        df_prognose.iloc[4, i + 1] = df_prognose.iloc[3, i + 1]
                                    else:
                                        # individual_hwm_list[k] = df_prognose.iloc[k + 6, i + 1]
                                        # input_value = df_prognose.iloc[k + 6, i]
                                        # output_value = df_prognose.iloc[k + 6, i + 1]
                                        # profit = (output_value - input_value)
                                        # performance_fee_fund = profit * df_prognose.iloc[2, i]

                                        input_value = individual_hwm_list[k]
                                        output_value = df_prognose.iloc[k + 6,i + 1]
                                        profit = (output_value - input_value)
                                        individual_hwm_list[k] = df_prognose.iloc[k + 6, i + 1]
                                        performance_fee_fund = profit * df_prognose.iloc[2, i + 1]

                                        individual_per_fee_list[k] = 0
                                        df_prognose.iloc[3, i + 1] += individual_per_fee_list[k]
                                        if df_prognose.iloc[k + 6,i + 1] < 0:
                                            pass
                                        else:
                                            fund_value += df_prognose.iloc[k + 6,i + 1]

                                        # print('nice')
                                else:
                                    if k == (i + 1):
                                        individual_hwm_list[k] = df_prognose.iloc[k + 6, i + 1]
                                        fund_value += df_prognose.iloc[k + 6, i + 1]
                                        df_prognose.iloc[5, i + 1] = fund_value
                                        df_prognose.iloc[4, i + 1] = df_prognose.iloc[3, i + 1] + df_prognose.iloc[4, i]
                                    else:

                                        if df_prognose.iloc[k + 6,i + 1] < 0:
                                            pass
                                        else:
                                            fund_value += df_prognose.iloc[k + 6,i + 1]

                                        # print('nice')

                    else:
                        for k in range(i + 2):
                            if k <= 48:
                                if individual_hwm_list[k] < df_prognose.iloc[k + 6, i + 1]:
                                    if k == (i + 1):
                                        individual_hwm_list[k] = df_prognose.iloc[k + 6, i + 1]
                                        fund_value += df_prognose.iloc[k + 6, i + 1]
                                        df_prognose.iloc[5, i + 1] = fund_value
                                        df_prognose.iloc[4, i + 1] = df_prognose.iloc[3, i + 1] + df_prognose.iloc[4, i]
                                    else:
                                        input_value = individual_hwm_list[k]
                                        output_value = df_prognose.iloc[k + 6,i + 1]
                                        profit = (output_value - input_value)
                                        performance_fee_fund = profit * df_prognose.iloc[2, i + 1]
                                        individual_per_fee_list[k] = performance_fee_fund
                                        individual_hwm_list[k] = df_prognose.iloc[k + 6, i + 1]


                                        df_prognose.iloc[k + 6, i + 1] = individual_hwm_list[k] - individual_per_fee_list[k]
                                        individual_hwm_list[k] = individual_hwm_list[k] - individual_per_fee_list[k]
                                        df_prognose.iloc[3, i + 1] += individual_per_fee_list[k]
                                        if df_prognose.iloc[k + 6,i + 1] < 0:
                                            pass
                                        else:
                                            fund_value += df_prognose.iloc[k + 6,i + 1]
                                        # print('nice')
                                else:
                                    if k == (i + 1):
                                        individual_hwm_list[k] = df_prognose.iloc[k + 6, i + 1]
                                        fund_value += df_prognose.iloc[k + 6, i + 1]
                                        df_prognose.iloc[5, i + 1] = fund_value
                                        df_prognose.iloc[4, i + 1] = df_prognose.iloc[3, i + 1] + df_prognose.iloc[4, i]
                                    else:
                                        if df_prognose.iloc[k + 6,i + 1] < 0:
                                            pass
                                        else:
                                            fund_value += df_prognose.iloc[k + 6,i + 1]
                                        # print('nice')
    df_prognose.drop(df_prognose.columns[len(df_prognose.columns)-1], axis=1, inplace=True)


    df_prognose.to_csv('prognose.csv')
    df_prognose_change = df_prognose.copy()
    df_prognose_change.to_csv('prognose_change.csv')


def calculate_read_csv_calculate(amount_running: object) -> object:

    # this is a the part that uses the prognose_change csv as input
    # retrieve the data from the excel

    df_prognose = pd.read_csv('prognose_change.csv', index_col = 0)

    year1_input = df_prognose.iloc[8,0]
    year2_input = df_prognose.iloc[9,0]
    year3_input = df_prognose.iloc[10,0]
    year4_input = df_prognose.iloc[11,0]
    std_input = df_prognose.iloc[12,0]
    hwm_start_value = df_prognose.iloc[13,0]
    realisation_months = df_prognose.iloc[14,0]


    yearly_input_list = [year1_input,year2_input,year3_input,year4_input]




    zero_list =[]
    percentage_list = []
    perf_fee_list = []
    for i in range(49):
        zero_list.append(0)
        percentage_list.append(0)


    final_amount_list = []
    for l in range(amount_running):
        print(l)
        year1_input = df_prognose.iloc[8, 0]
        year2_input = df_prognose.iloc[9, 0]
        year3_input = df_prognose.iloc[10, 0]
        year4_input = df_prognose.iloc[11, 0]
        std_input = df_prognose.iloc[12, 0]
        hwm_start_value = df_prognose.iloc[13, 0]
        realisation_months = df_prognose.iloc[14, 0]

        yearly_input_list = [year1_input, year2_input, year3_input, year4_input]

        df_prognose.iloc[4:, 1:] = 0

        zero_list = []
        percentage_list = []
        perf_fee_list = []
        for i in range(49):
            zero_list.append(0)
            percentage_list.append(0)

        # estimate the new return, thinking about the realisation
        for j in range(4):
            found = 0
            input_year = yearly_input_list[j]
            while found == 0:

                temp_value = 1
                temp_value_realisation = 1
                temp_value_with_realisation = 1
                if j == 0:
                    new_end_year1 = yearly_input_list[0]

                for i in range(12):

                    temp_mean = yearly_input_list[j] / 12
                    random_percentage = (np.random.normal(temp_mean, daily_std_btc * std_input, 1))[0]

                    if j == 0:
                        if i < int(realisation_months):
                            percentage_list[i] = df_prognose.iloc[0, i + 1]
                            temp_value_with_realisation = temp_value_with_realisation * (1 + df_prognose.iloc[0, i + 1])
                            if i == ((realisation_months) - 1):
                                temp_value_realisation = temp_value_with_realisation
                                new_end_year1 = (((12 - realisation_months) / 12) * year1_input + temp_value_realisation) - 1
                                # print(new_end_year1)
                        else:
                            temp_value_with_realisation = temp_value_with_realisation * (1 + random_percentage)
                            percentage_list[i] = random_percentage
                    elif j == 1:
                        percentage_list[12 + i] = random_percentage
                        temp_value = temp_value * (1 + random_percentage)
                    elif j == 2:
                        percentage_list[24 + i] = random_percentage
                        temp_value = temp_value * (1 + random_percentage)
                    elif j == 3:
                        percentage_list[36 + i] = random_percentage
                        temp_value = temp_value * (1 + random_percentage)

                    if j == 0 and i == 11:
                        if (((new_end_year1) - 0.001) <= (temp_value_with_realisation - 1) <= ((new_end_year1) + 0.001)):
                            found = 1
                        else:
                            temp_value_with_realisation = 0
                    elif j != 0 and i == 11:
                        if i == 11 and ((input_year - 0.001) <= (temp_value - 1) <= (input_year + 0.001)):
                            found = 1

        new_values = pd.DataFrame(percentage_list[:48]).T
        df_prognose.iloc[0, 1:49] = new_values

        temp_list = []
        for i in range(len(df_prognose)):
            temp_list.append(0)
        df_prognose['last'] = temp_list
        individual_hwm_list = zero_list.copy()
        individual_per_fee_list = zero_list.copy()
        new_serie_list = zero_list.copy()

        individual_hwm_list[0] = hwm_start_value

        # add the inflow to the highwatermark value and to the fund value
        for i in range(48):

            if i == 0:
                temp_df = df_prognose.iloc[8:15, i].copy()
                df_prognose.iloc[8:15, i] = 0
            if i == 1:
                df_prognose.iloc[8:15, 0] = temp_df

            for j in range(52):
                if j < 48:

                    # the inflow
                    if j == i:
                        df_prognose.iloc[i + 8, j + 1] = df_prognose.iloc[1, j + 1]
                    else:
                        pass

                if df_prognose.iloc[0, i + 1] > 0:

                    index_on = 0
                    # calculate the next step
                    if j < 48:
                        if df_prognose.iloc[j + 7, i] > 0:
                            df_prognose.iloc[j + 7, i + 1] = df_prognose.iloc[j + 7, i] * (
                                        1 + (df_prognose.iloc[0, i + 1]))
                        else:
                            # df_prognose.iloc[j + 6, i + 1] = df_prognose.iloc[j + 6, i]
                            df_prognose.iloc[j + 7, i + 2] = 0
                            if df_prognose.iloc[2, i + 1] < 0:
                                if j == 47:
                                    index_on = 1

                    if j == 48:
                        # initiate new lists. The individual high water mark. individual performance fee
                        fund_value = 0
                        print(i + 1, individual_hwm_list)
                        if i == 0:
                            # individual_hwm_list[0] = hwm_start_value
                            for k in range(i + 2):
                                if k <= 48:
                                    if individual_hwm_list[k] < df_prognose.iloc[k + 7, i + 1]:
                                        if k == (i + 1):
                                            individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                            fund_value += df_prognose.iloc[k + 7, i + 1] + df_prognose.iloc[2, i + 1]
                                            df_prognose.iloc[6, i + 1] = fund_value

                                            # outflow in the lead serie for the hwm
                                            # individual_hwm_list[0] += df_prognose.iloc[2, i + 1] * (
                                            #             individual_hwm_list[0] / df_prognose.iloc[7, i + 1])
                                            # df_prognose.iloc[7, i + 1] = df_prognose.iloc[7, i + 1] + df_prognose.iloc[2, i + 1]

                                            #   new
                                            # There are K series calculated, the outflow is substracted from the series by the money amount of the series / total amount of all the series,
                                            df_hwm = pd.DataFrame(individual_hwm_list[:])
                                            sum_hwm = df_hwm.sum()
                                            partial_df_hwm = df_hwm / sum_hwm
                                            # The money amount that is substracted from the hwm and the outflow is then multiplied by the (hwm / next value)
                                            new_df_hwm = df_hwm + (partial_df_hwm * df_prognose.iloc[2, i + 1])
                                            df_prognose.iloc[7:, i + 1] = df_prognose.iloc[7:, i + 1].values + (
                                                        partial_df_hwm * df_prognose.iloc[2, i + 1]).squeeze()
                                            df_hwm = df_hwm + (partial_df_hwm * df_prognose.iloc[2, i + 1])
                                            individual_hwm_list = list(df_hwm.iloc[:, 0])

                                            df_prognose.iloc[5, i + 1] = df_prognose.iloc[5, i] + df_prognose.iloc[
                                                4, i + 1]
                                        else:
                                            input_value = individual_hwm_list[k]
                                            output_value = df_prognose.iloc[k + 7, i + 1]
                                            profit = (output_value - input_value)
                                            if input_value >= 0:
                                                performance_fee_fund = profit * df_prognose.iloc[3, i + 1]
                                            individual_per_fee_list[k] = performance_fee_fund

                                            if input_value >= 0:
                                                df_prognose.iloc[k + 7, i + 1] = df_prognose.iloc[
                                                                                     k + 7, i + 1] - performance_fee_fund
                                            else:
                                                df_prognose.iloc[k + 7, i + 1] = df_prognose.iloc[k + 7, i]

                                            individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                            # individual_hwm_list[k] = individual_hwm_list[k] - individual_per_fee_list[k]
                                            df_prognose.iloc[4, i + 1] += individual_per_fee_list[k]

                                            if df_prognose.iloc[k + 7, i + 1] < 0:
                                                pass
                                            else:
                                                fund_value += df_prognose.iloc[k + 7, i + 1]

                                    else:
                                        if k == (i + 1):
                                            # individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                            fund_value += df_prognose.iloc[k + 7, i + 1]
                                            df_prognose.iloc[6, i + 1] = fund_value + df_prognose.iloc[2, i + 1]
                                            # individual_hwm_list[0] += df_prognose.iloc[2, i + 1] * (
                                            #             individual_hwm_list[0] / df_prognose.iloc[7, i + 1])
                                            # df_prognose.iloc[7, i + 1] = df_prognose.iloc[7, i + 1] + df_prognose.iloc[2, i + 1]

                                            #   new
                                            # There are K series calculated, the outflow is substracted from the series by the money amount of the series / total amount of all the series,
                                            df_hwm = pd.DataFrame(individual_hwm_list[:])
                                            sum_hwm = df_hwm.sum()
                                            partial_df_hwm = df_hwm / sum_hwm
                                            # The money amount that is substracted from the hwm and the outflow is then multiplied by the (hwm / next value)
                                            new_df_hwm = df_hwm + (partial_df_hwm * df_prognose.iloc[2, i + 1])
                                            df_prognose.iloc[7:, i + 1] = df_prognose.iloc[7:, i + 1].values + (
                                                        partial_df_hwm * df_prognose.iloc[2, i + 1]).squeeze()
                                            df_hwm = df_hwm + (partial_df_hwm * df_prognose.iloc[2, i + 1])
                                            individual_hwm_list = list(df_hwm.iloc[:, 0])

                                            df_prognose.iloc[5, i + 1] = df_prognose.iloc[5, i]

                                        else:
                                            fund_value += df_prognose.iloc[k + 7, i + 1]

                        # The columns after the first one
                        else:
                            for k in range(i + 2):
                                if k <= 48:
                                    if individual_hwm_list[k] < df_prognose.iloc[k + 7, i + 1]:
                                        if k == (i + 1):
                                            individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                            fund_value += df_prognose.iloc[k + 7, i + 1]
                                            df_prognose.iloc[6, i + 1] = fund_value + df_prognose.iloc[2, i + 1]

                                            # #   old
                                            # individual_hwm_list[0] += df_prognose.iloc[2, i + 1] * (individual_hwm_list[0] / df_prognose.iloc[7, i + 1])
                                            # df_prognose.iloc[7, i + 1] = df_prognose.iloc[7, i + 1] + df_prognose.iloc[2, i + 1]

                                            #   new
                                            # There are K series calculated, the outflow is substracted from the series by the money amount of the series / total amount of all the series,
                                            df_hwm = pd.DataFrame(individual_hwm_list[:])
                                            sum_hwm = df_hwm.sum()
                                            partial_df_hwm = df_hwm / sum_hwm
                                            # The money amount that is substracted from the hwm and the outflow is then multiplied by the (hwm / next value)
                                            new_df_hwm = df_hwm + (partial_df_hwm * df_prognose.iloc[2, i + 1])
                                            df_prognose.iloc[7:, i + 1] = df_prognose.iloc[7:, i + 1].values + (
                                                        partial_df_hwm * df_prognose.iloc[2, i + 1]).squeeze()
                                            df_hwm = df_hwm + (partial_df_hwm * df_prognose.iloc[2, i + 1])
                                            individual_hwm_list = list(df_hwm.iloc[:, 0])

                                            df_prognose.iloc[5, i + 1] = df_prognose.iloc[5, i] + df_prognose.iloc[
                                                4, i + 1]


                                        else:
                                            input_value = individual_hwm_list[k]
                                            output_value = df_prognose.iloc[k + 7, i + 1]
                                            profit = (output_value - input_value)
                                            if input_value >= 0:
                                                performance_fee_fund = profit * df_prognose.iloc[3, i + 1]
                                            individual_per_fee_list[k] = performance_fee_fund
                                            individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                            if input_value >= 0:
                                                df_prognose.iloc[k + 7, i + 1] = individual_hwm_list[k] - \
                                                                                 individual_per_fee_list[k]
                                            else:
                                                df_prognose.iloc[k + 7, i + 1] = df_prognose.iloc[k + 7, i]

                                            individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]

                                            # individual_hwm_list[k] = individual_hwm_list[k] - individual_per_fee_list[k]
                                            df_prognose.iloc[4, i + 1] += individual_per_fee_list[k]

                                            if df_prognose.iloc[k + 7, i + 1] < 0:
                                                pass
                                            else:
                                                fund_value += df_prognose.iloc[k + 7, i + 1]
                                    else:
                                        if k == (i + 1):
                                            # individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                            fund_value += df_prognose.iloc[k + 7, i + 1]
                                            df_prognose.iloc[6, i + 1] = fund_value + df_prognose.iloc[2, i + 1]
                                            # individual_hwm_list[0] += df_prognose.iloc[2, i + 1] * (individual_hwm_list[0] / df_prognose.iloc[7, i + 1])
                                            # df_prognose.iloc[7, i + 1] = df_prognose.iloc[7, i + 1] + df_prognose.iloc[2, i + 1]
                                            #   new
                                            # There are K series calculated, the outflow is substracted from the series by the money amount of the series / total amount of all the series,
                                            df_hwm = pd.DataFrame(individual_hwm_list[:])
                                            sum_hwm = df_hwm.sum()
                                            partial_df_hwm = df_hwm / sum_hwm
                                            # The money amount that is substracted from the hwm and the outflow is then multiplied by the (hwm / next value)
                                            new_df_hwm = df_hwm + (partial_df_hwm * df_prognose.iloc[2, i + 1])
                                            df_prognose.iloc[7:, i + 1] = df_prognose.iloc[7:, i + 1].values + (
                                                    partial_df_hwm * df_prognose.iloc[2, i + 1]).squeeze()
                                            df_hwm = df_hwm + (partial_df_hwm * df_prognose.iloc[2, i + 1])
                                            individual_hwm_list = list(df_hwm.iloc[:, 0])

                                            df_prognose.iloc[5, i + 1] = df_prognose.iloc[5, i]

                                        else:
                                            fund_value += df_prognose.iloc[k + 7, i + 1]



                # if the return is negative
                else:
                    if j < 48:

                        if df_prognose.iloc[j + 7, i] > 0:
                            df_prognose.iloc[j + 7, i + 1] = df_prognose.iloc[j + 7, i] * (
                                        1 + (df_prognose.iloc[0, i + 1]))
                        else:
                            # df_prognose.iloc[j + 6, i + 1] = df_prognose.iloc[j + 6, i]
                            df_prognose.iloc[j + 7, i + 2] = 0
                            if df_prognose.iloc[1, i + 1] < 0:
                                if j == 47:
                                    index_on = 1

                    if j == 48:
                        # initiate new lists. The individual high water mark. individual performance fee
                        fund_value = 0
                        print(i + 1, individual_hwm_list)
                        if i == 0:
                            # individual_hwm_list[0] = hwm_start_value
                            for k in range(i + 2):
                                if k <= 48:
                                    if individual_hwm_list[k] < df_prognose.iloc[k + 7, i + 1]:
                                        if k == (i + 1):
                                            individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                            fund_value += df_prognose.iloc[k + 7, i + 1] + df_prognose.iloc[2, i + 1]
                                            df_prognose.iloc[6, i + 1] = fund_value

                                            # outflow in the lead serie for the hwm
                                            # individual_hwm_list[0] += df_prognose.iloc[2, i + 1]
                                            # individual_hwm_list[0] += df_prognose.iloc[2, i + 1] * (individual_hwm_list[0] / df_prognose.iloc[7, i + 1])
                                            # df_prognose.iloc[7, i + 1] = df_prognose.iloc[7, i + 1] + df_prognose.iloc[2, i + 1]
                                            #   new
                                            #   new
                                            # There are K series calculated, the outflow is substracted from the series by the money amount of the series / total amount of all the series,
                                            df_hwm = pd.DataFrame(individual_hwm_list[:])
                                            sum_hwm = df_hwm.sum()
                                            partial_df_hwm = df_hwm / sum_hwm
                                            # The money amount that is substracted from the hwm and the outflow is then multiplied by the (hwm / next value)
                                            new_df_hwm = df_hwm + (partial_df_hwm * df_prognose.iloc[2, i + 1])
                                            df_prognose.iloc[7:, i + 1] = df_prognose.iloc[7:, i + 1].values + (
                                                        partial_df_hwm * df_prognose.iloc[2, i + 1]).squeeze()
                                            df_hwm = df_hwm + (partial_df_hwm * df_prognose.iloc[2, i + 1])
                                            individual_hwm_list = list(df_hwm.iloc[:, 0])

                                            df_prognose.iloc[5, i + 1] = df_prognose.iloc[5, i] + df_prognose.iloc[
                                                4, i + 1]


                                        else:
                                            input_value = individual_hwm_list[k]
                                            output_value = df_prognose.iloc[k + 7, i + 1]
                                            profit = (output_value - input_value)
                                            if input_value >= 0:
                                                performance_fee_fund = profit * df_prognose.iloc[3, i + 1]
                                            individual_per_fee_list[k] = performance_fee_fund

                                            if input_value >= 0:
                                                df_prognose.iloc[k + 7, i + 1] = df_prognose.iloc[
                                                                                     k + 7, i + 1] - performance_fee_fund
                                            else:
                                                df_prognose.iloc[k + 7, i + 1] = df_prognose.iloc[k + 7, i]

                                            individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                            # individual_hwm_list[k] = individual_hwm_list[k] - individual_per_fee_list[k]
                                            df_prognose.iloc[4, i + 1] += individual_per_fee_list[k]

                                            if df_prognose.iloc[k + 7, i + 1] < 0:
                                                pass
                                            else:
                                                fund_value += df_prognose.iloc[k + 7, i + 1]

                                    else:
                                        if k == (i + 1):
                                            # individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                            fund_value += df_prognose.iloc[k + 7, i + 1]
                                            df_prognose.iloc[6, i + 1] = fund_value + df_prognose.iloc[2, i + 1]
                                            # individual_hwm_list[0] += df_prognose.iloc[2, i + 1] * (individual_hwm_list[0] / df_prognose.iloc[7, i + 1])
                                            # df_prognose.iloc[7, i + 1] = df_prognose.iloc[7, i + 1] + df_prognose.iloc[2, i + 1]
                                            #   new
                                            # There are K series calculated, the outflow is substracted from the series by the money amount of the series / total amount of all the series,
                                            df_hwm = pd.DataFrame(individual_hwm_list[:])
                                            sum_hwm = df_hwm.sum()
                                            partial_df_hwm = df_hwm / sum_hwm
                                            # The money amount that is substracted from the hwm and the outflow is then multiplied by the (hwm / next value)
                                            new_df_hwm = df_hwm + (partial_df_hwm * df_prognose.iloc[2, i + 1])
                                            df_prognose.iloc[7:, i + 1] = df_prognose.iloc[7:, i + 1].values + (
                                                        partial_df_hwm * df_prognose.iloc[2, i + 1]).squeeze()
                                            df_hwm = df_hwm + (partial_df_hwm * df_prognose.iloc[2, i + 1])
                                            individual_hwm_list = list(df_hwm.iloc[:, 0])
                                            df_prognose.iloc[7:, i + 1] = df_prognose.iloc[7:, i + 1].values + (
                                                        partial_df_hwm * df_prognose.iloc[2, i + 1]).squeeze()

                                            df_prognose.iloc[5, i + 1] = df_prognose.iloc[5, i]

                                        else:
                                            fund_value += df_prognose.iloc[k + 7, i + 1]

                        # The columns after the first column
                        else:
                            for k in range(i + 2):
                                if k <= 48:
                                    if individual_hwm_list[k] < df_prognose.iloc[k + 7, i + 1]:
                                        if k == (i + 1):
                                            individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                            fund_value += df_prognose.iloc[k + 7, i + 1]
                                            df_prognose.iloc[6, i + 1] = fund_value + df_prognose.iloc[2, i + 1]
                                            # individual_hwm_list[0] += df_prognose.iloc[2, i + 1] * (individual_hwm_list[0] / df_prognose.iloc[7, i + 1])
                                            # df_prognose.iloc[7, i + 1] = df_prognose.iloc[7, i + 1] + df_prognose.iloc[2, i + 1]
                                            #   new
                                            # There are K series calculated, the outflow is substracted from the series by the money amount of the series / total amount of all the series,
                                            df_hwm = pd.DataFrame(individual_hwm_list[:])
                                            sum_hwm = df_hwm.sum()
                                            partial_df_hwm = df_hwm / sum_hwm
                                            # The money amount that is substracted from the hwm and the outflow is then multiplied by the (hwm / next value)
                                            new_df_hwm = df_hwm + (partial_df_hwm * df_prognose.iloc[2, i + 1])
                                            df_prognose.iloc[7:, i + 1] = df_prognose.iloc[7:, i + 1].values + (
                                                        partial_df_hwm * df_prognose.iloc[2, i + 1]).squeeze()
                                            df_hwm = df_hwm + (partial_df_hwm * df_prognose.iloc[2, i + 1])
                                            individual_hwm_list = list(df_hwm.iloc[:, 0])

                                            df_prognose.iloc[5, i + 1] = df_prognose.iloc[5, i] + df_prognose.iloc[
                                                4, i + 1]

                                        else:
                                            input_value = individual_hwm_list[k]
                                            output_value = df_prognose.iloc[k + 7, i + 1]
                                            profit = (output_value - input_value)
                                            if input_value >= 0:
                                                performance_fee_fund = profit * df_prognose.iloc[3, i + 1]
                                            individual_per_fee_list[k] = performance_fee_fund
                                            individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                            if input_value >= 0:
                                                df_prognose.iloc[k + 7, i + 1] = individual_hwm_list[k] - \
                                                                                 individual_per_fee_list[k]
                                            else:
                                                df_prognose.iloc[k + 7, i + 1] = df_prognose.iloc[k + 7, i]

                                            individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]

                                            # individual_hwm_list[k] = individual_hwm_list[k] - individual_per_fee_list[k]
                                            df_prognose.iloc[4, i + 1] += individual_per_fee_list[k]

                                            if df_prognose.iloc[k + 7, i + 1] < 0:
                                                pass
                                            else:
                                                fund_value += df_prognose.iloc[k + 7, i + 1]
                                    else:
                                        if k == (i + 1):
                                            # individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                            fund_value += df_prognose.iloc[k + 7, i + 1]
                                            df_prognose.iloc[6, i + 1] = fund_value + df_prognose.iloc[2, i + 1]
                                            # individual_hwm_list[0] += df_prognose.iloc[2, i + 1] * (individual_hwm_list[0] / df_prognose.iloc[7, i + 1])
                                            # df_prognose.iloc[7, i + 1] = df_prognose.iloc[7, i + 1] + df_prognose.iloc[2, i + 1]
                                            #   new
                                            # There are K series calculated, the outflow is substracted from the series by the money amount of the series / total amount of all the series,
                                            df_hwm = pd.DataFrame(individual_hwm_list[:])
                                            sum_hwm = df_hwm.sum()
                                            partial_df_hwm = df_hwm / sum_hwm
                                            # The money amount that is substracted from the hwm and the outflow is then multiplied by the (hwm / next value)
                                            new_df_hwm = df_hwm + (partial_df_hwm * df_prognose.iloc[2, i + 1])
                                            df_prognose.iloc[7:, i + 1] = df_prognose.iloc[7:, i + 1].values + (
                                                        partial_df_hwm * df_prognose.iloc[2, i + 1]).squeeze()
                                            df_hwm = df_hwm + (partial_df_hwm * df_prognose.iloc[2, i + 1])
                                            individual_hwm_list = list(df_hwm.iloc[:, 0])

                                            df_prognose.iloc[5, i + 1] = df_prognose.iloc[5, i]

                                        else:
                                            fund_value += df_prognose.iloc[k + 7, i + 1]

        df_prognose.drop(df_prognose.columns[len(df_prognose.columns)-1], axis=1, inplace=True)
        final_amount_list.append(df_prognose.iloc[4,48])

    print(final_amount_list)
    df = pd.DataFrame(final_amount_list)
    df.columns = ['final_price']
    print(df)
    quantile25 = df.quantile(0.25).iloc[0]
    quantile75 = df.quantile(0.75).iloc[0]
    mean_stat = df.mean().iloc[0]
    maximum = df.max().iloc[0]
    minimum = df.min().iloc[0]
    # std = df.std().iloc[0]
    print(quantile75)
    plt.figure(figsize=(10, 5))
    plt.hist(df['final_price'], bins=100, color='teal')
    plt.axvline(x=mean_stat, color='red', ls='--')
    plt.axvline(x=quantile25, color='purple', ls=':')
    plt.axvline(x=quantile75, color='lime', ls=':')
    plt.axvline(x=minimum, color='blue', ls='-.')
    plt.axvline(x=maximum, color='navy', ls='-.')
    montecarlo_title = str('Prognose simulation: ') + str(len(df))
    plt.title(montecarlo_title, size=16, color='blue', pad=20)
    plt.xlabel('Profit after four years', color='blue')
    plt.ylabel('Estimation amount', color='blue')
    mean_text = str('Mean ') + str(int(mean_stat))
    quantile25_text = str('25th Percentile ') + str(int(quantile25))
    quantile75_text = str('75th Percentile ') + str(int(quantile75))
    min_text = str('Minimum value ') + str(int(minimum))
    max_text = str('Maximum value ') + str(int(maximum))
    print( str(quantile25_text) + str(', ') + str(mean_text) + str(', ') + str(quantile75_text) + str(', ') + str(min_text) + str(', ') + str(max_text))
    plt.legend([mean_text, quantile25_text, quantile75_text, min_text, max_text])

    savefig_text = str('Prognose_simulation_') + str(int(len(df))) + str('.jpg')
    plt.savefig(savefig_text, dpi=200)
    plt.tight_layout()
    plt.show()

    return final_amount_list

def calculate_read_csv():
    # this is a the part that uses the prognose_change csv as input
    # retrieve the data from the excel

    df_prognose = pd.read_csv('prognose_change.csv', index_col = 0)

    year1_input = df_prognose.iloc[8,0]
    year2_input = df_prognose.iloc[9,0]
    year3_input = df_prognose.iloc[10,0]
    year4_input = df_prognose.iloc[11,0]
    std_input = df_prognose.iloc[12,0]
    hwm_start_value = df_prognose.iloc[13,0]
    realisation_months = df_prognose.iloc[14,0]


    yearly_input_list = [year1_input,year2_input,year3_input,year4_input]

    df_prognose.iloc[4:, 1:] = 0

    zero_list =[]
    percentage_list = []
    perf_fee_list = []
    for i in range(49):
        zero_list.append(0)
        percentage_list.append(0)

    # estimate the new return, thinking about the realisation
    for j in range(4):
        found = 0
        input_year = yearly_input_list[j]
        while found == 0:

            temp_value = 1
            temp_value_realisation = 1
            temp_value_with_realisation = 1
            if j == 0:
                new_end_year1 = yearly_input_list[0]

            for i in range(12):

                temp_mean = yearly_input_list[j] / 12
                random_percentage = (np.random.normal(temp_mean, daily_std_btc * std_input, 1))[0]

                if j == 0:
                    if i < int(realisation_months):
                        percentage_list[i] = df_prognose.iloc[0, i + 1]
                        temp_value_with_realisation = temp_value_with_realisation * (1 + df_prognose.iloc[0, i + 1])
                        if i == ((realisation_months) - 1):
                            temp_value_realisation = temp_value_with_realisation
                            new_end_year1 = (((12 - realisation_months) / 12) * year1_input + temp_value_realisation) - 1
                            # print(new_end_year1)
                    else:
                        temp_value_with_realisation = temp_value_with_realisation * (1 + random_percentage)
                        percentage_list[i] = random_percentage
                elif j == 1:
                    percentage_list[12 + i] = random_percentage
                    temp_value = temp_value * (1 + random_percentage)
                elif j == 2:
                    percentage_list[24 + i] = random_percentage
                    temp_value = temp_value * (1 + random_percentage)
                elif j == 3:
                    percentage_list[36 + i] = random_percentage
                    temp_value = temp_value * (1 + random_percentage)

                if j == 0 and i == 11:
                    if (((new_end_year1) - 0.001) <= (temp_value_with_realisation - 1) <= ((new_end_year1) + 0.001)):
                        found = 1
                    else:
                        temp_value_with_realisation = 0
                elif j != 0 and i == 11:
                    if i == 11 and ((input_year - 0.001) <= (temp_value - 1) <= (input_year + 0.001)):
                        found = 1

    new_values = pd.DataFrame(percentage_list[:48]).T
    df_prognose.iloc[0, 1:49] = new_values

    temp_list = []
    for i in range(len(df_prognose)):
        temp_list.append(0)
    df_prognose['last'] = temp_list
    individual_hwm_list = zero_list.copy()
    individual_per_fee_list = zero_list.copy()
    new_serie_list = zero_list.copy()

    individual_hwm_list[0] = hwm_start_value

    # add the inflow to the highwatermark value and to the fund value
    for i in range(48):

        if i == 0:
            temp_df = df_prognose.iloc[8:15, i].copy()
            df_prognose.iloc[8:15, i] = 0
        if i == 1:
            df_prognose.iloc[8:15, 0] = temp_df

        for j in range(52):
            if j < 48:

                # the inflow
                if j == i:
                    df_prognose.iloc[i + 8, j + 1] = df_prognose.iloc[1, j + 1]
                else:
                    pass

            if df_prognose.iloc[0, i + 1] > 0:

                index_on = 0
                # calculate the next step
                if j < 48:
                    if df_prognose.iloc[j + 7, i] > 0:
                        df_prognose.iloc[j + 7, i + 1] = df_prognose.iloc[j + 7, i] * (1 + (df_prognose.iloc[0, i + 1] ))
                    else:
                        # df_prognose.iloc[j + 6, i + 1] = df_prognose.iloc[j + 6, i]
                        df_prognose.iloc[j + 7, i + 2] = 0
                        if df_prognose.iloc[2, i + 1] < 0:
                            if j == 47:
                                index_on = 1

                if j == 48:
                    # initiate new lists. The individual high water mark. individual performance fee
                    fund_value = 0
                    print(i + 1, individual_hwm_list)
                    if i == 0:
                        # individual_hwm_list[0] = hwm_start_value
                        for k in range(i + 2):
                            if k <= 48:
                                if individual_hwm_list[k] < df_prognose.iloc[k + 7, i + 1]:
                                    if k == (i + 1):
                                        individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                        fund_value += df_prognose.iloc[k + 7, i + 1] + df_prognose.iloc[2, i + 1]
                                        df_prognose.iloc[6, i + 1] = fund_value

                                        # outflow in the lead serie for the hwm
                                        # individual_hwm_list[0] += df_prognose.iloc[2, i + 1] * (
                                        #             individual_hwm_list[0] / df_prognose.iloc[7, i + 1])
                                        # df_prognose.iloc[7, i + 1] = df_prognose.iloc[7, i + 1] + df_prognose.iloc[2, i + 1]

                                        #   new
                                        # There are K series calculated, the outflow is substracted from the series by the money amount of the series / total amount of all the series,
                                        df_hwm = pd.DataFrame(individual_hwm_list[:])
                                        df_serie = df_prognose.iloc[7:, i + 1]
                                        df_serie_part = df_serie / df_serie.sum()
                                        df_serie_part_substract = df_serie_part * df_prognose.iloc[2, i + 1]
                                        df_serie = df_serie + df_serie_part_substract
                                        df_hwm = df_hwm.squeeze() + df_serie_part_substract.values * (df_hwm.squeeze() / df_serie.values).fillna(0)
                                        df_prognose.iloc[7:, i + 1] = df_serie
                                        individual_hwm_list = list(df_hwm.iloc[:])



                                        df_prognose.iloc[5, i + 1] = df_prognose.iloc[5, i] + df_prognose.iloc[4, i + 1]
                                    else:
                                        input_value = individual_hwm_list[k]
                                        output_value = df_prognose.iloc[k + 7, i + 1]
                                        profit = (output_value - input_value)
                                        if input_value >= 0:
                                            performance_fee_fund = profit * df_prognose.iloc[3, i + 1]
                                        individual_per_fee_list[k] = performance_fee_fund


                                        if input_value >= 0:
                                            df_prognose.iloc[k + 7, i + 1] = df_prognose.iloc[k + 7, i + 1] - performance_fee_fund
                                        else:
                                            df_prognose.iloc[k + 7, i + 1] = df_prognose.iloc[k + 7, i]

                                        individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                        # individual_hwm_list[k] = individual_hwm_list[k] - individual_per_fee_list[k]
                                        df_prognose.iloc[4, i + 1] += individual_per_fee_list[k]

                                        if df_prognose.iloc[k + 7, i + 1] < 0:
                                            pass
                                        else:
                                            fund_value += df_prognose.iloc[k + 7, i + 1]

                                else:
                                    if k == (i + 1):
                                        # individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                        fund_value += df_prognose.iloc[k + 7, i + 1]
                                        df_prognose.iloc[6, i + 1] = fund_value + df_prognose.iloc[2, i + 1]
                                        # individual_hwm_list[0] += df_prognose.iloc[2, i + 1] * (
                                        #             individual_hwm_list[0] / df_prognose.iloc[7, i + 1])
                                        # df_prognose.iloc[7, i + 1] = df_prognose.iloc[7, i + 1] + df_prognose.iloc[2, i + 1]

                                        #   new
                                        # There are K series calculated, the outflow is substracted from the series by the money amount of the series / total amount of all the series,
                                        df_hwm = pd.DataFrame(individual_hwm_list[:])
                                        df_serie = df_prognose.iloc[7:, i + 1]
                                        df_serie_part = df_serie / df_serie.sum()
                                        df_serie_part_substract = df_serie_part * df_prognose.iloc[2, i + 1]
                                        df_serie = df_serie + df_serie_part_substract
                                        df_hwm = df_hwm.squeeze() + df_serie_part_substract.values * (df_hwm.squeeze() / df_serie.values).fillna(0)
                                        df_prognose.iloc[7:, i + 1] = df_serie
                                        individual_hwm_list = list(df_hwm.iloc[:])


                                        df_prognose.iloc[5, i + 1] = df_prognose.iloc[5, i]

                                    else:
                                        fund_value += df_prognose.iloc[k + 7, i + 1]

                    # The columns after the first one
                    else:
                        for k in range(i + 2):
                            if k <= 48:
                                if individual_hwm_list[k] < df_prognose.iloc[k + 7, i + 1]:
                                    if k == (i + 1):
                                        individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                        fund_value += df_prognose.iloc[k + 7, i + 1]
                                        df_prognose.iloc[6, i + 1] = fund_value + df_prognose.iloc[2, i + 1]

                                        # #   old
                                        # individual_hwm_list[0] += df_prognose.iloc[2, i + 1] * (individual_hwm_list[0] / df_prognose.iloc[7, i + 1])
                                        # df_prognose.iloc[7, i + 1] = df_prognose.iloc[7, i + 1] + df_prognose.iloc[2, i + 1]

                                        #   new
                                        # There are K series calculated, the outflow is substracted from the series by the money amount of the series / total amount of all the series,
                                        df_hwm = pd.DataFrame(individual_hwm_list[:])
                                        df_serie = df_prognose.iloc[7:, i + 1]
                                        df_serie_part = df_serie / df_serie.sum()
                                        df_serie_part_substract = df_serie_part * df_prognose.iloc[2, i + 1]
                                        df_serie = df_serie + df_serie_part_substract
                                        df_hwm = df_hwm.squeeze() + df_serie_part_substract.values * (df_hwm.squeeze() / df_serie.values).fillna(0)
                                        df_prognose.iloc[7:, i + 1] = df_serie
                                        individual_hwm_list = list(df_hwm.iloc[:])


                                        df_prognose.iloc[5, i + 1] = df_prognose.iloc[5, i ] + df_prognose.iloc[4, i + 1]


                                    else:
                                        input_value = individual_hwm_list[k]
                                        output_value = df_prognose.iloc[k + 7, i + 1]
                                        profit = (output_value - input_value)
                                        if input_value >= 0:
                                            performance_fee_fund = profit * df_prognose.iloc[3, i + 1]
                                        individual_per_fee_list[k] = performance_fee_fund
                                        individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                        if input_value >= 0:
                                            df_prognose.iloc[k + 7, i + 1] = individual_hwm_list[k] - individual_per_fee_list[k]
                                        else:
                                            df_prognose.iloc[k + 7, i + 1] = df_prognose.iloc[k + 7, i]

                                        individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]

                                        # individual_hwm_list[k] = individual_hwm_list[k] - individual_per_fee_list[k]
                                        df_prognose.iloc[4, i + 1] += individual_per_fee_list[k]

                                        if df_prognose.iloc[k + 7, i + 1] < 0:
                                            pass
                                        else:
                                            fund_value += df_prognose.iloc[k + 7, i + 1]
                                else:
                                    if k == (i + 1):
                                        # individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                        fund_value += df_prognose.iloc[k + 7, i + 1]
                                        df_prognose.iloc[6, i + 1] = fund_value + df_prognose.iloc[2, i + 1]
                                        # individual_hwm_list[0] += df_prognose.iloc[2, i + 1] * (individual_hwm_list[0] / df_prognose.iloc[7, i + 1])
                                        # df_prognose.iloc[7, i + 1] = df_prognose.iloc[7, i + 1] + df_prognose.iloc[2, i + 1]
                                        #   new
                                        # There are K series calculated, the outflow is substracted from the series by the money amount of the series / total amount of all the series,
                                        df_hwm = pd.DataFrame(individual_hwm_list[:])
                                        df_serie = df_prognose.iloc[7:, i + 1]
                                        df_serie_part = df_serie / df_serie.sum()
                                        df_serie_part_substract = df_serie_part * df_prognose.iloc[2, i + 1]
                                        df_serie = df_serie + df_serie_part_substract
                                        df_hwm = df_hwm.squeeze() + df_serie_part_substract.values * (df_hwm.squeeze() / df_serie.values).fillna(0)
                                        df_prognose.iloc[7:, i + 1] = df_serie
                                        individual_hwm_list = list(df_hwm.iloc[:])



                                        df_prognose.iloc[5, i + 1] = df_prognose.iloc[5, i]

                                    else:
                                        fund_value += df_prognose.iloc[k + 7, i + 1]



            # if the return is negative
            else:
                if j < 48:

                    if df_prognose.iloc[j + 7, i] > 0:
                        df_prognose.iloc[j + 7, i + 1] = df_prognose.iloc[j + 7, i] * (1 + (df_prognose.iloc[0, i + 1]))
                    else:
                        # df_prognose.iloc[j + 6, i + 1] = df_prognose.iloc[j + 6, i]
                        df_prognose.iloc[j + 7, i + 2] = 0
                        if df_prognose.iloc[1, i + 1] < 0:
                            if j == 47:
                                index_on = 1

                if j == 48:
                    # initiate new lists. The individual high water mark. individual performance fee
                    fund_value = 0
                    print(i + 1, individual_hwm_list)
                    if i == 0:
                        # individual_hwm_list[0] = hwm_start_value
                        for k in range(i + 2):
                            if k <= 48:
                                if individual_hwm_list[k] < df_prognose.iloc[k + 7, i + 1]:
                                    if k == (i + 1):
                                        individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                        fund_value += df_prognose.iloc[k + 7, i + 1] + df_prognose.iloc[2, i + 1]
                                        df_prognose.iloc[6, i + 1] = fund_value

                                        # outflow in the lead serie for the hwm
                                        # individual_hwm_list[0] += df_prognose.iloc[2, i + 1]
                                        # individual_hwm_list[0] += df_prognose.iloc[2, i + 1] * (individual_hwm_list[0] / df_prognose.iloc[7, i + 1])
                                        # df_prognose.iloc[7, i + 1] = df_prognose.iloc[7, i + 1] + df_prognose.iloc[2, i + 1]
                                        #   new
                                        # There are K series calculated, the outflow is substracted from the series by the money amount of the series / total amount of all the series,
                                        df_hwm = pd.DataFrame(individual_hwm_list[:])
                                        df_serie = df_prognose.iloc[7:, i + 1]
                                        df_serie_part = df_serie / df_serie.sum()
                                        df_serie_part_substract = df_serie_part * df_prognose.iloc[2, i + 1]
                                        df_serie = df_serie + df_serie_part_substract
                                        df_hwm = df_hwm.squeeze() + df_serie_part_substract.values * (df_hwm.squeeze() / df_serie.values).fillna(0)
                                        df_prognose.iloc[7:, i + 1] = df_serie
                                        individual_hwm_list = list(df_hwm.iloc[:])



                                        df_prognose.iloc[5, i + 1] = df_prognose.iloc[5, i] + df_prognose.iloc[4, i + 1]


                                    else:
                                        input_value = individual_hwm_list[k]
                                        output_value = df_prognose.iloc[k + 7, i + 1]
                                        profit = (output_value - input_value)
                                        if input_value >= 0:
                                            performance_fee_fund = profit * df_prognose.iloc[3, i + 1]
                                        individual_per_fee_list[k] = performance_fee_fund


                                        if input_value >= 0:
                                            df_prognose.iloc[k + 7, i + 1] = df_prognose.iloc[k + 7, i + 1] - performance_fee_fund
                                        else:
                                            df_prognose.iloc[k + 7, i + 1] = df_prognose.iloc[k + 7, i]

                                        individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                        # individual_hwm_list[k] = individual_hwm_list[k] - individual_per_fee_list[k]
                                        df_prognose.iloc[4, i + 1] += individual_per_fee_list[k]

                                        if df_prognose.iloc[k + 7, i + 1] < 0:
                                            pass
                                        else:
                                            fund_value += df_prognose.iloc[k + 7, i + 1]

                                else:
                                    if k == (i + 1):
                                        # individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                        fund_value += df_prognose.iloc[k + 7, i + 1]
                                        df_prognose.iloc[6, i + 1] = fund_value + df_prognose.iloc[2, i + 1]
                                        # individual_hwm_list[0] += df_prognose.iloc[2, i + 1] * (individual_hwm_list[0] / df_prognose.iloc[7, i + 1])
                                        # df_prognose.iloc[7, i + 1] = df_prognose.iloc[7, i + 1] + df_prognose.iloc[2, i + 1]
                                        #   new
                                        # There are K series calculated, the outflow is substracted from the series by the money amount of the series / total amount of all the series,
                                        df_hwm = pd.DataFrame(individual_hwm_list[:])
                                        df_serie = df_prognose.iloc[7:, i + 1]
                                        df_serie_part = df_serie / df_serie.sum()
                                        df_serie_part_substract = df_serie_part * df_prognose.iloc[2, i + 1]
                                        df_serie = df_serie + df_serie_part_substract
                                        df_hwm = df_hwm.squeeze() + df_serie_part_substract.values * (df_hwm.squeeze() / df_serie.values).fillna(0)
                                        df_prognose.iloc[7:, i + 1] = df_serie
                                        individual_hwm_list = list(df_hwm.iloc[:])

                                        df_prognose.iloc[5, i + 1] = df_prognose.iloc[5, i]

                                    else:
                                        fund_value += df_prognose.iloc[k + 7, i + 1]

                    # The columns after the first column
                    else:
                        for k in range(i + 2):
                            if k <= 48:
                                if individual_hwm_list[k] < df_prognose.iloc[k + 7, i + 1]:
                                    if k == (i + 1):
                                        individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                        fund_value += df_prognose.iloc[k + 7, i + 1]
                                        df_prognose.iloc[6, i + 1] = fund_value + df_prognose.iloc[2, i + 1]
                                        # individual_hwm_list[0] += df_prognose.iloc[2, i + 1] * (individual_hwm_list[0] / df_prognose.iloc[7, i + 1])
                                        # df_prognose.iloc[7, i + 1] = df_prognose.iloc[7, i + 1] + df_prognose.iloc[2, i + 1]
                                        #   new
                                        # There are K series calculated, the outflow is substracted from the series by the money amount of the series / total amount of all the series,
                                        df_hwm = pd.DataFrame(individual_hwm_list[:])
                                        df_serie = df_prognose.iloc[7:, i + 1]
                                        df_serie_part = df_serie / df_serie.sum()
                                        df_serie_part_substract = df_serie_part * df_prognose.iloc[2, i + 1]
                                        df_serie = df_serie + df_serie_part_substract
                                        df_hwm = df_hwm.squeeze() + df_serie_part_substract.values * (df_hwm.squeeze() / df_serie.values).fillna(0)
                                        df_prognose.iloc[7:, i + 1] = df_serie
                                        individual_hwm_list = list(df_hwm.iloc[:])



                                        df_prognose.iloc[5, i + 1] = df_prognose.iloc[5, i ] + df_prognose.iloc[4, i + 1]

                                    else:
                                        input_value = individual_hwm_list[k]
                                        output_value = df_prognose.iloc[k + 7, i + 1]
                                        profit = (output_value - input_value)
                                        if input_value >= 0:
                                            performance_fee_fund = profit * df_prognose.iloc[3, i + 1]
                                        individual_per_fee_list[k] = performance_fee_fund
                                        individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                        if input_value >= 0:
                                            df_prognose.iloc[k + 7, i + 1] = individual_hwm_list[k] - individual_per_fee_list[k]
                                        else:
                                            df_prognose.iloc[k + 7, i + 1] = df_prognose.iloc[k + 7, i]

                                        individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]

                                        # individual_hwm_list[k] = individual_hwm_list[k] - individual_per_fee_list[k]
                                        df_prognose.iloc[4, i + 1] += individual_per_fee_list[k]

                                        if df_prognose.iloc[k + 7, i + 1] < 0:
                                            pass
                                        else:
                                            fund_value += df_prognose.iloc[k + 7, i + 1]
                                else:
                                    if k == (i + 1):
                                        # individual_hwm_list[k] = df_prognose.iloc[k + 7, i + 1]
                                        fund_value += df_prognose.iloc[k + 7, i + 1]
                                        df_prognose.iloc[6, i + 1] = fund_value + df_prognose.iloc[2, i + 1]
                                        # individual_hwm_list[0] += df_prognose.iloc[2, i + 1] * (individual_hwm_list[0] / df_prognose.iloc[7, i + 1])
                                        # df_prognose.iloc[7, i + 1] = df_prognose.iloc[7, i + 1] + df_prognose.iloc[2, i + 1]
                                        #   new
                                        # There are K series calculated, the outflow is substracted from the series by the money amount of the series / total amount of all the series,
                                        df_hwm = pd.DataFrame(individual_hwm_list[:])
                                        df_serie = df_prognose.iloc[7:, i + 1]
                                        df_serie_part = df_serie / df_serie.sum()
                                        df_serie_part_substract = df_serie_part * df_prognose.iloc[2, i + 1]
                                        df_serie = df_serie + df_serie_part_substract
                                        df_hwm = df_hwm.squeeze() + df_serie_part_substract.values * (df_hwm.squeeze() / df_serie.values).fillna(0)
                                        df_prognose.iloc[7:, i + 1] = df_serie
                                        individual_hwm_list = list(df_hwm.iloc[:])



                                        df_prognose.iloc[5, i + 1] = df_prognose.iloc[5, i]

                                    else:
                                        fund_value += df_prognose.iloc[k + 7, i + 1]


                            elif k == 49:
                                df_prognose.iloc[7, i + 1] = df_prognose.iloc[7, i + 1] + df_prognose.iloc[1, i + 1] + df_prognose.iloc[2, i + 1]
                    index_on = 0
    print('Considering the realisation the percentage after one year will be: ' + str(new_end_year1 + 1))
    print('nice')
    df_prognose.drop(df_prognose.columns[len(df_prognose.columns)-1], axis=1, inplace=True)

    df_prognose.to_csv('prognose_change.csv')











