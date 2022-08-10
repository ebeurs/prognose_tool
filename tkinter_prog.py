from prognose import *
import tkinter as tk
# def calculate_write_csv(startvaluefund,monthlyaverageinflow,perfomancefee,year1input,year2input,year3input,year4input,stddeviateinput,hwmstartvalue):

fields = 'Start value of the fund', 'Monthly inflow', 'Performance fee', ' % growth year 1', ' % growth year 2', ' % growth year 3', ' % growth year 4', 'The daily std of btc times ', 'Highwater mark start value'

def fetch(entries):
    for entry in entries:
        field = entry[0]
        value = np.float(entry[1].get())
        print('%s: "%s"' % (field, value))

def fetch1(entries):
    try:
        global input_list
        input_list = []
        for entry in entries:
            field = entry[0]
            value = np.float(entry[1].get())
            input_list.append(value)
            print('%s: "%s"' % (field, value))
        input_list_string = str('The input is correct :  \n ') + str(input_list)
        answer.config(text=input_list_string)
        switchButtonState()
    except:
        answer.config(text='INCORRECT, only numbers are possible!')
    # answer = tk.Label(master=window, text="...")

def makeform(window, fields):
    entries = []
    for field in fields:
        row = tk.Frame(window)
        lab = tk.Label(row, width=25, text=field, anchor='w')
        ent = tk.Entry(row)
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries.append((field, ent))
    return entries


def switchButtonState():
    if (button1['state'] == tk.DISABLED):
        button1['state'] = tk.NORMAL
    else:
        button1['state'] = tk.DISABLED

def amount_running():
    amount = amount_running_entry.get()
    final_amount_list = calculate_read_csv_calculate(int(amount))
    print(final_amount_list)
    df_final_amount = pd.DataFrame(final_amount_list)
    df_final_amount.to_csv('final_list.csv')

if __name__ == '__main__':
    window = tk.Tk()
    greeting = tk.Label(master=window, text="DEAR CFO, LET'S ESTIMATE!")
    answer = tk.Label(master=window, text="...")
    button1 = tk.Button(master=window,
                        text="CALCULATE WITHOUT CSV",
                        width=40,
                        height=1,
                        bg="blue",
                        fg="white",
                        state = tk.DISABLED,
                        command = lambda: calculate_write_csv(input_list[0],input_list[1],input_list[2],input_list[3],input_list[4],input_list[5],input_list[6],input_list[7],input_list[8],0)
                        )

    button2 = tk.Button(master=window,
                        text="CALCULATE WITH CSV AS INPUT",
                        width=40,
                        height=1,
                        bg="blue",
                        fg="white",
                        command=calculate_read_csv
                        )

    button3 = tk.Button(master=window,
                        text="CALCULATE MULTIPLE TIMES",
                        width=40,
                        height=1,
                        bg="blue",
                        fg="white",
                        command = amount_running
                        )




    # frame = tk.Frame(master=window, width=450, height=180)
    greeting.pack(padx=5, pady=5)
    answer.pack(padx=5, pady=5)
    ents = makeform(window, fields)
    window.bind('<Return>', (lambda event, e=ents: fetch(e)))
    b1 = tk.Button(window, text='Check correctness',
                   width=20,
                   height=1,
                   bg="blue",
                   fg="white",
                   command=(lambda e=ents: fetch1(e)))


    b1.pack(padx=5, pady=5)

    button1.pack(padx=5, pady=20)

    canvas = tk.Canvas(window, width=250, height=4, bg = "grey")
    canvas.pack(padx=5, pady=20)

    button2.pack( padx=5, pady=20)

    canvas = tk.Canvas(window, width=250, height=4, bg = "grey")
    canvas.pack(padx=5, pady=20)

    button3.pack( padx=5, pady=20)

    amount_running_entry = tk.Entry(window)
    amount_running_entry.pack( padx=5, pady=20)



    window.mainloop()





print('nice')