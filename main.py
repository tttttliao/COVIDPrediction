import pandas
import numpy as np
from datetime import datetime

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%m-%d-%Y")
    d2 = datetime.strptime(d2, "%m-%d-%Y")
    return abs((d2 - d1).days)

base_line_date = "04-12-2020"
states = ["Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]
state_to_idx_dict = {}
for idx, state in enumerate(states):
  state_to_idx_dict[state] = idx

def pre_process(data):
     # Drop ID column as it's not helpful in our prediction.
    first_column = data.columns[0]
    data = data.drop([first_column], axis=1)
    data = data.values

    # Change state name to corresponding state indices.
    # Change date to difference from baseline date.
    for i in range(data.shape[0]):
        data[i][0] = state_to_idx_dict[data[i][0]]
        data[i][1] = days_between(base_line_date, data[i][1])
    confirmed = data[:,2]
    death = data[:,3]
    data = np.delete(data, 2, 1)
    data = np.delete(data, 2, 1)

    return data, confirmed, death

def main():
    data = pandas.read_csv("data/train.csv")
    data, confirmed, death = pre_process(data)

if __name__ == '__main__':
    main()