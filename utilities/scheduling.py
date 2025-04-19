import pandas as pd
from collections import OrderedDict

class Schedule:
    def __init__(self, schedule_df):
        self.df = schedule_df
        self.running_epoch_to_parameters = OrderedDict()
        running_epoch = 0
        schedule_df_dict = schedule_df.to_dict(orient='index')  # {0: {'phase': 1, 'epochs': 1000, ...}, 1: {...} ...}
        for index in range(len(schedule_df_dict)):
            phase_dict = schedule_df_dict[index]  # {'phase': 1, 'epochs': 1000, ...}
            running_epoch += phase_dict['epochs']
            self.running_epoch_to_parameters[running_epoch] = phase_dict

    def parameters(self, epoch):
        for maximum_epochs, params in self.running_epoch_to_parameters.items():  # (1000, {'phase': 1, 'epochs': 1000, ...})
            if epoch <= maximum_epochs:
                return params
        # The end was reached: return the parameters from the last phase
        return params

    def last_epoch(self):
        last_epoch_key = list(self.running_epoch_to_parameters)[-1]
        return last_epoch_key