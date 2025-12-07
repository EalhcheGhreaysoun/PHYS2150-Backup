import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
takes three files with data generated from the dataProcessing.py file and creates a csv file with coefficients for the lines of best fit and a plot of the lines of best fit with a scatter plot of the data.
'''

#set the below properly before running;
C60_00_file_path = 'outputData/C60_00_mean_for_graph.csv'
C60_05_file_path = 'outputData/C60_05_mean_for_graph.csv'
C60_10_file_path = 'outputData/C60_10_mean_for_graph.csv'
output_path = 'outputData/best_fit_coefficient.csv'
APPROXIMATION_ORDER = 3
C60_00_color = 'b'
C60_05_color = 'r'
C60_10_color = 'y'



C60_00_df = pd.read_csv(C60_00_file_path)
C60_05_df = pd.read_csv(C60_05_file_path)
C60_10_df = pd.read_csv(C60_10_file_path)

C60_00_time = C60_00_df['hours']
C60_00_eqe  = C60_00_df['EQE']

C60_05_time = C60_05_df['hours']
C60_05_eqe  = C60_05_df['EQE']

C60_10_time = C60_10_df['hours']
C60_10_eqe  = C60_10_df['EQE']

C60_eqe = np.array([C60_00_eqe, C60_05_eqe, C60_10_eqe])

coeff = np.array([np.polyfit(C60_00_time, eqe, APPROXIMATION_ORDER) for eqe in C60_eqe])

p_00 = np.poly1d(coeff[0])
p_05 = np.poly1d(coeff[1])
p_10 = np.poly1d(coeff[2])

xp = np.linspace(min(C60_00_time), max(C60_00_time), 100)

plt.scatter(C60_00_time, C60_00_eqe, color=C60_00_color)
plt.scatter(C60_05_time, C60_05_eqe, color=C60_05_color)
plt.scatter(C60_10_time, C60_10_eqe, color=C60_10_color)

plt.plot(xp, p_00(xp), color='b')
plt.plot(xp, p_05(xp), color='r')
plt.plot(xp, p_10(xp), color='y')
plt.savefig('outputData/C60_cycles_best_fit.png')

coeff_df = pd.DataFrame(coeff.T, columns=['00 coefficients', '05 coefficients', '10 coefficients'])

coeff_df.to_csv(output_path)
