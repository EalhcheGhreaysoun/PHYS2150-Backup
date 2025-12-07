import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

C60_00_df = pd.read_csv('outputData/C60_00_mean_for_graph.csv')
C60_05_df = pd.read_csv('outputData/C60_05_mean_for_graph.csv')
C60_10_df = pd.read_csv('outputData/C60_10_mean_for_graph.csv')

C60_00_time = C60_00_df['hours']
C60_00_eqe  = C60_00_df['EQE']

C60_05_time = C60_05_df['hours']
C60_05_eqe  = C60_05_df['EQE']

C60_10_time = C60_10_df['hours']
C60_10_eqe  = C60_10_df['EQE']

C60_eqe = np.array([C60_00_eqe, C60_05_eqe, C60_10_eqe])

coeff = np.array([np.polyfit(C60_00_time, eqe, 3) for eqe in C60_eqe])
print(coeff)
p_00 = np.poly1d(coeff[0])
p_05 = np.poly1d(coeff[1])
p_10 = np.poly1d(coeff[2])

xp = np.linspace(min(C60_00_time), max(C60_00_time), 100)

plt.scatter(C60_00_time, C60_00_eqe, color='b')
plt.scatter(C60_05_time, C60_05_eqe, color='r')
plt.scatter(C60_10_time, C60_10_eqe, color='y')

plt.plot(xp, p_00(xp), color='b')
plt.plot(xp, p_05(xp), color='r')
plt.plot(xp, p_10(xp), color='y')
plt.savefig('outputData/C60_cycles_best_fit.png')

coeff_df = pd.DataFrame(coeff.T, columns=['00 coefficients', '05 coefficients', '10 coefficients'])

coeff_df.to_csv('outputData/best_fit_coefficient.csv')
