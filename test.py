
'''

the following code requires the modules listed below, namely:
numpy as p
matplotlib.pyplot as plt
pandas pd
os
h5py

the code requires the following file structure:

parent:
---> dataAnalysis.py
---> processedData
    |---> 
---> rawData
    |---> data_fa25.h5

The log files lists operations that the code performs, including data omission, reading files out of the H5 file, and more, Do note that it overwrites each time the code runs and previous logs are deleted.

WARNINGS:
* do note that the code generates about 2000 files in the rawData folder, about 280 files in the processedData folder, 4 images and a log file.
  the large amount of files might overload some cloud based system, however, this has not been tested

* currently, the C60 05 cycle set has missing data for 2025_10_03, but i'm not sure why. To be figured out

'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import h5py


# Constants
h = 6.62607015e-34  # Planck's constant (Joule second)
c = 3.0e8  # Speed of light (meters per second)
e = 1.602176634e-19  # Elementary charge (Coulombs)
active_area_cm2 = 0.14  # Active area in cm^2
active_area_m2 = active_area_cm2 * 1e-4  # Convert cm^2 to m^2

# array of dates for looping and file detection
dates = np.array(['2025_09_09', "2025_09_11", '2025_09_16', '2025_09_23', '2025_09_25', '2025_09_30', '2025_10_03', '2025_10_14', '2025_10_16', '2025_10_21', '2025_10_23', '2025_10_28', '2025_10_30'])

# initialize arrays and dataframes
eqe_day = np.empty(0)
eqe_current_df = pd.DataFrame()
eqe = pd.DataFrame()
power_df = pd.DataFrame()

# cell # in each set of C60 cycles
C60_00Cycle = np.array(['129', '131', '132', '134', '135', '136', '141', '145', '146', '161', '162', '170', '184', '190', '194'])
C60_05Cycle = np.array(['139', '143', '144', '157', '166', '172', '174', '176', '187', '189', '191', '193', '196'])
C60_10Cycle = np.array(['126', '127', '130', '133', '137', '138', '147', '158', '159', '163', '164', '169', '179', '180', '182', '183', '186', '195'])
all_cells = np.concatenate((C60_00Cycle, C60_05Cycle, C60_10Cycle))

# arrays that will contain file names for each set
cycle00_files = np.empty(0)
cycle05_files = np.empty(0)
cycle10_files = np.empty(0)

#print to signal to user that the processing has begun
print('running...')


#clears log file
open('dataAnalysis.log', 'w').close()

#resuable function for averaging data across C60 sets
def average_pixel_data(files_list, df):
    temp_df = pd.DataFrame()
    df["Wavelength (nm)"] = pd.read_csv('processedData/eqe_results_2025_09_11_cell170.csv')['Wavelength (nm)']
    for file in files_list:
        i = 2
        temp_df[f'EQE pixel 1'] = pd.read_csv(f'processedData/{file}')['EQE pixel 1']
        while i < 9:  #adds all pixel values of the current file to the temp df
            temp_df[f"EQE pixel {i}"] = pd.read_csv(f'processedData/{file}')[f"EQE pixel {i}"]
            i += 1

        temp_df["average EQE"] = temp_df.sum(axis=1)/8
        df[f'{file[12:16]} {file[17:19]} {file[20:22]} cell {file[27:30]}'] = temp_df['average EQE']
        temp_df = temp_df.iloc[0:0]

#resuable function for slicing data
def slice_data(df, lower_range, upper_range):
    temp = df[df["Wavelength (nm)"] > lower_range]
    temp = temp[temp["Wavelength (nm)"] < upper_range]
    return temp

#function for getting power data
def get_power_data(hdf5_path, cell_number, date):
    """
    Retrieve Power data (wavelength vs power) for a given cell and date.

    Power is measured once per cell (not per pixel).
    UPDATED FOR FA25: Cell numbers now use 3-digit padding.

    Parameters:
    -----------
    hdf5_path : str
        Path to the HDF5 file.
    cell_number : str or int
        Cell number (e.g., 129 or '129').
    date : str
        Date in the format 'YYYY_MM_DD' (e.g., '2025_09_23').

    Returns:
    --------
    pd.DataFrame or None
        DataFrame containing columns ['Wavelength (nm)', 'Power (W)'],
        or None if not found.

    Example:
    --------
    power_df = get_power_data(hdf5_path, 129, '2025_09_23')
    """
    # FA25: Ensure 3-digit zero-padding
    cell_number = str(cell_number).zfill(3)

    with h5py.File(hdf5_path, 'r') as hdf:
        # Construct the full path to the Power data for this cell and date
        group_path = f'Cell{cell_number}/Power/{date}'

        # Check if the data exists at this path
        if group_path in hdf:
            group = hdf[group_path]
            # Read the numerical data array
            data = group['Data'][:]
            # Read and decode the column headers (stored as bytes)
            headers = [h.decode('utf-8') for h in group['Headers'][:]]
            # Create a pandas DataFrame with the data and headers
            df = pd.DataFrame(data, columns=headers)
            return df
        else:
            with open('dataAnalysis.log', 'a') as f:
                    f.write(f"Power data for Cell{cell_number} on {date} not found.\n")
            return None

#function for getting current data
def get_current_data(hdf5_path, cell_number, pixel_number, date):
    """
    Retrieve Current data (wavelength vs current) for a specific cell, pixel, and date.

    Current is measured separately for each pixel.
    UPDATED FOR FA25: Cell numbers now use 3-digit padding.

    Parameters:
    -----------
    hdf5_path : str
        Path to the HDF5 file.
    cell_number : str or int
        Cell number (e.g., 129 or '129').
    pixel_number : int
        Pixel number (1-8 for FA25).
    date : str
        Date in the format 'YYYY_MM_DD'.

    Returns:
    --------
    pd.DataFrame or None
        DataFrame containing columns ['Wavelength (nm)', 'Current (A)'],
        or None if not found.

    Example:
    --------
    current_df = get_current_data(hdf5_path, 129, 4, '2025_09_23')
    """
    # FA25: Ensure 3-digit zero-padding
    cell_number = str(cell_number).zfill(3)
    pixel_number = str(pixel_number)

    with h5py.File(hdf5_path, 'r') as hdf:
        # Construct the full path to the Current data
        group_path = f'Cell{cell_number}/Pixel{pixel_number}/Current/{date}'

        if group_path in hdf:
            group = hdf[group_path]
            data = group['Data'][:]
            headers = [h.decode('utf-8') for h in group['Headers'][:]]
            df = pd.DataFrame(data, columns=headers)
            return df
        else:
            with open('dataAnalysis.log', 'a') as f:
                f.write(f"Current data for Cell{cell_number}, Pixel{pixel_number} on {date} not found.\n")
            return None

#function for writing data from h5 file to csvs
def retrieve_and_write_data(hdf5_path, cells, dates):
    '''
    batch retrieve and write data from HDF5 file as csv files

    creates files of the following form:
    YYYY_MM_DD_power_cell###.csv
    YYYY_MM_DD_current_cell###pixel#.csv
    '''
    for date in dates:
        for cell in cells:
            # Retrieve Power data (one per cell, not per pixel)
            power_data = get_power_data(hdf5_path, cell, date)
            if power_data is not None:
                # Create a descriptive variable name
                var_name = f'rawData/{date}_power_cell{cell}.csv'
                # Store the DataFrame as a global variable
                power_data.to_csv(var_name, index=False)
            
            # Retrieve Current data (specific to this pixel)
            pixel = 1
            while pixel <= 8:
                current_data = get_current_data(hdf5_path, cell, pixel, date)
                #current_data.drop(current_data.columns[0], axis=1)
                if current_data is not None:
                    var_name = f'rawData/{date}_current_cell{cell}_pixel{pixel}.csv'
                    current_data.to_csv(var_name, index=False)
                pixel += 1


#checks if data from the h5 file has be processed, if not, processes data
if len(os.listdir('rawData')) == 1:
    retrieve_and_write_data('rawData/data_fa25.h5', C60_00Cycle, dates)
    retrieve_and_write_data('rawData/data_fa25.h5', C60_05Cycle, dates)
    retrieve_and_write_data('rawData/data_fa25.h5', C60_10Cycle, dates)
else:
    with open('dataAnalysis.log', 'a') as f:
        f.write('skipping get data from h5 file as files already exist. \n')

#code for processing data
#loops through all dates
if len(os.listdir('processedData')) == 0:
    for date in dates:
        #loops through all cells
        for cell in all_cells:
            #check if the given date and cell already have processed data, cuts down on computation time and redundant operations
            processed_data_names = os.listdir('processedData')
            if not np.isin(f'eqe_results_{date}_cell{cell}.csv', processed_data_names):
                #loops through the files in 'rawData'
                for fileName in os.listdir('rawData'):
                    #if the file  is the power data file for the given date and cell, add it to the temp array of files
                    if fileName == f'{date}_power_cell{cell}.csv':
                        eqe_day = np.append(eqe_day, fileName)
                    #if the file is one of the current data files for the given date and cell, add it to the temp array of files
                    if fileName[0:26] == f'{date}_current_cell{cell}':
                        eqe_day = np.append(eqe_day, fileName)
                #checks if the number of files that have been found for a given date and cell are the required number
                if eqe_day.size > 8:
                    eqe_day = np.sort(eqe_day)
                    # reads power data from file
                    power_df = pd.read_csv(f'rawData/{eqe_day[-1]}')

                    # reads wavelength data from second file in array, it does not matter which file we read from, but the first entry in the array contains information about the array type
                    eqe_current_df["Wavelength (nm)"] = pd.read_csv(f"rawData/{date}_current_cell{eqe_day[1][23:26]}_pixel{1}.csv")["Wavelength (nm)"]

                    # loops over all files and adds current data to the dataframe
                    i = 0
                    while i < 8:
                        eqe_current_df[f"Pixel {i+1} Current (A)"] = pd.read_csv(f"rawData/{date}_current_cell{eqe_day[1][23:26]}_pixel{i+1}.csv")["Current (A)"]
                        i += 1
                    
                    # inits eqe dataframe and adds wavelength as the first column
                    eqe = pd.DataFrame()
                    eqe["Wavelength (nm)"] = eqe_current_df["Wavelength (nm)"]
                    # loops over all the columns in the current df and does math to convert it to eqe
                    i = 0
                    while i < 8:
                        eqe[f"EQE pixel {i+1}"] = ((eqe_current_df[f"Pixel {i+1} Current (A)"] / power_df['Power (W)']) * (h * c / (e * eqe_current_df["Wavelength (nm)"]*1e-9)))
                        i += 1
                    # writes final EQE data for each date for each cell to a file in folder processed data
                    eqe.to_csv(f'processedData/eqe_results_{date}_cell{eqe_day[1][23:26]}.csv', index=False)
                    with open('dataAnalysis.log', 'a') as f:
                        f.write(f'processed data from {eqe_day[0]} \n')
                else:
                    with open('dataAnalysis.log', 'a') as f:
                        f.write(f'insufficient number of files for {date} and cell {cell}. Found files are as follows: {eqe_day}\n')
                # clears the EQE file name array for next date
                eqe_day = np.empty(0)
else:
    with open('dataAnalysis.log', 'a') as f:
        f.write('skipping data processing as processed data already exists. \n')

# loops over files in processed data and sorts file names into arrays according to C60 cycles
# if cell does not exist in any of the C60 cycle sets, ommits data and prints that data is ommited from that cell
for fileName in os.listdir('processedData'):
    if (fileName[27:30] in C60_00Cycle):
        cycle00_files = np.append(cycle00_files, fileName)
        with open('dataAnalysis.log', 'a') as f:
            f.write(f"added {fileName} to cycles 00\n")
    elif (fileName[27:30] in C60_05Cycle):            
        cycle05_files = np.append(cycle05_files, fileName)
        with open('dataAnalysis.log', 'a') as f:
           f.write(f"added {fileName} to cycles 05\n")
    elif (fileName[27:30] in C60_10Cycle):
        cycle10_files = np.append(cycle10_files, fileName)
        with open('dataAnalysis.log', 'a') as f:
            f.write(f"added {fileName} to cycles 10\n")
    else:
        with open('dataAnalysis.log', 'a') as f:
            f.write(f'invalid cell: {fileName}. Ommiting data')
        
#inits df for C60 data
C60_00_df = pd.DataFrame()
C60_05_df = pd.DataFrame()
C60_10_df = pd.DataFrame()

#set data frame for 00 cycles and slices off unneeded wavelengths
average_pixel_data(cycle00_files, C60_00_df)
C60_00_df = slice_data(C60_00_df, 700, 750)
#set data frame for 05 cycles and slices off unneeded wavelengths
average_pixel_data(cycle05_files, C60_05_df)
C60_05_df = slice_data(C60_05_df, 700, 750)
#set data frame for 10 cycles and slices off unneeded wavelengths
average_pixel_data(cycle10_files, C60_10_df)
C60_10_df = slice_data(C60_10_df, 700, 750)

#creates a list of dataframes
C60_cycles = [C60_00_df, C60_05_df, C60_10_df]

#initializes dataframes
df_day = pd.DataFrame()
mean_eqe_C60_00 = pd.DataFrame()
mean_eqe_C60_05 = pd.DataFrame()
mean_eqe_C60_10 = pd.DataFrame()
#creates a list of mean eqe dataframes
mean_eqe = [mean_eqe_C60_00, mean_eqe_C60_05, mean_eqe_C60_10]
sdev = np.empty(0)

#loops over all dates
for date in dates:
    for i, cycle_data in enumerate(C60_cycles):
        mean_eqe[i]["Wavelength (nm)"] = C60_00_df['Wavelength (nm)']
        #loops over columns in the 0 cycle dataframe
        for column in cycle_data.columns:
            # if the first characters of the file name match the date...
            if column[0:10] == (date[0:4]+' '+date[5:7] + ' ' + date[8:10]):
                # add it to the df_day dataframe
                df_day[column] = cycle_data[column]
                sdev = np.append(sdev, cycle_data[column].std())
                df_day = df_day.copy()
        #once all data for a certain day is added, take the mean and assign it to the mean eqe data frame with the correct date
        mean_eqe[i][f"{date} average"] = df_day.mean(axis=1)
        #clears the temp data frame for reuse
        df_day = df_day.iloc[:0]

mean_mean_eqe_C60_00 = np.empty(0)
mean_mean_eqe_C60_05 = np.empty(0)
mean_mean_eqe_C60_10 = np.empty(0)
#drops all empty columns so that there are no redundant empty columns
for i, df_name in enumerate(mean_eqe):
    mean_eqe[i] = mean_eqe[i].dropna(axis=0, how='all')

columns = mean_eqe_C60_00.columns
plt.figure(figsize=(10, 10))
for i, name in enumerate(columns):
    if i != 0:
        plt.plot(mean_eqe_C60_00['Wavelength (nm)'], mean_eqe_C60_00[columns[i]], color=((i/len(columns)), 0, (len(columns) - i)/len(columns)))
        plt.errorbar(mean_eqe_C60_00['Wavelength (nm)'], mean_eqe_C60_00[columns[i]], yerr=sdev[i],color=((i/len(columns)), 0, (len(columns) - i)/len(columns)), fmt='o', capsize=5, capthick=2)
plt.legend(np.delete(columns, 0), loc='upper right')
plt.xlabel('Wavelength (nm)')
plt.ylabel('EQE')
plt.savefig('C60_00_cycle_day_average.png')

columns = mean_eqe_C60_05.columns
plt.figure(figsize=(10, 10))
for i, name in enumerate(columns):
    if i != 0:
        plt.plot(mean_eqe_C60_05['Wavelength (nm)'], mean_eqe_C60_05[columns[i]], color=((i/len(columns)), 0, (len(columns) - i)/len(columns)))
        plt.errorbar(mean_eqe_C60_00['Wavelength (nm)'], mean_eqe_C60_00[columns[i]], yerr=sdev[i],color=((i/len(columns)), 0, (len(columns) - i)/len(columns)), fmt='o', capsize=5, capthick=2)
plt.legend(np.delete(columns, 0), loc='upper right')
plt.xlabel('Wavelength (nm)')
plt.ylabel('EQE')
plt.savefig('C60_05_cycle_day_average.png')


columns = mean_eqe_C60_10.columns
plt.figure(figsize=(10, 10))
for i, name in enumerate(columns):
    if i != 0:
        plt.plot(mean_eqe_C60_10['Wavelength (nm)'], mean_eqe_C60_10[columns[i]], color=((i/len(columns)), 0, (len(columns) - i)/len(columns)))
        plt.errorbar(mean_eqe_C60_00['Wavelength (nm)'], mean_eqe_C60_00[columns[i]], yerr=sdev[i],color=((i/len(columns)), 0, (len(columns) - i)/len(columns)), fmt='o', capsize=5, capthick=2)
plt.legend(np.delete(columns, 0), loc='upper right')
plt.xlabel('Wavelength (nm)')
plt.ylabel('EQE')
plt.savefig('C60_10_cycle_day_average.png')


mean_mean_eqe_C60_00 = mean_eqe_C60_00.mean()
mean_mean_eqe_C60_05 = mean_eqe_C60_05.mean()
mean_mean_eqe_C60_10 = mean_eqe_C60_10.mean()

t = np.linspace(0, 10, len(mean_mean_eqe_C60_00)-1)

plt.figure(figsize=(12, 9))
plt.plot(t, np.delete(mean_mean_eqe_C60_00, 0))
plt.plot(t, np.delete(mean_mean_eqe_C60_05, 0))
plt.plot(t, np.delete(mean_mean_eqe_C60_10, 0))

plt.legend(["00 cycle", "05 cycle", "10 cycle"])
plt.savefig('C60_00_cycle_total_average.png')

#signals to user that processing has ended
print('complete.')

