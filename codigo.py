import numpy as np 
import os as os 
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

################################### Lectura de datos##########################
orig = os.getcwd()
fn_vient = os.path.join(orig, 'Viento.csv')
fn_dir = os.path.join(orig, 'Direccion.csv')
fn_temp = os.path.join(orig, 'Temperatura.csv')

data_vel = pd.read_csv(fn_vient,header = 0, delimiter=';')
data_dir = pd.read_csv(fn_dir,header = 0, delimiter=';')
data_temp = pd.read_csv(fn_temp,header = 0, delimiter=';')
dates = pd.date_range(start='01/03/2020 01:00', end='01/03/2021 23:00', freq='H')
data_vel = data_vel.stack().str.replace(',','.').unstack()
data_dir = data_dir.stack().str.replace(',','.').unstack()
data_temp = data_temp.stack().str.replace(',','.').unstack()

Vel = pd.to_numeric(data_vel['Unnamed: 2'], errors='coerce')
Dir = pd.to_numeric(data_dir['Unnamed: 2'], errors='coerce')
Temp = pd.to_numeric(data_temp['Unnamed: 2'], errors='coerce')

############################## DataFrame ################################

Data = pd.DataFrame(data={'velocidad':Vel.values, 
                          'Direccion':Dir.values,
                          'Temperatura':Temp.values}, 
                           index=dates)

u = -Data["velocidad"]*np.sin(np.deg2rad(Data.Direccion))
v = -Data["velocidad"]*np.cos(np.deg2rad(Data.Direccion))
Data = Data.assign(u=u.values, v=v.values)

fig1 = plt.figure(1)
fig1.clf()
ax1 = fig1.add_subplot(311)
ax1.plot(Data.index, Data.velocidad)
ax2 = fig1.add_subplot(312)
ax2.plot(Data.index, Data.u)
ax3 = fig1.add_subplot(313)
ax3.plot(Data.index, Data.v)
ax1.set_title('Serie de tiempo de vientos en estación parque OHiggins')
ax1.set_ylabel('$V_{Magnitud}$ [m/s]')
ax2.set_ylabel('u [m/s]')
ax3.set_ylabel('v [m/s]')
ax3.set_xlabel('Fecha')
fig1.savefig('Serie de Tiempo de vientos Parque OHiggins', dpi=400)

fig2 = plt.figure(2)
fig2.clf()
ax4 = fig2.add_subplot(111)
ax4.plot(Data.index, Data.Temperatura)
ax4.set_xlabel('Fecha')
ax4.set_ylabel('Temperatura [°C]')
ax4.set_title('Serie de tiempo de Temperatura en estación Parque OHiggins')
fig2.savefig('Serie de Tiempo de Temperatura Parque OHiggins', dpi=400)

################################ Filtro ######################################
Data.Temperatura  <-10

Data[Data.Temperatura <-10]=np.nan

fig3 = plt.figure(3)
fig3.clf()
ax5 = fig3.add_subplot(111)
ax5.plot(Data.index, Data.Temperatura)
ax5.set_xlabel('Fecha')
ax5.set_ylabel('Temperatura [°C]')
ax5.set_title('Serie de tiempo de Temperatura filtrada en estación Parque OHiggins')
fig3.savefig('Serie de Tiempo de Temperatura filtrada Parque OHigginsa', dpi=400)