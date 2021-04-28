#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 18:14:29 2021

@author: sebastian
"""

import numpy as np 
import os as os 
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

################################### Lectura de datos##########################
orig = os.getcwd()
fn_vient = os.path.join(orig, 'viento1.csv')
fn_dir = os.path.join(orig, 'direccion1.csv')
fn_temp = os.path.join(orig, 'temperatura1.csv')


data_vel = pd.read_csv(fn_vient,header = 0, delimiter=';')
data_dir = pd.read_csv(fn_dir,header = 0, delimiter=';')
data_temp = pd.read_csv(fn_temp,header = 0, delimiter=';')

dates = pd.date_range(start='03/01/2020 01:00', end='03/01/2021 23:00', freq='H')

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
fig1.savefig('Serie de Tiempo  de vientos parque OHiggins', dpi=400)

fig2 = plt.figure(2)
fig2.clf()
ax4 = fig2.add_subplot(111)
ax4.plot(Data.index, Data.Temperatura)
ax4.set_xlabel('Fecha')
ax4.set_ylabel('Temperatura [°C]')
ax4.set_title('Serie de tiempo de Temperatura en estación Parque OHiggins')
fig2.savefig('Serie de Tiempo  de Temperatura parque OHiggins', dpi=400)


################################ Histogramas #################################

##########################  magnitud ########################
magnitud=np.sqrt(Data.u**2+Data.v**2)
textstr = '\n'.join((
    r'$\mu=%.2f$ [m/s]' % (magnitud.mean(), ),
    r'$\sigma=%.2f$ [m/s]' % (magnitud.std(), )))

fig3 = plt.figure(3)
ax5 = fig3.add_subplot(111)
ax5.hist(magnitud, bins = 30, density=True)
ax5.text(2.5, 1, textstr
    , fontsize=10, va='bottom')
ax5.set_title('Histograma Magnitud')
ax5.set_xlabel('Magnitud [m/s]', fontsize=12)
ax5.set_ylabel('Probabilidad', fontsize = 12)
fig3.savefig('Histograma temperatura.png',dpi= 400)

########################## Temperatura ########################
textstr = '\n'.join((
    r'$\mu=%.2f$ [°C]' % (Data.Temperatura.mean(), ),
    r'$\sigma=%.2f$ [°C]' % (Data.Temperatura.std(), )))

fig4 = plt.figure(4)
ax6 = fig4.add_subplot(111)
ax6.hist(Data.Temperatura, bins = 30, density=True)
ax6.text(20, 0.045, textstr
    , fontsize=10, va='bottom')
ax6.set_title('Histograma Temperatura')
ax6.set_xlabel('Temperatura [°C]', fontsize=12)
ax6.set_ylabel('Probabilidad', fontsize = 12)
fig4.savefig('Histograma temperatura.png',dpi= 400)

########################## Dirección ########################
textstr1 = '\n'.join((
    r'$\mu_u=%.2f$ [m/s]' % (Data.u.mean(), ),
    r'$\sigma_u=%.2f$ [m/s]' % (Data.u.std(), )))
textstr2 = '\n'.join((
    r'$\mu_v=%.2f$ [m/s]' % (Data.v.mean(), ),
    r'$\sigma_v=%.2f$ [m/s]' % (Data.v.std(), )))

fig5 = plt.figure(5)
ax7 = fig5.add_subplot(211)
ax8 = fig5.add_subplot(212)
ax7.hist(Data.u, bins = 30, density=True)
ax8.hist(Data.v, bins = 30, density=True)
ax7.text(2, 0.5, textstr1
    , fontsize=10, va='bottom')
ax8.text(2, 0.5, textstr2
    , fontsize=10, va='bottom')
ax7.set_title('Histograma Dirección')
ax8.set_xlabel('Direccion [m/s]', fontsize=12)
ax7.set_ylabel('Probabilidad u', fontsize = 12)
ax8.set_ylabel('Probabilidad v', fontsize = 12)
ax7.set_xlim(-2, 4)
ax8.set_xlim(-2, 4)
fig5.savefig('Histograma temperatura.png',dpi= 400)



Data_aux= Data['2020':'2021']
################################ Vientos Promedios############################
#u_diario = Data_aux.u.groupby(Data_aux.index.hour).mean()
#v_diario = Data_aux.v.groupby(Data_aux.index.hour).mean()
#Viento_magnitud_diario = np.sqrt(u_diario**2 + v_diario**2)
#Dir_horario = np.mod(270 - 180*np.angle(u_diario+v_diario*1j)/np.pi,360)
#fig6 = plt.figure(6)
#ax9 = fig6.add_subplot(111)
#ax9.plot(u_diario.index, Viento_magnitud_diario)
#ax9.set_xlabel('Hora')
#ax9.set_ylabel('Magnitud de Viento [m/s]')
#ax9.set_title('Ciclo Diario de Magnitud de viento en estación Parque OHiggins')
#fig6.savefig('Ciclo diario magnitud de viento Parque OHiggins', dpi=400)



#################################### Ciclo Diario Temperatura ########################
ciclo_diario = Data_aux.groupby(Data_aux.index.hour).mean()
fig7 = plt.figure(7)
fig7.clf()
ax10 = fig7.add_subplot(111)
ax10.plot(ciclo_diario.index, ciclo_diario.Temperatura)
ax10.set_xlabel('Hora')
ax10.set_ylabel('Temperatura [°C]')
ax10.set_title('Ciclo Diario de Temperatura en estación Parque OHiggins')
fig7.savefig('Ciclo Diario Temperatura parque OHiggins', dpi=400)

#################################### Ciclo Diario Velocidad ########################
u_diario = Data_aux.u.groupby(Data_aux.index.hour).mean()
v_diario = Data_aux.v.groupby(Data_aux.index.hour).mean()
Viento_magnitud_diario = np.sqrt(u_diario**2 + v_diario**2)
fig12 = plt.figure(12)
fig12.clf()
ax16 = fig12.add_subplot(111)
ax16.plot(ciclo_diario.index, ciclo_diario.velocidad)
ax16.set_xlabel('Hora')
ax16.set_ylabel('velocidad [m/s]')
ax16.set_title('Ciclo Diario de Magnitud de viento en estación Parque OHiggins')
fig12.savefig('Ciclo Diario magnitud de viento parque OHiggins', dpi=400)

#################################### Ciclo Diario Direccion ########################
u_diario = Data_aux.u.groupby(Data_aux.index.hour).mean()
v_diario = Data_aux.v.groupby(Data_aux.index.hour).mean()
Dir_diario = np.mod(270 - 180*np.angle(u_diario+v_diario*1j)/np.pi,360)

fig13 = plt.figure(13)
fig13.clf()
ax17 = fig13.add_subplot(111)
ax17.plot(ciclo_diario.index, Dir_diario)
ax17.set_xlabel('Hora')
ax17.set_ylabel('Direccion [m/s]')
ax17.set_title('Ciclo Diario de Direccion en estación Parque OHiggins')
fig13.savefig('Ciclo Diario Direccion parque OHiggins', dpi=400)


ciclo_estacional = Data_aux.groupby(Data_aux.index.month).mean()
###################### Ciclo Estacional magnitud ########################
u_est = Data_aux.u.groupby(Data_aux.index.month).mean()
v_est = Data_aux.v.groupby(Data_aux.index.month).mean()
Viento_magnitud_estacional = np.sqrt(u_est**2 + v_est**2)

fig8 = plt.figure(8)
fig8.clf()
ax11 = fig8.add_subplot(111)
ax11.plot(ciclo_estacional.index, Viento_magnitud_estacional)
ax11.set_xlabel('Mes')
ax11.set_ylabel('Velocidad [m/s]')
ax11.set_title('Ciclo Estacional de Magnitud en estación Parque OHiggins')
fig8.savefig('Ciclo Estacional magnitud parque OHiggins', dpi=400)

###################### Ciclo Estacional Dirección ########################
u_est = Data_aux.u.groupby(Data_aux.index.month).mean()
v_est = Data_aux.v.groupby(Data_aux.index.month).mean()
Dir_horario = np.mod(270 - 180*np.angle(u_est+v_est*1j)/np.pi,360)
fig9 = plt.figure(9)
fig9.clf()
ax12 = fig9.add_subplot(111)

ax12.plot(ciclo_estacional.index, Dir_horario)

ax12.set_xlabel('Mes')
ax12.set_ylabel('Dirección[m/s]')
ax12.set_title('Ciclo Estacional de Dirección en estación Parque OHiggins')
ax12.legend()

fig9.savefig('Ciclo Estacional Direccion parque OHiggins', dpi=400)

###################### Ciclo Estacional Temperatura ########################
fig10 = plt.figure(10)
fig10.clf()
ax15 = fig10.add_subplot(111)
ax15.plot(ciclo_estacional.index, ciclo_estacional.Temperatura)
ax15.set_xlabel('Mes')
ax15.set_ylabel('Temperatura [°C]')
ax15.set_title('Ciclo Estacional de Temperatura en estación Parque OHiggins')
fig10.savefig('Ciclo Estacional Temperatura parque OHiggins', dpi=400)



####################################### Mes Hora #############################
mes_hora = Data_aux.Temperatura.groupby([Data_aux.Temperatura.index.month, 
                                      Data_aux.Temperatura.index.hour]).mean()

X = np.linspace(1,24,24)
Y = np.linspace(1,12,12)
Z = mes_hora.unstack()

x, y = np.meshgrid(X,Y)

fig11 = plt.figure(11)
ax13 = fig11.add_subplot(111)
im = ax13.contourf(x, y, Z)
ax13.contour(x,y,Z, colors='k')
ax13.set_xlabel("Hora", loc='center', fontsize=15)
ax13.set_ylabel("Mes", loc='center', fontsize=15)
divider = make_axes_locatable(ax13)
cax = divider.append_axes("right", size="5%", pad=0.35)
plt.colorbar(im, cax=cax)
plt.ylabel("Temperatura [°C]", fontsize=15)
fig11.savefig('Diagrama Mes-Hora Temperatura parque OHiggins', dpi=400)

