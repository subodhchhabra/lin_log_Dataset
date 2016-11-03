import pandas as pd
import sqlite3
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from smote import SMOTE

#loads into db zip->lat,long
db_connection = sqlite3.connect(":memory:")
zip_codes_lat_long = pd.read_csv('info.data')
zip_codes_lat_long.to_sql("TEST", db_connection)
db_connection.commit()
cur = db_connection.cursor()
colors = ['rx','bx']

#convert zip to lat, long
def convert_zip(zip):
    cur.execute("SELECT LAT,LNG FROM TEST WHERE ZIP=?;", (int(zip), ))
    return cur.fetchone()

info_raw = np.loadtxt('seed.csv', delimiter=',')
zip_code = np.asarray([convert_zip(i) for i in info_raw[:, 1]])
print(zip_code.shape)
print(info_raw[:,0].shape)
final = np.hstack((zip_code, info_raw[:,0].reshape((zip_code.shape[0], 1))))
final = SMOTE(final, 400, 4)

#plot on a map
dis = 500
m  = Basemap(projection='ortho',lon_0=-119,lat_0=37,resolution='i',
             llcrnrx=-1000*dis,llcrnry=-1000*dis,
             urcrnrx=+1150*dis,urcrnry=+1700*dis)
m.drawcoastlines()
m.drawstates()
m.drawcountries()
x, y = m(final[:,1], final[:,0])
for i in range(x.size):
    m.plot(x[i], y[i], colors[int(final[:,2][i])] , markersize=4)
plt.show()
