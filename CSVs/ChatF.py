import pandas as pd

# Cargar el archivo CSV
data = pd.read_csv('CSVs\mfcc_1_segmentos.csv')

# Extraer las columnas relevantes y filtrar por la categoría "zanahoria"
data['Category'] = data['Unnamed: 0'].str.extract(r'(\D+)_')
zanahoria_data = data[data['Category'] == 'berenjena'][['Segmento_1','Segmento_2','Segmento_3','Segmento_4','Segmento_5','Segmento_6', 'Segmento_7', 'Segmento_8']]

# Calcular la desviación estándar de cada segmento
variability = zanahoria_data.std()
print(variability.sort_values())
