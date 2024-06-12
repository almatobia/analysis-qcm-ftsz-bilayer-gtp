# analysis-qcm-ftsz-bilayer-gtp

Código empleado para el análisis exhaustivo de los datos obtenidos con el software de la microbalanza de cuarzo.
Este programa nos proporciona un fichero con los datos de frecuencia y disipación para cada sobretono en cada momento del experimento.
Con estos datos, buscamos graficar el experimento completo y ver a grandes rasgos los efectos que provocamos sobre el cristal.
También, vamos a separar la parte de la bicapa y el resto del experimento (anclaje de la proteína y polimerización con GTP), para poder estudiarlos más a fondo.
Obtenemos distintas gráficas, ajustes gaussianos, etc. para poder obtener los valores que deseamos en cada momento, como por ejemplo, el efecto de pasar la proteína diluida sobre la bicapa formada sobre el cristal.
En el script llamado `experimentLibrary.py` se definen todas las funciones que utilizaremos para llevar a cabo el análisis y el notebook llamado `analizar_datos.ipynb` muestra los análisis realizados para cada uno de los experimentos realizados.
