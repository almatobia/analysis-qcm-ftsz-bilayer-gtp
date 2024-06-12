import os, re
import matplotlib.axes
import matplotlib.figure
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from lmfit.models import GaussianModel

class Experiment:

    # Constructor
    def __init__(
            self, fileName: str, fileSep: str = '\t', timeStart: int | float = 0, timeUnits: str = 'min',
            dataDir: str = 'datos', figDir: str = 'figuras', resultDir: str = 'resultados',
            saveFigs: bool = True, saveFormat: str = 'png', saveResults: bool = True, discardedArmonics = 1):
        
        # Guardar propiedades
        self.fileName = fileName
        self.timeUnits = timeUnits
        self._dataDir = dataDir
        self._figDir = figDir
        self._resultDir = resultDir
        self._saveFigs = saveFigs
        self._saveFormat = saveFormat
        self._saveResults = saveResults

        # Crear directorios
        if not os.path.exists(self._figDir):
            os.makedirs(self._figDir)

        if not os.path.exists(self._resultDir):
            os.makedirs(self._resultDir)

        # Definir unidades de disipación
        self._disipUnitLabel = ' · 10^6' # No cambiar, está hardcodeado
        
        # Cargar los datos
        filePath = os.path.join(self._dataDir, self.fileName)
        self.df = pd.read_csv(filePath, sep = fileSep)
        self.colNames = self.df.columns.tolist()
        self.timeColName = self.colNames[0]
        
        # Cambiar unidades de tiempo
        if self.timeUnits == 'min':
            self.df[self.timeColName] = self.df[self.timeColName] / 60

        # Setear el origen de tiempo a 0
        self.df = self.df[self.df[self.timeColName] >= 0]
        self.df = self.df.reset_index()
        self.df[self.timeColName] = self.df[self.timeColName] - self.df[self.timeColName][0]
        if timeStart != 0:
            self.df = self.df[self.df[self.timeColName] >= timeStart]
            self.df = self.df.reset_index()
            self.df[self.timeColName] = self.df[self.timeColName] - self.df[self.timeColName][0]

        # Detectar armónicos
        self.armonicos = set()
        for colName in self.colNames:
            match = re.search(r'n=(\d+)', colName)
            if match:
                self.armonicos.add(int(match.group(1)))

        if discardedArmonics is not None:
            if isinstance(discardedArmonics, int):
                self.armonicos.discard(discardedArmonics)
            else:
                for n in discardedArmonics:
                    self.armonicos.discard(n)

        print('Armónicos encontrados:\n', self.armonicos)

        for n in self.armonicos:

            frecColName = f'Delta_F/n_n={n:d}_(Hz)'
            disipColName = f'Delta_D_n={n:d}_()'
            
            # Setear los orígenes de frecuencia y disipación a 0
            self.df[frecColName] = self.df[frecColName] - self.df[frecColName][0]
            self.df[disipColName] = self.df[disipColName] - self.df[disipColName][0]

            # Cambiar orden de magnitud de disipación
            self.df[disipColName] = self.df[disipColName] * 1e6
    
    def filterTime(self, start = None, end = None):
        """
        Filtra los datos en tiempo en función de los inputs.
        
        Argumentos:
        - start (None o num): Tiempo inicial.
        - end (None o num): Tiempo final.
        """
        # Filtrar por tiempo inicial
        if start is not None:
            self.df = self.df[(self.df[self.timeColName] >= start)]

        # Filtrar por tiempo final
        if end is not None:
            self.df = self.df[(self.df[self.timeColName] <= end)]

        # Resetear el índice de los datos
        self.df = self.df.reset_index()

        return self
    
    def checkBadPoints(self, minValidValue = None, maxValidValue = None, checkFrecuency = False, checkDisipation = False):
        """
        Elimina puntos anómalos.
        """

        # Comprobar argumentos
        if not (checkFrecuency or checkDisipation):
            raise Exception('Debes setear a True al menos una de las dos opciones: checkFrecuency, checkDisipation.')
        
        if not (minValidValue or maxValidValue):
            raise Exception('Debes setear un valor para al menos una de las dos opciones: minValidValue, maxValidValue.')
        
        # Escoger columnas a comprobar
        colsToCheck = []
        if checkFrecuency:
            colsToCheck.extend([f'Delta_F/n_n={n:d}_(Hz)' for n in self.armonicos])
        if checkDisipation:
            colsToCheck.extend([f'Delta_D_n={n:d}_()' for n in self.armonicos])
        
        # Filtrar por valor mínimo
        if minValidValue is not None:
            self.df = self.df[(self.df[colsToCheck] >= minValidValue).all(axis = 1)]

        # Filtrar por valor máximo
        if maxValidValue is not None:
            self.df = self.df[(self.df[colsToCheck] <= maxValidValue).all(axis = 1)]

        return self
    
    def plot(self, timeLabels: dict[str, list[int|float]] | None = None, saveFig: bool | None = None, disipAxisLims = None):
        """
        Representa todos los datos del experimento.

        Argumentos:
        - timeLabels (dict o None): Diccionario de etiqueta: tiempo.
        - saveFig (bool o None): Control de guardado de figuras. Por defecto, sigue el control global.
        - disipAxisLims(list o None): Lista o equivalente de longitud 2 con los límites de eje Y del plot de disipación.
        """

        # Crear la figura
        fig, axs = plt.subplots(2, 1, sharex = 'col')
        fig.suptitle(self.fileName)

        for n in self.armonicos:

            frecColName = f'Delta_F/n_n={n:d}_(Hz)'
            disipColName = f'Delta_D_n={n:d}_()'

            # Plot de frecuencia vs time
            self.df.plot(x = self.timeColName, y = frecColName, ax = axs[0], label = f'n = {n:d}')

            # # Plot de disipación vs time
            self.df.plot(x = self.timeColName, y = disipColName, ax = axs[1], label = f'n = {n:d}')

        if timeLabels is not None:
            # Obtener valores máximos para ajustar la posición de las etiquetas
            maxFrec, maxDisip = - np.inf, - np.inf
            for n in self.armonicos:

                frecColName = f'Delta_F/n_n={n:d}_(Hz)'
                disipColName = f'Delta_D_n={n:d}_()'

                maxFrec = max(maxFrec, self.df[frecColName].max())
                maxDisip = max(maxDisip, self.df[disipColName].max())
            
            # Coger límite de disipación si está seteado
            if disipAxisLims is not None:
                maxDisip = 0.95 * disipAxisLims[1]

            for label, timeValue in timeLabels.items():
                # Añadir etiquetas en el plot de frecuencia
                axs[0].axvline(x = timeValue - 0.5, color = 'gray', linestyle = '--')
                axs[0].text(timeValue, maxFrec, label, color = 'black',
                            fontsize = 8, ha = 'left', va = 'top')
                
                # Añadir etiquetas en el plot de disipación
                axs[1].axvline(x = timeValue - 0.5, color = 'gray', linestyle = '--')
                axs[1].text(timeValue, maxDisip, label, color = 'black',
                            fontsize = 8, ha = 'left', va = 'top')


        # Setear opciones del plot de frecuencias
        axs[0].set_xlabel(f'Time_({self.timeUnits})')
        axs[0].set_ylabel('Delta_F/n (Hz)')
        #axs[0].yaxis.set_major_locator(MultipleLocator(20))
        axs[0].legend(bbox_to_anchor = (1, 1))
        axs[0].grid(True)

        # Setear opciones del plot de disipaciones
        axs[1].set_xlabel(f'Time_({self.timeUnits})')
        axs[1].set_ylabel(f'Delta_D{self._disipUnitLabel}')
        if disipAxisLims is not None:
            axs[1].set_ylim(disipAxisLims)
        axs[1].legend(bbox_to_anchor = (1, 1))
        axs[1].grid(True)

        fig.tight_layout()
        self._saveFigFn(fig, saveFig)
        plt.show(block = False)

        return self
    
    def plotNice(
            self, timeLabels: dict[str, list[int|float]] | None = None, saveFig: bool | None = None, disipAxisLims = None,
            figSize: tuple[int] = (18, 8), labelSize: int = 14, legendSize: int = 10, legendLoc: tuple[int] | str = 'best'):
        """
        Representa todos los datos del experimento con estilo mejorado.

        Argumentos:
        - timeLabels (dict o None): Diccionario de etiqueta: tiempo.
        - saveFig (bool o None): Control de guardado de figuras. Por defecto, sigue el control global.
        - disipAxisLims(list o None): Lista o equivalente de longitud 2 con los límites de eje Y del plot de disipación.
        """

        # Crear la figura
        fig, axs = plt.subplots(2, 1, sharex = 'col', figsize = figSize)

        for n in self.armonicos:

            frecColName = f'Delta_F/n_n={n:d}_(Hz)'
            disipColName = f'Delta_D_n={n:d}_()'

            # Plot de frecuencia vs time
            self.df.plot(x = self.timeColName, y = frecColName, ax = axs[0], label = f'n = {n:d}', linewidth = 1)

            # # Plot de disipación vs time
            self.df.plot(x = self.timeColName, y = disipColName, ax = axs[1], linewidth = 1, legend = False)

        if timeLabels is not None:
            # Obtener valores máximos para ajustar la posición de las etiquetas
            maxFrec, maxDisip = - np.inf, - np.inf
            for n in self.armonicos:

                frecColName = f'Delta_F/n_n={n:d}_(Hz)'
                disipColName = f'Delta_D_n={n:d}_()'

                maxFrec = max(maxFrec, self.df[frecColName].max())
                maxDisip = max(maxDisip, self.df[disipColName].max())
            
            # Coger límite de disipación si está seteado
            if disipAxisLims is not None:
                maxDisip = 0.95 * disipAxisLims[1]

            for label, timeValue in timeLabels.items():
                # Añadir etiquetas en el plot de frecuencia
                axs[0].axvline(x = timeValue, color = 'gray', linestyle = '--')
                axs[0].text(timeValue + 0.7, maxFrec, label, color = 'black',
                            fontsize = labelSize, ha = 'left', va = 'top')
                
                # Añadir etiquetas en el plot de disipación
                axs[1].axvline(x = timeValue, color = 'gray', linestyle = '--')
                axs[1].text(timeValue + 0.7, maxDisip, label, color = 'black',
                            fontsize = labelSize, ha = 'left', va = 'top')


        # Setear opciones del plot de frecuencias
        axs[0].set_xlabel(f't / {self.timeUnits}')
        axs[0].set_ylabel(r'$\Delta$F / Hz')
        axs[0].legend(loc = legendLoc, prop = {'size': legendSize})

        # Setear opciones del plot de disipaciones
        axs[1].set_xlabel(f't / {self.timeUnits}')
        axs[1].set_ylabel(r'$\Delta$D / $10^{-6}$')
        if disipAxisLims is not None:
            axs[1].set_ylim(disipAxisLims)

        fig.tight_layout()

        return self
    
    def analyzeBilayer(self, startPlateau, endPlateau, disipAxisLims = None, addValuesToPlot: bool = False, saveFig: bool | None = None, saveResult: bool | None = None):
        """
        Analiza y representa los datos de la creación de la bicapa definida por los plateaus de comienzo y final.
        
        Argumentos:
        - startPlateau (list): Lista o equivalente de longitud 2 que contiene los tiempos inicial y final del plateau anterior a la formación de la bicapa.
        - endPlateau (list): Lista o equivalente de longitud 2 que contiene los tiempos inicial y final del plateau posterior a la formación de la bicapa.
        - disipAxisLims(list o None): Lista o equivalente de longitud 2 con los límites de eje Y del plot de disipación.
        - addValuesToPlot (bool): Control de colocar los valores de los plateaus sobre la gráfica.
        - saveFig (bool o None): Control de guardado de figuras.
        - saveResult (bool o None): Control de guardado de ficheros.
        """
        # Filtrar los datos de la bicapa
        dfBilayer = self.df[(self.df[self.timeColName] >= startPlateau[0]) & (self.df[self.timeColName] <= endPlateau[1])].copy()

        # Representar la formación de la bicapa
        fig, axs = plt.subplots(2, 1, sharex = 'col')
        fig.suptitle(f'{self.fileName} - Bilayer formation')

        for n in self.armonicos:

            frecColName = f'Delta_F/n_n={n:d}_(Hz)'
            disipColName = f'Delta_D_n={n:d}_()'

            # Plot de frecuencia vs time
            dfBilayer.plot(x = self.timeColName, y = frecColName, ax = axs[0], label = f'n = {n:d}')

            # # Plot de disipación vs time
            dfBilayer.plot(x = self.timeColName, y = disipColName, ax = axs[1], label = f'n = {n:d}')

        # Añadir etiquetas de los valores de los plateaus
        if addValuesToPlot:
            self._addTextLabels(dfBilayer, startPlateau, axs)
            self._addTextLabels(dfBilayer, endPlateau, axs)

        # Setear opciones del plot de frecuencias
        axs[0].set_xlabel(f'Time_({self.timeUnits})')
        axs[0].set_ylabel('Delta_F/n (Hz)')
        axs[0].yaxis.set_major_locator(MultipleLocator(20))
        axs[0].legend(bbox_to_anchor = (1, 1))
        axs[0].grid(True)

        # Setear opciones del plot de disipaciones
        axs[1].set_xlabel(f'Time_({self.timeUnits})')
        axs[1].set_ylabel(f'Delta_D{self._disipUnitLabel}')
        if disipAxisLims is not None:
            axs[1].set_ylim(disipAxisLims)
        axs[1].legend(bbox_to_anchor = (1, 1))
        axs[1].grid(True)

        # Ajustar los subplots
        fig.tight_layout()

        # Guardar la figura
        self._saveFigFn(fig, saveFig, '_bilayer')

        # Mostrar la figura
        plt.show(block = False)

        # Filtrar datos de los plateaus de inicio y final
        startDf = dfBilayer[(dfBilayer[self.timeColName] >= startPlateau[0]) & (dfBilayer[self.timeColName] <= startPlateau[1])]
        endDf = dfBilayer[(dfBilayer[self.timeColName] >= endPlateau[0]) & (dfBilayer[self.timeColName] <= endPlateau[1])]

        # Crear la figura
        fig, axs = plt.subplots(2, 2, sharex = False, sharey = False)
        fig.suptitle(f'{self.fileName} - Bilayer Gaussian Fit')

        # DataFrame para guardar resultados
        gaussianResults = pd.DataFrame(
            columns = [
                'armonico', 'meanStartTime', 'meanEndTime',
                'frecStartCenter', 'frecStartSd', 'frecEndCenter', 'frecEndSd',
                'disipStartCenter', 'disipStartSd', 'disipEndCenter', 'disipEndSd'])

        # Calcular y representar gaussianas
        for n in self.armonicos:

            frecColName = f'Delta_F/n_n={n:d}_(Hz)'
            disipColName = f'Delta_D_n={n:d}_()'

            # Frecuencia
            frecStartResult = fitAndPlotGaussian(startDf[frecColName].values, axs[0,0], f'n = {n:d}')
            frecEndResult = fitAndPlotGaussian(endDf[frecColName].values, axs[0,1], f'n = {n:d}')

            # Disipación
            disipStartResult = fitAndPlotGaussian(startDf[disipColName].values, axs[1,0], f'n = {n:d}')
            disipEndResult = fitAndPlotGaussian(endDf[disipColName].values, axs[1,1], f'n = {n:d}')

            # Añadir los resultados al DataFrame
            gaussianResults.loc[gaussianResults.shape[0]] = {
                'armonico': n, 'meanStartTime': np.mean(startPlateau), 'meanEndTime': np.mean(endPlateau),
                'frecStartCenter': frecStartResult['center'], 'frecStartSd': frecStartResult['sigma'],
                'frecEndCenter': frecEndResult['center'], 'frecEndSd': frecEndResult['sigma'],
                'disipStartCenter': disipStartResult['center'], 'disipStartSd': disipStartResult['sigma'],
                'disipEndCenter': disipEndResult['center'], 'disipEndSd': disipEndResult['sigma']}
        
        # Calcular las variaciones de frecuencia y disipación
        gaussianResults['frecVariation'] = gaussianResults['frecEndCenter'] - gaussianResults['frecStartCenter']
        gaussianResults['disipVariation'] = gaussianResults['disipEndCenter'] - gaussianResults['disipStartCenter']

        # Setear propiedades de los ejes
        axs[0,0].set_xlabel('Delta_F/n (Hz)')
        axs[0,0].set_ylabel('Frecuencia acumulada')
        axs[0,0].set_title(f'Equilibrio inicial ({startPlateau[0]}-{startPlateau[1]})')
        axs[0,0].grid(True)
        axs[0,1].set_xlabel('Delta_F/n (Hz)')
        axs[0,1].set_ylabel('Frecuencia acumulada')
        axs[0,1].set_title(f'Equilibrio final ({endPlateau[0]}-{endPlateau[1]})')
        axs[0,1].legend(bbox_to_anchor = (1, 1))
        axs[0,1].grid(True)
        axs[1,0].set_xlabel(f'Delta_D{self._disipUnitLabel}')
        axs[1,0].set_ylabel('Frecuencia acumulada')
        axs[1,0].set_title(f'Equilibrio inicial ({startPlateau[0]}-{startPlateau[1]})')
        axs[1,0].grid(True)
        axs[1,1].set_xlabel(f'Delta_D{self._disipUnitLabel}')
        axs[1,1].set_ylabel('Frecuencia acumulada')
        axs[1,1].set_title(f'Equilibrio final ({endPlateau[0]}-{endPlateau[1]})')
        axs[1,1].legend(bbox_to_anchor = (1, 1))
        axs[1,1].grid(True)

        # Ajustar los subplots
        fig.tight_layout()

        # Guardar la figura
        self._saveFigFn(fig, saveFig, '_bilayerGaussians')

        # Mostrar la figura
        plt.show(block = False)

        # Almacenar los datos en la clase
        self.bilayerResults = gaussianResults

        # Escribir resultados en txt
        self._saveResultFn(gaussianResults, saveResult, '_bilayerGaussians')

        # Representar variaciones frente a armonicos
        fig, axs =  plt.subplots(2, 1, sharex = 'col')
        fig.suptitle(f'{self.fileName} - Bilayer variation')

        axs[0].plot(gaussianResults['armonico'], gaussianResults['frecVariation'], '.-')
        axs[1].plot(gaussianResults['armonico'], gaussianResults['disipVariation'], '.-')

        # Setear opciones del plot de frecuencia
        axs[0].set_xlabel('n')
        axs[0].set_ylabel('variación de Delta_F/n (Hz)')
        axs[0].grid(True)

        # Setear opciones del plot de disipación
        axs[1].set_xlabel('n')
        axs[1].set_ylabel(f'variación de Delta_D{self._disipUnitLabel}')
        axs[1].set_xticks(list(self.armonicos))
        axs[1].set_xticklabels(list(self.armonicos))
        axs[1].grid(True)

        
        # Ajustar los subplots
        fig.tight_layout()

        # Guardar la figura
        self._saveFigFn(fig, saveFig, '_bilayerVariations')

        # Mostrar la figura
        plt.show(block = False)

        return self
    
    def analyzePostBilayer(self, intervals: dict[str, list[int|float]], disipAxisLims = None, addValuesToPlot: bool = False, saveFig: bool | None = None, saveResult: bool | None = None):
        """
        Analiza y representa los datos de los plateaus correspondientes a los experimentos realizados sobre la bicapa ya formada.
        
        Argumentos:
        - intervals (dict): Etiqueta e intervalo de tiempo de cada plateau.
        - disipAxisLims(list o None): Lista o equivalente de longitud 2 con los límites de eje Y del plot de disipación.
        - addValuesToPlot (bool): Control de colocar los valores de los plateaus sobre la gráfica.
        - saveFig (bool o None): Control de guardado de figuras.
        - saveResult (bool o None): Control de guardado de ficheros.
        """
        # Buscar el inicio y el final de tiempo de todos los intervalos
        startTime = np.inf
        endTime = - np.inf
        for interval in intervals.values():
            startTime = min(startTime, *interval)
            endTime = max(endTime, *interval)

        # Filtrar los datos de la bicapa
        dfPost = self.df[(self.df[self.timeColName] >= startTime) & (self.df[self.timeColName] <= endTime)].copy()

        # Representar la formación de la bicapa
        fig, axs = plt.subplots(2, 1, sharex = 'col')
        fig.suptitle(f'{self.fileName} - Plateaus')

        for n in self.armonicos:

            frecColName = f'Delta_F/n_n={n:d}_(Hz)'
            disipColName = f'Delta_D_n={n:d}_()'

            # Plot de frecuencia vs time
            dfPost.plot(x = self.timeColName, y = frecColName, ax = axs[0], label = f'n = {n:d}')

            # # Plot de disipación vs time
            dfPost.plot(x = self.timeColName, y = disipColName, ax = axs[1], label = f'n = {n:d}')

        # Añadir etiquetas de los valores de los plateaus
        if addValuesToPlot:
            for interval in intervals.values():
                self._addTextLabels(dfPost, interval, axs)

        # Setear opciones del plot de frecuencias
        axs[0].set_xlabel(f'Time_({self.timeUnits})')
        axs[0].set_ylabel('Delta_F/n (Hz)')
        axs[0].yaxis.set_major_locator(MultipleLocator(20))
        axs[0].legend(bbox_to_anchor = (1, 1))
        axs[0].grid(True)
        # Setear opciones del plot de disipaciones
        axs[1].set_xlabel(f'Time_({self.timeUnits})')
        axs[1].set_ylabel(f'Delta_D{self._disipUnitLabel}')
        if disipAxisLims is not None:
            axs[1].set_ylim(disipAxisLims)
        axs[1].legend(bbox_to_anchor = (1, 1))
        axs[1].grid(True)

        # Ajustar los subplots
        fig.tight_layout()

        # Guardar la figura
        self._saveFigFn(fig, saveFig, '_plateaus')

        # Mostrar la figura
        plt.show(block = False)

        # Calcular y representar la media y sd de cada plateau
        results = pd.DataFrame(
            columns = [
                'label', 'armonico', 'timePoint',
                'frecCenter', 'frecSd',
                'disipCenter', 'disipSd'])

        # Crear figura
        fig, axs = plt.subplots(2, len(intervals), sharex = False, sharey = False)
        fig.suptitle(f'{self.fileName} - Plateaus Gaussian Fit')

        for i, (label, interval) in enumerate(intervals.items()):
            
            # Filtrar los datos del plateau
            dfFiltered = self.df[(self.df[self.timeColName] >= interval[0]) & (self.df[self.timeColName] <= interval[1])]
            meanTime = np.mean(interval)

            # Calcular la media y sd de cada armónico para el plateau
            for n in self.armonicos:

                frecColName = f'Delta_F/n_n={n:d}_(Hz)'
                disipColName = f'Delta_D_n={n:d}_()'

                # Obtener la media y la sd
                frecResult = fitAndPlotGaussian(dfFiltered[frecColName].values, axs[0, i], f'n = {n:d}', restrictCenter = True)
                disipResult = fitAndPlotGaussian(dfFiltered[disipColName].values, axs[1, i], f'n = {n:d}', restrictCenter = True)
                
                # Añadir los resultados al DataFrame
                results.loc[results.shape[0]] = {
                    'label': label, 'armonico': n, 'timePoint': meanTime,
                    'frecCenter': frecResult['center'], 'frecSd': frecResult['sigma'],
                    'disipCenter': disipResult['center'], 'disipSd': disipResult['sigma']}
            
            # Setear opciones de los plots de frecuencia
            axs[0, i].set_xlabel('Delta_F/n (Hz)')
            axs[0, i].set_ylabel('Frecuencia acumulada')
            axs[0, i].set_title(label)
            axs[0, i].grid(True)
            # Setear opciones de los plots de disipación
            axs[1, i].set_xlabel(f'Delta_D{self._disipUnitLabel}')
            axs[1, i].set_ylabel('Frecuencia acumulada')
            axs[1, i].set_title(label)
            axs[1, i].grid(True)
        
        # Añadir la leyenda únicamente en la última columna de los subplots
        axs[0, -1].legend(bbox_to_anchor = (1, 1))
        axs[1, -1].legend(bbox_to_anchor = (1, 1))

        # Ajustar los subplots
        fig.tight_layout()

        # Guardar la figura
        self._saveFigFn(fig, saveFig, '_plateausGaussians')

        # Mostrar la figura
        plt.show(block = False)

        # Almacenar los datos en la clase
        self.plateausResults = results

        # Escribir resultados en txt
        self._saveResultFn(results, saveResult, '_plateausGaussians')

        # Representar las medias de los plateaus vs tiempo
        fig, axs = plt.subplots(2, 1, sharex = True, sharey = False)
        fig.suptitle(f'{self.fileName} - Plateaus Vs Time')

        for n in self.armonicos:
            auxDf = results[results['armonico'] == n].copy()
            auxDf.sort_values('timePoint', inplace = True)

            # Representar con barras de error
            axs[0].errorbar(auxDf['timePoint'], auxDf['frecCenter'], yerr = auxDf['frecSd'], label = f'n = {n:d}')
            axs[1].errorbar(auxDf['timePoint'], auxDf['disipCenter'], yerr = auxDf['disipSd'], label = f'n = {n:d}')

        # Setear opciones del plot de frecuencia
        axs[0].set_xlabel(f'Time_({self.timeUnits})')
        axs[0].set_ylabel('Delta_F/n (Hz)')
        axs[0].grid(True)
        axs[0].legend(bbox_to_anchor = (1, 1))
        # Setear opciones del plot de disipación
        axs[1].set_xlabel(f'Time_({self.timeUnits})')
        axs[1].set_ylabel(f'Delta_D{self._disipUnitLabel}')
        if disipAxisLims is not None:
            axs[1].set_ylim(disipAxisLims)
        axs[1].grid(True)
        axs[1].legend(bbox_to_anchor = (1, 1))

        # Ajustar los subplots
        fig.tight_layout()

        # Guardar la figura
        self._saveFigFn(fig, saveFig, '_plateausVsTime')

        # Mostrar la figura
        plt.show(block = False)

    def plotVariationVsHarmonic(self, intervalDict: dict[str, list[list[int|float]]], saveFig: bool | None = None):
        """
        Calcula y representa las variaciones frente a los armónicos.
        
        Argumentos:
        - intervals (dict): Etiqueta y lista con intervalos de tiempo de inicio y final de cada variación.
        - saveFig (bool o None): Control de guardado de figuras.
        """
        # DataFrame para guardar resultados
        gaussianResults = pd.DataFrame(
            columns = [
                'intervalName', 'armonico',
                'frecStartCenter', 'frecStartSd', 'frecEndCenter', 'frecEndSd',
                'disipStartCenter', 'disipStartSd', 'disipEndCenter', 'disipEndSd',
                'frecVariation', 'disipVariation'])
        
        # Para cada variación
        for intervalName, intervals in intervalDict.items():

            # Filtrar los datos de la bicapa
            intervalDf = self.df[(self.df[self.timeColName] >= intervals[0][0]) & (self.df[self.timeColName] <= intervals[1][1])].copy()
            # Filtrar datos de los plateaus de inicio y final
            startDf = intervalDf[(intervalDf[self.timeColName] >= intervals[0][0]) & (intervalDf[self.timeColName] <= intervals[0][1])]
            endDf = intervalDf[(intervalDf[self.timeColName] >= intervals[1][0]) & (intervalDf[self.timeColName] <= intervals[1][1])]

            # Crear la figura
            fig, axs = plt.subplots(2, 2, sharex = False, sharey = False)
            fig.suptitle(f'{self.fileName} - Variation {intervalName} Gaussian Fit')

            # Calcular y representar gaussianas
            for n in self.armonicos:

                frecColName = f'Delta_F/n_n={n:d}_(Hz)'
                disipColName = f'Delta_D_n={n:d}_()'

                # Frecuencia
                frecStartResult = fitAndPlotGaussian(startDf[frecColName].values, axs[0,0], f'n = {n:d}')
                frecEndResult = fitAndPlotGaussian(endDf[frecColName].values, axs[0,1], f'n = {n:d}')

                # Disipación
                disipStartResult = fitAndPlotGaussian(startDf[disipColName].values, axs[1,0], f'n = {n:d}')
                disipEndResult = fitAndPlotGaussian(endDf[disipColName].values, axs[1,1], f'n = {n:d}')

                # Calcular las variaciones de frecuencia y disipación
                gaussianResults['frecVariation'] = gaussianResults['frecEndCenter'] - gaussianResults['frecStartCenter']
                gaussianResults['disipVariation'] = gaussianResults['disipEndCenter'] - gaussianResults['disipStartCenter']

                # Añadir los resultados al DataFrame
                gaussianResults.loc[gaussianResults.shape[0]] = {
                    'intervalName': intervalName, 'armonico': n,
                    'frecStartCenter': frecStartResult['center'], 'frecStartSd': frecStartResult['sigma'],
                    'frecEndCenter': frecEndResult['center'], 'frecEndSd': frecEndResult['sigma'],
                    'disipStartCenter': disipStartResult['center'], 'disipStartSd': disipStartResult['sigma'],
                    'disipEndCenter': disipEndResult['center'], 'disipEndSd': disipEndResult['sigma'],
                    'frecVariation': frecEndResult['center'] - frecStartResult['center'], 'disipVariation': disipEndResult['center'] -  disipStartResult['center']}
            
            # Setear propiedades de los ejes
            axs[0,0].set_xlabel('Delta_F/n (Hz)')
            axs[0,0].set_ylabel('Frecuencia acumulada')
            axs[0,0].set_title(f'Equilibrio inicial ({intervals[0][0]}-{intervals[0][1]})')
            axs[0,0].grid(True)
            axs[0,1].set_xlabel('Delta_F/n (Hz)')
            axs[0,1].set_ylabel('Frecuencia acumulada')
            axs[0,1].set_title(f'Equilibrio final ({intervals[1][0]}-{intervals[1][1]})')
            axs[0,1].legend(bbox_to_anchor = (1, 1))
            axs[0,1].grid(True)
            axs[1,0].set_xlabel(f'Delta_D{self._disipUnitLabel}')
            axs[1,0].set_ylabel('Frecuencia acumulada')
            axs[1,0].set_title(f'Equilibrio inicial ({intervals[0][0]}-{intervals[0][1]})')
            axs[1,0].grid(True)
            axs[1,1].set_xlabel(f'Delta_D{self._disipUnitLabel}')
            axs[1,1].set_ylabel('Frecuencia acumulada')
            axs[1,1].set_title(f'Equilibrio final ({intervals[1][0]}-{intervals[1][1]})')
            axs[1,1].legend(bbox_to_anchor = (1, 1))
            axs[1,1].grid(True)

            # Ajustar los subplots
            fig.tight_layout()

            # Mostrar la figura
            plt.show(block = False)

        # Representar variaciones frente a armonicos
        fig, axs =  plt.subplots(2, 1, sharex = 'col')
        fig.suptitle(f'{self.fileName} - Variations')

        for intervalName in gaussianResults['intervalName'].unique():

            # Filtrar por el intervalo
            auxDf = gaussianResults[gaussianResults['intervalName'] == intervalName]

            axs[0].plot(auxDf['armonico'], auxDf['frecVariation'], '.-', label = intervalName)
            axs[1].plot(auxDf['armonico'], auxDf['disipVariation'], '.-', label = intervalName)

        # Setear opciones del plot de frecuencia
        axs[0].set_xlabel('n')
        axs[0].set_ylabel('variación de Delta_F/n (Hz)')
        axs[0].grid(True)
        axs[0].legend(loc = (1.03, 0.5))

        # Setear opciones del plot de disipación
        axs[1].set_xlabel('n')
        axs[1].set_ylabel(f'variación de Delta_D{self._disipUnitLabel}')
        axs[1].set_xticks(list(self.armonicos))
        axs[1].set_xticklabels(list(self.armonicos))
        axs[1].grid(True)
        axs[1].legend(loc = (1.03, 0.5))

        # Ajustar los subplots
        fig.tight_layout()

        # Guardar la figura
        self._saveFigFn(fig, saveFig, '_Variations')

        # Mostrar la figura
        plt.show(block = False)

    def _addTextLabels(self, data: pd.DataFrame, interval, axs: list[matplotlib.axes.Axes]):
        # Calcular posición media en tiempo
        timeMean = np.mean(interval)

        auxDf = data[(data[self.timeColName] >= interval[0]) & (data[self.timeColName] <= interval[1])]

        # Calcular posición media en frecuencia y disipación para todos los armónicos
        frecuencyArray = np.empty((1, len(self.armonicos)))
        disipationArray = np.empty((1, len(self.armonicos)))

        for n in self.armonicos:

            frecColName = f'Delta_F/n_n={n:d}_(Hz)'
            disipColName = f'Delta_D_n={n:d}_()'

            frecuencyArray = np.append(frecuencyArray, auxDf[frecColName].values)
            disipationArray = np.append(disipationArray, auxDf[disipColName].values)

        frecMean = np.mean(frecuencyArray)
        frecSd = np.std(frecuencyArray)
        disipMean = np.mean(disipationArray)
        disipSd = np.std(disipationArray)

        axs[0].text(timeMean, frecMean + 2, f'{frecMean:.2f} ± {frecSd:.2f}', fontsize = 10, color = 'black')
        axs[1].text(timeMean, disipMean + 1, f'{disipMean:.2f} ± {disipSd:.2f}', fontsize = 10, color = 'black')


    def _saveFigFn(self, fig: matplotlib.figure.Figure, saveFig: bool | None = None, label: str = ''):
        """
        Guarda las figuras en un fichero de formato definido en la propiedad `saveFormat`.
        
        Argumentos:
        - fig (plt.figure): Figura a guardar.
        - saveFig (bool o None): Control de guardado. En caso de `None`, sigue la configuración global.
        - label (str): Texto a añadir en el nombre del fichero.
        """
        if saveFig is None:
            saveFig = self._saveFigs

        if saveFig:
            if not os.path.exists(self._figDir):
                os.makedirs(self._figDir)
            figName, _ = os.path.splitext(self.fileName) # Quitar extensión .txt
            figName = f'{figName}{label}.{self._saveFormat}'
            figPath = os.path.join(self._figDir, figName)
            fig.savefig(figPath)
    
    def _saveResultFn(self, data: pd.DataFrame, saveResult: bool | None = None, label: str = ''):
        """
        Guarda los resultados en un fichero txt.
        
        Argumentos:
        - data (pd.DataFrame): Datos a guardar.
        - saveResult (bool o None): Control de guardado. En caso de `None`, sigue la configuración global.
        - label (str): Texto a añadir en el nombre del fichero.
        """
        if saveResult is None:
            saveResult = self._saveResults

        if saveResult:
            if not os.path.exists(self._resultDir):
                os.makedirs(self._resultDir)
            resultName, _ = os.path.splitext(self.fileName)
            resultName = f'{resultName}{label}.txt'
            resultPath = os.path.join(self._resultDir, resultName)
            data.to_csv(resultPath, sep = '\t', index = False, float_format = '%.4E')


def fitAndPlotGaussian(values, ax: matplotlib.axes.Axes, plotLabel: str = '', restrictCenter: bool = True, doFit = True):
    """
    Ajusta los valores a una gaussiana y los representa.
    
    Argumentos:
    - values (array): valores a ajustar.
    - ax (plt.Axes): ejes sobre los que representar.
    - plotLabel (str): Etiqueta de los valores.
    - restrictCenter (bool): Control de la restricción del centroide de la gaussiana alrededor de la media de los datos.
    """
    if doFit:
        # Obtener frecuencias de valores (histograma)
        frecuency, binEdges = np.histogram(values, bins = 20)
        binCenters = (binEdges[:-1] + binEdges[1:]) / 2

        # Fit a gaussiana
        model = GaussianModel()

        # Obtener parámetros iniciales aproximados
        params = model.guess(frecuency, x = binCenters)
        if restrictCenter:
            # Restringir el centro de la gaussiana cerca de la media de los valores
            params['center'].min = 0.9 * np.mean(values)
            params['center'].max = 1.1 * np.mean(values)

        result = model.fit(frecuency, params, x = binCenters) # Hace el fit

        lineObj = ax.plot(binCenters, frecuency, '.', label = plotLabel)
        pltColor = lineObj[-1].get_color()
        xLinSpace = np.linspace(binCenters.min(), binCenters.max(), 200)
        ax.plot(xLinSpace, result.eval(x = xLinSpace), '-', c = pltColor)

        results = result.best_values
    else:
        results = {}
        results['center'] = np.mean(values)
        results['sigma'] = np.std(values)

    return results

def plotCombinedExperiments(experimentList: list[Experiment], disipAxisLims = None):
    """
    Representa las medias de los plateaus de varios experimentos frente al tiempo agrupadas por armónico.

    Argumentos:
    - experimentList (list): lista de instancias de la clase Experiment.
    - disipAxisLims(list o None): Lista o equivalente de longitud 2 con los límites de eje Y del plot de disipación.
    """

    # Comprobar que se ha realizado el análisis de los plateaus
    for exp in experimentList:
        if not hasattr(exp, 'plateausResults'):
            raise Exception(f'Plateaus analysis is previously required. Please, run it for Experiment {exp.fileName}.')
        
    # Comprobar que todas los experimentos tienen las mismas unidades de tiempo y disipación
    timeUnits = set()
    disipUnits = set()
    for exp in experimentList:
        timeUnits.add(exp.timeUnits)
        disipUnits.add(exp._disipUnitLabel)
    if len(timeUnits) > 1:
        raise Exception(f'Experiments must have the same time units. Different units found: {timeUnits}.')
    if len(disipUnits) > 1:
        raise Exception(f'Experiments must have the same disipation units. Different units found: {disipUnits}.')
        
    # Combinar los armónicos de todos los experimentos
    armonicos = set()
    for exp in experimentList:
        armonicos = armonicos.union(exp.armonicos)

    # Representar las medias de los plateaus vs tiempo
    fig, axs = plt.subplots(len(armonicos), 2, sharex = False, sharey = False, figsize = (10, 3 * len(armonicos)))
    fig.suptitle(f'Plateaus Vs Time')

    # Para cada armónico
    for idx, n in enumerate(armonicos):

        # Y para cada experimento
        for exp in experimentList:
            
            # Sacar los resultados del experimento
            data = exp.plateausResults

            # Comprobar que el experimento tiene el armónico
            if not n in data.armonico:
                continue
            
            # Filtrar el armónico
            auxDf = data[data['armonico'] == n].copy()
            auxDf.sort_values('timePoint', inplace = True)

            # Representar medias vs tiempo con barras de error
            axs[idx, 0].errorbar(auxDf['timePoint'], auxDf['frecCenter'], yerr = auxDf['frecSd'], label = exp.fileName)
            axs[idx, 1].errorbar(auxDf['timePoint'], auxDf['disipCenter'], yerr = auxDf['disipSd'], label = exp.fileName)

        # Setear opciones del plot de frecuencia
        axs[idx, 0].set_title(f'n = {n}')
        axs[idx, 0].set_xlabel(f'Time_({experimentList[0].timeUnits})')
        axs[idx, 0].set_ylabel('Delta_F/n (Hz)')
        axs[idx, 0].grid(True)
        # Setear opciones del plot de disipación
        axs[idx, 1].set_title(f'n = {n}')
        axs[idx, 1].set_xlabel(f'Time_({experimentList[0].timeUnits})')
        axs[idx, 1].set_ylabel(f'Delta_D{experimentList[0]._disipUnitLabel}')
        if disipAxisLims is not None:
            axs[idx, 1].set_ylim(disipAxisLims)
        axs[idx, 1].grid(True)
        axs[idx, 1].legend(loc = (1.03, 0.5))

    # Ajustar los subplots
    fig.tight_layout()

    # Mostrar la figura
    plt.show(block = False)

def plotCombinedExperimentPlateaus(experimentDict: dict[str, list[Experiment]], disipAxisLims = None):
    """
    Representa las medias de los plateaus de varios experimentos frente al tiempo agrupadas por armónico.

    Argumentos:
    - experimentDict (dict): diccionarios de listas de instancias de la clase Experiment.
    - disipAxisLims(list o None): Lista o equivalente de longitud 2 con los límites de eje Y del plot de disipación.
    """

    # Funciones auxiliares
    def calculateMeanAndSd(data:pd.DataFrame):
        
        # Calcular media
        results = data[['label', 'frecCenter', 'disipCenter']].groupby(['label'], as_index = False).mean()
        
        # Calcular sd propagada (media cuadrática de las sd's)
        sdDf = data.groupby(['label'], as_index = False)[['frecSd', 'disipSd']].apply(lambda x: np.sqrt((x ** 2).sum()) / len(x))
        # Calcular sd estadística de los valores
        if data.shape[0] > 1:
            sdDf[['frecSdStat', 'disipSdStat']] = data.groupby(['label'], as_index = False)[['frecCenter', 'disipCenter']].std().rename(columns = {'frecCenter': 'frecSdStat', 'disipCenter': 'disipSdStat'})[['frecSdStat', 'disipSdStat']]
        else:
            sdDf[['frecSdStat', 'disipSdStat']] = [0, 0]
        # Combinar sd propagada y estadística
        sdDf['frecSd'] = np.sqrt(sdDf['frecSd']**2 + sdDf['frecSdStat']**2)
        sdDf['disipSd'] = np.sqrt(sdDf['disipSd']**2 + sdDf['disipSdStat']**2)
        
        # Añadir sd a los resultados
        results[['frecSd', 'disipSd']] = sdDf[['frecSd', 'disipSd']]

        return results
    
    # Crear diccionario de resultados
    results = {}
    
    # Comprobar que se ha realizado el análisis de los plateaus
    for experimentList in experimentDict.values():
        for exp in experimentList:
            if not hasattr(exp, 'plateausResults'):
                raise Exception(f'Plateaus analysis is previously required. Please, run it for Experiment {exp.fileName}.')
        
    # Comprobar que todos los experimentos tienen las mismas unidades de tiempo y disipación
    timeUnits = set()
    disipUnits = set()
    for experimentList in experimentDict.values():
        for exp in experimentList:
            timeUnits.add(exp.timeUnits)
            disipUnits.add(exp._disipUnitLabel)
    if len(timeUnits) > 1:
        raise Exception(f'Experiments must have the same time units. Different units found: {timeUnits}.')
    if len(disipUnits) > 1:
        raise Exception(f'Experiments must have the same disipation units. Different units found: {disipUnits}.')
    
    # Crear df para guardar los resultados de todos los grupos
    combinedResults = pd.DataFrame()

    # Procesar cada grupo de experimentos
    for groupName, experimentList in experimentDict.items():

        # Crear df para almacenar todos los resultados del grupo
        groupDf = pd.DataFrame(columns = ['fileName', 'label', 'frecCenter', 'frecSd', 'disipCenter', 'disipSd'])

        # Iterar sobre los experimentos
        for exp in experimentList:

            data = exp.plateausResults.copy()
            # Ordenar cronológicamente
            data.sort_values('timePoint', inplace = True)
            # Promediar todos los armónicos para cada plateau (cada label) y suma cuadrática de sd
            auxDf = calculateMeanAndSd(data)
            # Añadir nombre del fichero
            auxDf['fileName'] = exp.fileName
            
            # Añadir al df del grupo
            if groupDf.shape[0] < 1:
                groupDf = auxDf
            else:
                groupDf = pd.concat([groupDf, auxDf], ignore_index = True)

        # Promediar todos los experimentos para cada plateau (cada label) y suma cuadrática de sd
        groupResults = calculateMeanAndSd(groupDf)
        # Añadir nombre del grupo
        groupResults['grupo'] = groupName

        # Añadir al df de resultados
        if combinedResults.shape[0] < 1:
            combinedResults = groupResults
        else:
            combinedResults = pd.concat([combinedResults, groupResults], ignore_index = True)

        # Crear variables de frecuencia y disipación (mean ± sd)
        groupDf['Delta_F/n (Hz)'] = groupDf['frecCenter'].map('{:.4f}'.format) + ' ± ' + groupDf['frecSd'].map('{:.4f}'.format)
        disipColName = f'Delta_D{list(disipUnits)[0]}' # Añade las unidades de disipación
        groupDf[disipColName] = groupDf['disipCenter'].map('{:.4f}'.format) + ' ± ' + groupDf['disipSd'].map('{:.4f}'.format)
        
        # Extender frecuencia y disipación como variable y valor
        groupDf = pd.melt(groupDf, id_vars = ['label', 'fileName'], value_vars = ['Delta_F/n (Hz)', disipColName], var_name = 'variable')
        
        # Renombrar columnas
        groupDf.rename(columns = {'label': 'etiqueta'}, inplace = True)
        # Reordenar df
        groupDf = groupDf.pivot_table(index = ['variable', 'etiqueta'], columns = 'fileName', values = 'value', aggfunc = lambda x: ' '.join(x))

        # Añadir a resultados
        results[groupName] = groupDf


    # Crear variables de frecuencia y disipación (mean ± sd)
    combinedResults['Delta_F/n (Hz)'] = combinedResults['frecCenter'].map('{:.4f}'.format) + ' ± ' + combinedResults['frecSd'].map('{:.4f}'.format)
    disipColName = f'Delta_D{list(disipUnits)[0]}' # Añade las unidades de disipación
    combinedResults[disipColName] = combinedResults['disipCenter'].map('{:.4f}'.format) + ' ± ' + combinedResults['disipSd'].map('{:.4f}'.format)
    
    # Extender frecuencia y disipación como variable y valor
    combinedResults = pd.melt(combinedResults, id_vars = ['label', 'grupo'], value_vars = ['Delta_F/n (Hz)', disipColName], var_name = 'variable')
    
    # Renombrar columnas
    combinedResults.rename(columns = {'label': 'etiqueta'}, inplace = True)
    # Reordenar df
    combinedResults = combinedResults.pivot_table(index = ['variable', 'etiqueta'], columns = 'grupo', values = 'value', aggfunc = lambda x: ' '.join(x))

    # Añadir a resultados
    results['allGroups'] = combinedResults

    # Representar cada grupo de experimentos
    for groupName, experimentList in experimentDict.items():
        # Combinar los armónicos de todos los experimentos
        armonicos = set()
        for exp in experimentList:
            armonicos = armonicos.union(exp.armonicos)

        # Representar las medias de los plateaus vs tiempo
        fig, axs = plt.subplots(len(armonicos), 2, sharex = False, sharey = False, figsize = (10, 3 * len(armonicos)))
        fig.suptitle(f'{groupName} - Plateaus Vs Time')

        # Para cada armónico
        for idx, n in enumerate(armonicos):

            # Y para cada experimento
            for exp in experimentList:
                
                # Sacar los resultados del experimento
                data = exp.plateausResults

                # Comprobar que el experimento tiene el armónico
                if not n in data['armonico'].unique():
                    continue
                
                # Filtrar el armónico
                auxDf = data[data['armonico'] == n].copy()

                # Ordenar cronológicamente
                auxDf.sort_values('timePoint', inplace = True)

                # Representar medias vs tiempo con barras de error
                axs[idx, 0].errorbar(auxDf['timePoint'], auxDf['frecCenter'], yerr = auxDf['frecSd'], label = exp.fileName)
                axs[idx, 1].errorbar(auxDf['timePoint'], auxDf['disipCenter'], yerr = auxDf['disipSd'], label = exp.fileName)

            # Setear opciones del plot de frecuencia
            axs[idx, 0].set_title(f'n = {n}')
            axs[idx, 0].set_xlabel(f'Time_({experimentList[0].timeUnits})')
            axs[idx, 0].set_ylabel('Delta_F/n (Hz)')
            axs[idx, 0].grid(True)
            # Setear opciones del plot de disipación
            axs[idx, 1].set_title(f'n = {n}')
            axs[idx, 1].set_xlabel(f'Time_({experimentList[0].timeUnits})')
            axs[idx, 1].set_ylabel(f'Delta_D{experimentList[0]._disipUnitLabel}')
            if disipAxisLims is not None:
                axs[idx, 1].set_ylim(disipAxisLims)
            axs[idx, 1].grid(True)
            axs[idx, 1].legend(loc = (1.03, 0.5))

        # Ajustar los subplots
        fig.tight_layout(rect = [0, 0.02, 1, 0.98])

    # Mostrar la figura
    plt.show(block = False)

    return results

def plotCombinedExperimentBilayers(experimentDict: dict[str, list[Experiment]], disipAxisLims = None):
    """
    Representa las medias de los plateaus de varios experimentos frente al tiempo agrupadas por armónico.

    Argumentos:
    - experimentDict (dict): diccionarios de listas de instancias de la clase Experiment.
    - disipAxisLims(list o None): Lista o equivalente de longitud 2 con los límites de eje Y del plot de disipación.
    """

    # Funciones auxiliares
    def calculateMeanAndSd(data:pd.DataFrame):
        
        # Calcular media
        results = data[['frecStartCenter', 'frecEndCenter', 'disipStartCenter', 'disipEndCenter']].mean().to_frame().transpose()
        
        # Calcular sd propagada (media cuadrática de las sd's)
        sdDf = data[['frecStartSd', 'frecEndSd', 'disipStartSd', 'disipEndSd']].apply(lambda x: np.sqrt((x ** 2).sum()) / len(x)).to_frame().transpose()
        # Calcular sd estadística de los valores
        if data.shape[0] > 1:
            sdDf[['frecStartSdStat', 'frecEndSdStat', 'disipStartSdStat', 'disipEndSdStat']] = data[['frecStartCenter', 'frecEndCenter', 'disipStartCenter', 'disipEndCenter']].std().rename({'frecStartCenter': 'frecStartSdStat', 'frecEndCenter': 'frecEndSdStat', 'disipStartCenter': 'disipStartSdStat', 'disipEndCenter': 'disipEndSdStat'}).to_frame().transpose()
        else:
            sdDf[['frecStartSdStat', 'frecEndSdStat', 'disipStartSdStat', 'disipEndSdStat']] = [0, 0, 0, 0]
            # Combinar sd propagada y estadística
        sdDf['frecStartSd'] = np.sqrt(sdDf['frecStartSd']**2 + sdDf['frecStartSdStat']**2)
        sdDf['frecEndSd'] = np.sqrt(sdDf['frecEndSd']**2 + sdDf['frecEndSdStat']**2)
        sdDf['disipStartSd'] = np.sqrt(sdDf['disipStartSd']**2 + sdDf['disipStartSdStat']**2)
        sdDf['disipEndSd'] = np.sqrt(sdDf['disipEndSd']**2 + sdDf['disipEndSdStat']**2)
        
        # Añadir sd a los resultados
        results[['frecStartSd', 'frecEndSd', 'disipStartSd', 'disipEndSd']] = sdDf[['frecStartSd', 'frecEndSd', 'disipStartSd', 'disipEndSd']]

        return results
    
    # Crear diccionario de resultados
    results = {}
    
    # Comprobar que se ha realizado el análisis de la bicapa
    for experimentList in experimentDict.values():
        for exp in experimentList:
            if not hasattr(exp, 'bilayerResults'):
                raise Exception(f'Bilayer analysis is previously required. Please, run it for Experiment {exp.fileName}.')
        
    # Comprobar que todos los experimentos tienen las mismas unidades de tiempo y disipación
    timeUnits = set()
    disipUnits = set()
    for experimentList in experimentDict.values():
        for exp in experimentList:
            timeUnits.add(exp.timeUnits)
            disipUnits.add(exp._disipUnitLabel)
    if len(timeUnits) > 1:
        raise Exception(f'Experiments must have the same time units. Different units found: {timeUnits}.')
    if len(disipUnits) > 1:
        raise Exception(f'Experiments must have the same disipation units. Different units found: {disipUnits}.')
    
    # Crear df para guardar los resultados de todos los grupos
    combinedResults = pd.DataFrame()

    # Procesar cada grupo de experimentos
    for groupName, experimentList in experimentDict.items():

        # Crear df para almacenar todos los resultados del grupo
        groupDf = pd.DataFrame(columns = ['fileName', 'frecCenter', 'frecSd', 'disipCenter', 'disipSd'])

        # Iterar sobre los experimentos
        for exp in experimentList:

            data = exp.bilayerResults.copy()
            # Promediar todos los armónicos para cada punto (comienzo y final) y suma cuadrática de sd
            auxDf = calculateMeanAndSd(data)
            # Añadir nombre del fichero
            auxDf['fileName'] = exp.fileName
            
            # Añadir al df del grupo
            if groupDf.shape[0] < 1:
                groupDf = auxDf
            else:
                groupDf = pd.concat([groupDf, auxDf], ignore_index = True)

        # Promediar todos los experimentos para cada punto (comienzo y final) y suma cuadrática de sd
        groupResults = calculateMeanAndSd(groupDf)

        # Añadir nombre del grupo
        groupResults['grupo'] = groupName

        # Añadir al df de resultados
        if combinedResults.shape[0] < 1:
            combinedResults = groupResults
        else:
            combinedResults = pd.concat([combinedResults, groupResults], ignore_index = True)

        # Formatear resultados del grupo
        groupDf = pd.melt(groupDf, id_vars = ['fileName'], value_vars = ['frecStartCenter', 'frecEndCenter', 'disipStartCenter', 'disipEndCenter', 'frecStartSd', 'frecEndSd', 'disipStartSd', 'disipEndSd'])
        groupDf['etiqueta'] = groupDf['variable'].str.extract(r'(Start|End)')
        groupDf['stat'] = groupDf['variable'].str.extract(r'(Center|Sd)')
        groupDf['variable'] = groupDf['variable'].str.extract(r'(frec|disip)')

        # Recombinar frec/disip y center/sd
        groupDf = groupDf.pivot_table(index = ['fileName', 'etiqueta'], columns = ['variable', 'stat'], values = 'value', aggfunc = 'first').reset_index()
        groupDf.columns = [''.join(col) if col[1] != '' else col[0] for col in groupDf.columns.values]

        # Crear variables de frecuencia y disipación (mean ± sd)
        groupDf['Delta_F/n (Hz)'] = groupDf['frecCenter'].map('{:.4f}'.format) + ' ± ' + groupDf['frecSd'].map('{:.4f}'.format)
        disipColName = f'Delta_D{list(disipUnits)[0]}' # Añade las unidades de disipación
        groupDf[disipColName] = groupDf['disipCenter'].map('{:.4f}'.format) + ' ± ' + groupDf['disipSd'].map('{:.4f}'.format)
        
        # Ordenar por etiqueta (start -> end)
        groupDf.sort_values(by = 'etiqueta', ascending = False, inplace = True)

        # Extender frecuencia y disipación como variable y valor
        groupDf = pd.melt(groupDf, id_vars = ['etiqueta', 'fileName'], value_vars = ['Delta_F/n (Hz)', disipColName], var_name = 'variable')
        
        # Reordenar df
        groupDf = groupDf.pivot_table(index = ['variable', 'etiqueta'], columns = 'fileName', values = 'value', aggfunc = lambda x: ' '.join(x), sort = False)

        # Almacenar en resultados
        results[groupName] = groupDf


    # Extender frec/disip, start/end, center/sd como variables adicionales
    combinedResults = pd.melt(combinedResults, id_vars = ['grupo'], value_vars = ['frecStartCenter', 'frecEndCenter', 'disipStartCenter', 'disipEndCenter', 'frecStartSd', 'frecEndSd', 'disipStartSd', 'disipEndSd'])
    combinedResults['etiqueta'] = combinedResults['variable'].str.extract(r'(Start|End)')
    combinedResults['stat'] = combinedResults['variable'].str.extract(r'(Center|Sd)')
    combinedResults['variable'] = combinedResults['variable'].str.extract(r'(frec|disip)')

    # Recombinar frec/disip y center/sd
    combinedResults = combinedResults.pivot_table(index = ['grupo', 'etiqueta'], columns = ['variable', 'stat'], values = 'value', aggfunc = 'first').reset_index()
    combinedResults.columns = [''.join(col) if col[1] != '' else col[0] for col in combinedResults.columns.values]

    # Crear variables de frecuencia y disipación (mean ± sd)
    combinedResults['Delta_F/n (Hz)'] = combinedResults['frecCenter'].map('{:.4f}'.format) + ' ± ' + combinedResults['frecSd'].map('{:.4f}'.format)
    disipColName = f'Delta_D{list(disipUnits)[0]}' # Añade las unidades de disipación
    combinedResults[disipColName] = combinedResults['disipCenter'].map('{:.4f}'.format) + ' ± ' + combinedResults['disipSd'].map('{:.4f}'.format)
    
    # Ordenar por etiqueta (start -> end)
    combinedResults.sort_values(by = 'etiqueta', ascending = False, inplace = True)

    # Extender frecuencia y disipación como variable y valor
    combinedResults = pd.melt(combinedResults, id_vars = ['etiqueta', 'grupo'], value_vars = ['Delta_F/n (Hz)', disipColName], var_name = 'variable')
    
    # Reordenar df
    combinedResults = combinedResults.pivot_table(index = ['variable', 'etiqueta'], columns = 'grupo', values = 'value', aggfunc = lambda x: ' '.join(x), sort = False)

    # Añadir a resultados
    results['allGroups'] = combinedResults

    # Representar cada grupo de experimentos
    for groupName, experimentList in experimentDict.items():
        # Combinar los armónicos de todos los experimentos
        armonicos = set()
        for exp in experimentList:
            armonicos = armonicos.union(exp.armonicos)

        # Representar las medias de los plateaus vs tiempo
        fig, axs = plt.subplots(len(armonicos), 2, sharex = False, sharey = False, figsize = (10, 3 * len(armonicos)))
        fig.suptitle(f'{groupName} - Bilayer Vs Time')

        # Para cada armónico
        for idx, n in enumerate(armonicos):

            # Y para cada experimento
            for exp in experimentList:
                
                # Sacar los resultados del experimento
                data = exp.bilayerResults.copy()

                # Comprobar que el experimento tiene el armónico
                if not n in data['armonico'].unique():
                    continue
                
                # Filtrar el armónico
                auxDf = data[data['armonico'] == n].copy()

                # Reordenar la información para el plot
                plotDf = pd.DataFrame({
                    'timePoint': [auxDf['meanStartTime'].values[0], auxDf['meanEndTime'].values[0]],
                    'frecCenter': [auxDf['frecStartCenter'].values[0], auxDf['frecEndCenter'].values[0]],
                    'disipCenter': [auxDf['disipStartCenter'].values[0], auxDf['disipEndCenter'].values[0]],
                    'frecSd': [auxDf['frecStartSd'].values[0], auxDf['frecEndSd'].values[0]],
                    'disipSd': [auxDf['disipStartSd'].values[0], auxDf['disipEndSd'].values[0]]})

                # Representar medias vs tiempo con barras de error
                axs[idx, 0].errorbar(plotDf['timePoint'], plotDf['frecCenter'], yerr = plotDf['frecSd'], label = exp.fileName)
                axs[idx, 1].errorbar(plotDf['timePoint'], plotDf['disipCenter'], yerr = plotDf['disipSd'], label = exp.fileName)

            # Setear opciones del plot de frecuencia
            axs[idx, 0].set_title(f'n = {n}')
            axs[idx, 0].set_xlabel(f'Time_({experimentList[0].timeUnits})')
            axs[idx, 0].set_ylabel('Delta_F/n (Hz)')
            axs[idx, 0].grid(True)
            # Setear opciones del plot de disipación
            axs[idx, 1].set_title(f'n = {n}')
            axs[idx, 1].set_xlabel(f'Time_({experimentList[0].timeUnits})')
            axs[idx, 1].set_ylabel(f'Delta_D{experimentList[0]._disipUnitLabel}')
            if disipAxisLims is not None:
                axs[idx, 1].set_ylim(disipAxisLims)
            axs[idx, 1].grid(True)
            axs[idx, 1].legend(loc = (1.03, 0.5))

        # Ajustar los subplots
        fig.tight_layout(rect = [0, 0.02, 1, 0.98])

    # Mostrar la figura
    plt.show(block = False)

    return results
