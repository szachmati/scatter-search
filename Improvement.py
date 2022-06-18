import copy
import numpy as np

from utils import calculatePathCost


class Improvement:
    def twoOpt(self, distanceMatrix: np.array, pathWithCost: list) -> list:
        """ Funkcja 2-opt zamieniająca dwie krawędzie innymi krawędziami w celu utworzenia nowego cyklu
            i próby zmiejszenia kosztu ścieżki
            :parameter distanceMatrix - macierz sąsiedztwa
            :parameter pathWithCost - ścieżka z obecnym kosztem
            :return bestPath - ulepszona ścieżka z poprawionym kosztem
         """
        bestPath = copy.deepcopy(pathWithCost)
        pathLength = len(bestPath[0])
        currentPath = copy.deepcopy(bestPath)
        resetPathVariable = copy.deepcopy(bestPath)
        for i in range(0, pathLength - 2):
            for j in range(i + 1, pathLength - 1):
                currentPath[0] = self.twoOptSwap(bestPath[0], i, j)
                currentPath[1] = calculatePathCost(distanceMatrix, currentPath[0])
                if currentPath[1] < bestPath[1]:
                    for k in range(pathLength):
                        """ przekopiowanie pozycji miast z ulepszonej drogi """
                        bestPath[0][k] = currentPath[0][k]
                    bestPath[1] = currentPath[1]
                currentPath = copy.deepcopy(resetPathVariable)
        # print(f"pathWithCost: {pathWithCost}\nbestPath: {bestPath}")
        return bestPath

    def twoOptSwap(self, path: list, i: int, j: int) -> list:
        """ Funkcja zamieniająca kolejność elementów w tablicy od indeksu i do j """
        swapped = copy.deepcopy(path)
        swapped[i: j + 1] = list(reversed(swapped[i: j + 1]))
        swapped[-1] = swapped[0]
        return swapped