import itertools
from time import time
import copy
import numpy as np
from typing import List

from matplotlib import pyplot as plt
from DistanceMatrixGenerator import DistanceMatrixGenerator
from DiversificationGenerator import DiversificationGenerator
from SolutionCombinationMethod import SolutionCombinationMethod
from utils import calculatePathCost


def diversificationGeneratorForSeedPermutations(
        DG: DiversificationGenerator,
        distancesMatrix: np.array,
        startElement: int,
        n: int) -> list:
    """ tworzenie inicjalnych rozwiązań za pomocą Generatora Dywersyfikującego na podstawie ziarna
        :parameter DG - generator dywersyfikujący
        :parameter distancesMatrix - macierz sąsiedztwa miast
        :parameter startElement - element od którego tworzone są permutacje np. 5
        :parameter n - ilość rozwiązań do wygenerowania przez DG
        :return diverseTrialSolutions
     """
    seed = []
    [seed.append(i) for i in range(1, distancesMatrix.shape[0] + 1)]
    return DG.generate(seed, startElement, n)


def diversificationGeneratorForRandomPaths(DG: DiversificationGenerator, distancesMatrix: np.array, n: int) -> list:
    """ tworzenie inicjalnych, losowych rozwiązań
        :parameter DG - generator dywersyfikujący
        :parameter distancesMatrix - macierz sąsiedztwa miast
        :parameter n - ilość rozwiązań do wygenerowania
        :return diverseTrialSolutions - inicjalne rozwiązania
    """
    print("*** Diversification Generator :: start ***")
    diverseTrialSolutions = []
    for i in range(0, n):
        diverseTrialSolutions.append(DG.generate2(distancesMatrix))
    print(f"*** Diversification Generator :: end - utworzono {len(diverseTrialSolutions)} rozwiązań ***")
    return diverseTrialSolutions

def twoOptLocalImprovement(distanceMatrix: np.array, pathWithCost: list) -> list:
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
            currentPath[0] = twoOptSwap(bestPath[0], i, j)
            currentPath[1] = calculatePathCost(distanceMatrix, currentPath[0])
            if currentPath[1] < bestPath[1]:
                for k in range(pathLength):
                    """ przekopiowanie pozycji miast z ulepszonej drogi """
                    bestPath[0][k] = currentPath[0][k]
                bestPath[1] = currentPath[1]
            currentPath = copy.deepcopy(resetPathVariable)
    # print(f"pathWithCost: {pathWithCost}\nbestPath: {bestPath}")
    return bestPath


def twoOptSwap(path: list, i: int, j: int) -> list:
    """ Funkcja zamieniająca kolejność elementów w tablicy od indeksu i do j """
    swapped = copy.deepcopy(path)
    swapped[i: j + 1] = list(reversed(swapped[i: j + 1]))
    swapped[-1] = swapped[0]
    return swapped


def scatterSearch(
        temporaryRefSet: list,
        distancesMatrix: np.array,
        iterations=50,
        reverseProb=0.5,
        scrambleProb=0.3) -> tuple:
    """ Implementacja algorytmu przeszukiwania rozporoszonego
        :parameter temporaryRefSet - początkowy zbiór RefSet wygenerowany w fazie inicjującej
        :parameter distancesMatrix - macierz sąsiedztwa
        :parameter iterations - liczba iteracji przeszukiwania
        :parameter reverseProb - operator krzyżowania odwracający kolejność odwiedzin grupy miast, domyślnie 0.5
        :parameter scrambleProb - operator krzyżowania mieszający kolejność odwiedzin grupy miast, domyślnie 0.3
        :returns bestPathWithCost - najlepsza ścieżka i jej koszt , costs - koszt per iteracja, totalTime - łączny czas przeszukiwania
     """
    print("*** Scatter Search :: start ***")
    RefSet = copy.deepcopy(temporaryRefSet)
    b = len(RefSet)
    bestPathWithCost = RefSet[0]
    startTime = time()

    costs = []
    index = 0
    scm = SolutionCombinationMethod()
    for i in range(iterations):
        C = []
        """ 1. Krzyżowanie i poprawa rozwiązania algorytmem 2-opt """
        for j in range(b):
            C.append(scm.crossover(distancesMatrix, RefSet, reverseProb, scrambleProb))
            C[j] = twoOptLocalImprovement(distancesMatrix, pathWithCost=C[j])

        """ 2. Uzupełnienie zbioru RefSet elementami tablicy C """
        [RefSet.append(C[i]) for i in range(b)]

        """ 3. Sortowanie zbioru RefSet i wybór b najlepszych ścieżek """
        RefSet.sort(key=lambda x: x[1])
        RefSet = RefSet[:b]

        """ 4. Aktualizacja najlepszej ścieżki o najnizszym koszcie """
        if RefSet[0][1] < bestPathWithCost[1]:
            bestPathWithCost = RefSet[0]
        index += 1
        costs.append(RefSet[0][1])
        bestPathWithCost[1] = np.around(bestPathWithCost[1], 2)
        print(f"i = {i + 1}/{iterations}, koszt = {bestPathWithCost[1]}, trasa = {bestPathWithCost[0][0:5]},...,{bestPathWithCost[0][len(bestPathWithCost) - 5:]}")
    totalTime = time() - startTime
    print(f"*** Scatter Search :: end - łączny czas przeszukiwania - {totalTime} sekund ***")
    return bestPathWithCost, costs, totalTime


def initialPhase(n: int, b: int, startElement: int) -> List[list]:
    """ Faza inicjująca algorytmu przeszukiwania rozporoszonego
        :parameter n - liczba początkowych rozwiązań utworzonych przez DG
        :parameter b - liczba rozwiązań zbioru RefSet
        :parameter startElement - element od którego zaczynają się permutacje wykonane na ziarnie np. 5 dla wersji z generatorem permutującym
        :return RefSet - b najlepszych początkowych rozwiązań
    """
    if n < b:
        raise ValueError("b nie może być większe od n!")
    print("*** Faza inicjująca ***")
    RefSet = []
    DG = DiversificationGenerator()
    startTime = time()
    while len(RefSet) < b:
        """ 1.1. Wybieramy 1 z 2 generatorów dywersyfikujących - tworzenie początkowych rozwiązań """
        # diverseTrialSolutions = diversificationGeneratorForRandomPaths(DG, distancesMatrix, n)
        diverseTrialSolutions = diversificationGeneratorForSeedPermutations(DG, distancesMatrix, startElement, n)

        enhancedSolutions = []
        for i in range(len(diverseTrialSolutions)):
            path = diverseTrialSolutions[i]
            pathCost = calculatePathCost(distancesMatrix, path)
            """ 1.2. Poprawa inicjalnych rozwiązań """
            enhancedSolutions.append(twoOptLocalImprovement(distancesMatrix, [path, pathCost]))

        """ 1.3. Uzupełnienie zbioru RefSet ulepszonymi ścieżkami"""
        print(f"Poprawiono {len(enhancedSolutions)} rozwiązań wstępnych")
        [RefSet.append(enhanced) for enhanced in enhancedSolutions]

        if len(RefSet) >= b:
            RefSet.sort(key=lambda x: x[1])
            RefSet = RefSet[:b]

    totalTime = np.around(time() - startTime, 2)
    print(f"RefSet został wypełniony {len(RefSet)} rozwiązaniami - koniec fazy inicjującej")
    print(f"Czas generowania {b} rozwiązań - {totalTime} sek.")
    return RefSet


if __name__ == '__main__':
    """ zaladowananie problemu i obliczenie macierzy sąsiedztwa(odległości) pomiędzy miastami """
    matrixGenerator = DistanceMatrixGenerator(fileName="datasets/kroC100.tsp")
    points = list(matrixGenerator.problem.node_coords.values())
    distancesMatrix = np.array(matrixGenerator.createDistanceMatrix())

    """ Rozpoczęcie fazy inicjującej """
    RefSet = initialPhase(n=20, b=10, startElement=5)

    """ Rozpoczęcie fazy przeszukiwania rozproszonego """
    bestPath, costs, totalTime = scatterSearch(RefSet, distancesMatrix, iterations=50)
    print(f"""Wyniki:\nDroga: {bestPath[0]}\nKoszt: {bestPath[1]}\nCzas podróży: {np.around(totalTime, 2)} sekund""")

    # """ Przedstawienie wyników w formie wykresów """
    # drawCostsPlot()
    # drawResultPath(points, result)
