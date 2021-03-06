import copy
import random

import numpy as np

from utils import calculatePathCost


class SolutionCombinationMethod:
    def crossover(self, distanceMatrix, RefSet, reverseFactor=0.5, scrambleFactor=0.3):
        """" Funkcja dokonująca krzyżowania dwóch losowych ścieżek
            :parameter distanceMatrix - macierz sąsiedztwa
            :parameter RefSet
            :parameter reverseFactor - współczynnik dla którego odwracane są kolejnością elementy 1 rodzica
            :parameter scrambleFactor - współczynnik dla którego mieszane są elementy 2 rodzica
        """

        """ 1. Selekcja dwóch losowo wybranych rodziców - 2 losowych ścieżek """
        refSetLength = len(RefSet)
        parent1Index, parent2Index = random.sample(list(range(0, refSetLength)), 2)
        """ 1.1. Usuwamy zakończenia ścieżek """
        parent1 = RefSet[parent1Index][0][:-1]
        parent2 = RefSet[parent2Index][0][:-1]
        citiesCount = copy.deepcopy(len(parent1) + 1)
        """ 1.2. Inicjalizacja ścieżki potomnej - początkowo wypełniamy ją zerami """
        offspring = list(np.zeros_like(parent1))

        """ 2. Wyznaczamy punkt cięcia - dwie liczby losowe, jeśli i > j, wtedy zamieniamy je wartościami """
        i, j = random.sample(list(range(0, len(parent1))), 2)
        if i > j:
            i, j = j, i
        """ 2.1. Jeśli współczynnik reverseProb jest większy od liczby losowej, to odwracamy kolejność elementów w rodzicu nr 1 """
        if random.random() < reverseFactor:
            parent1[i:j + 1] = list(reversed(parent1[i:j + 1]))
        """ 2.2. Uzupełniamy potomka w indeksach i:j+1 wartościami z 1 rodzica """
        offspring[i:j + 1] = parent1[i:j + 1]
        """ 2.3. W rodzicu nr 2 zostawiamy tylko miasta nieodwiedzone pomiędzy [i; j+1] w rodzicu nr 1"""
        parent2 = [x for x in parent2 if x not in parent1[i:j + 1]]
        """ 2.4. Jeśli próg scrambleProb większy od liczby losowej, dokonujemy mieszania elementów rodzica nr 2 """
        if random.random() < scrambleFactor:
            random.shuffle(parent2)

        """ 3. Brakujące pozycje - "0" uzupełniamy elementami z rodzica nr 2 """
        count = 0
        for i in range(len(offspring)):
            if offspring[i] == 0:
                offspring[i] = parent2[count]
                count = count + 1
        """ 4. Dopisujemy miasto startowe """
        if len(offspring) < citiesCount:
            offspring.append(offspring[0])
        return [offspring, calculatePathCost(distanceMatrix, offspring)]

    def singlePointCrossover(self, distanceMatrix: list, RefSet: list) -> list:
        """ Funkcja dokonująca krzyżowania 2 losowych ścieżek przy wykorzystaniu losowego punktu cięcia"""

        """ 1. Selekcja dwóch losowo wybranych rodziców - 2 losowych ścieżek """
        refSetLength = len(RefSet)
        parent1Index, parent2Index = random.sample(list(range(0, refSetLength)), 2)
        """ 1.1. Usuwamy zakończenia ścieżek """
        parent1 = RefSet[parent1Index][0][:-1]
        parent2 = RefSet[parent2Index][0][:-1]

        """ 2. Losowanie punktu cięcia """
        cutPoint = random.randint(0, len(parent1))
        """ 3. Wybieramy populację od-do z 1 i 2 rodzica
            Pierwszy chromosom - parent1[:cutPoint], resztę pozycji uzupełniamy elementami z 2 rodzica nie będącymi w 1 rodzicu
            Drugi chromosom - parent2[:cutPoint], resztę pozycji uzupełniamy elementami z 1 rodzica nie będącymi w 2 rodzicu
        """
        firstHalf = parent1[:cutPoint]
        firstHalfComplement = [x for x in parent2 if x not in parent1[:cutPoint]]
        secondHalf = parent2[:cutPoint]
        secondHalfComplement = [x for x in parent1 if x not in parent2[:cutPoint]]
        """ 3.1. łączenie ścieżek z 1 i 2 rodzica"""
        firstChromosome = list(np.append(firstHalf, firstHalfComplement))
        secondChromosome = list(np.append(secondHalf, secondHalfComplement))

        """ 4. Dodanie elementu początkowego """
        firstChromosome.append(firstChromosome[0])
        secondChromosome.append(secondChromosome[0])

        """ 5. Obliczenie kosztów ścieżek potomnych """
        firstChromosomeWithCost = [firstChromosome, calculatePathCost(distanceMatrix, firstChromosome)]
        secondChromosomeWithCost = [secondChromosome, calculatePathCost(distanceMatrix, secondChromosome)]
        return [firstChromosomeWithCost, secondChromosomeWithCost]