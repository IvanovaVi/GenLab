// pch.h: это предварительно скомпилированный заголовочный файл.
// Перечисленные ниже файлы компилируются только один раз, что ускоряет последующие сборки.
// Это также влияет на работу IntelliSense, включая многие функции просмотра и завершения кода.
// Однако изменение любого из приведенных здесь файлов между операциями сборки приведет к повторной компиляции всех(!) этих файлов.
// Не добавляйте сюда файлы, которые планируете часто изменять, так как в этом случае выигрыша в производительности не будет.


#pragma once

#ifdef GenApp_H
#define GenApp_API __declspec(dllexport)
#else
#define GenApp_API __declspec(dllimport)
#endif

#include "framework.h"

extern "C" GenApp_API void crossover(
    const std::vector<std::vector<int>>&population, int numSelected, int numParents, int numOffspring);

#ifndef GenApp_H
#define GenApp_H

#include <vector>
#include <string>

std::vector<std::vector<int>> crossover1(const std::vector<std::vector<int>>& population, int numSelected, int numParents, int numOffspring);
void mutate(std::vector<int>& genes);
std::vector<std::vector<int>> createNewPopulation(std::vector<std::vector<int>>& population, int popSize, int n, int threshold);
std::vector<std::vector<int>> genitor(int population_size, int num_generations, double mutation_rate);
std::vector<double> genetic_algorithm(int population_size, double min, double max, int num_generations, double mutation_rate, int num_parents, int num_children);
double eggHolderEvaluation(const std::vector<double>& position);
void geneticAlgorithm(std::vector<char>& population, int numGenerations);

#endif // GenApp_H