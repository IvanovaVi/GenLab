// dllmain.cpp : Определяет точку входа для приложения DLL.

#include <stdlib.h>
#include <random>
#include<vector>
#include<utility>
#include <string>
#include <algorithm>
#include "GenApp.h"

//определеяем популяцию сами в приложении
// Оператор 1-точечного кроссинговера
std::vector<std::vector<int>> crossover1(const std::vector<std::vector<int>>& population, int numSelected, int numParents, int numOffspring) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, population[0].size() - 1);

    std::vector<std::vector<int>> offspring;
    for (int i = 0; i < numOffspring; i++) {
        int parent1 = i % numParents;
        int parent2 = (i + 1) % numParents;
        std::vector<int> parent1Genes = population[parent1];
        std::vector<int> parent2Genes = population[parent2];
        std::vector<int> childGenes = parent1Genes;
        int crossoverPoint = dis(gen);
        for (int j = crossoverPoint; j < parent2Genes.size(); j++) {
            childGenes[j] = parent2Genes[j];
        }
        offspring.push_back(childGenes);
    }
    return offspring;
}

//Функция crossover принимает следующие аргументы :
//const std::vector<std::vector<int>>& population : Вектор population, содержащий несколько векторов целых чисел, представляющих индивидов population.
//int numSelected : Число выбранных индивидов из population.
//int numParents : Число родительских индивидов из population.
//int numOffspring : Число поколений, которые будут созданы на основе population.
//Внутри функции выполняется следующее :
//Создается случайное число rd с помощью std::random_device.
//Инициализируется генератор псевдослучайных чисел gen с помощью rd.
//Создается вектор offspring, который будет содержать векторы целых чисел, представляющие поколения.
//Для каждого индивида(i) из поколения offspring :
//Вычисляются индексы родительских индивидов parent1 и parent2 на основе i и numParents.
//Берется вектор генетических информации родительского индивида parent1 из population.
//Берется вектор генетической информации родительского индивида parent2 из population.
//Создается вектор генетической информации дочернего индивида childGenes, который начинается с генетической информации parent1 и заканчивается с генетической информацией parent2.
//Вычисляется индекс перекрещивания(crossoverPoint), используя функцию dis и генератор псевдослучайных чисел gen.
//Добавляется дочерний индивид childGenes в вектор offspring.
//В результате функция crossover возвращает вектор offspring, содержащий векторы целых чисел, представляющие поколения, созданные на основе population.

//Оператор мутации по данному кроссинговеру (используется, как основной оператор мутации)
void mutate(std::vector<int>& genes) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, genes.size() - 1);
    int mutationPoint = dis(gen);
    genes[mutationPoint] = 1 - genes[mutationPoint]; // Инвертируем бит
}

//функция mutate, которая изменяет вектор genes, представляющий гены, случайным образом инвертируя один из их битов
//std::random_device rd создается объект random_device, который используется для генерации случайных чисел.
//std::mt19937 gen(rd()) создается генератор псевдослучайных чисел Mersenne Twister с начальным значением, полученным из random_device.
//std::uniform_int_distribution<int> dis(0, genes.size() - 1) создается объект uniform_int_distribution, который будет использоваться для генерации равномерно распределенных целых чисел в заданном диапазоне.Диапазон определяется от 0 до genes.size() - 1, так как мы хотим выбирать индексы элементов вектора genes.
//int mutationPoint = dis(gen) используется объект uniform_int_distribution для генерации случайной точки(mutationPoint), которая представляет собой индекс гена, подлежащего мутации.
//genes[mutationPoint] = 1 - genes[mutationPoint] выбранный ген инвертируется путем замены 0 на 1 и 1 на 0.
//Таким образом, функция mutate использует случайное число для выбора индекса гена, который затем инвертируется, моделируя процесс мутации генетического алгоритма.


//Для CHC-алгоритма на нахождение максимального значения (OneMax)
// Функция для вычисления количества единиц в бинарной строке
int countOnes(std::vector<int>& v) {
    int count = 0;
    for (int i = 0; i < v.size(); i++) {
        if (v[i] == 1) {
            count++;
        }
    }
    return count;
}

// Функция для генерации случайной бинарной строки
std::vector<int> generateRandomVector(int n) {
    std::vector<int> v(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);
    for (int i = 0; i < n; i++) {
        v[i] = dis(gen);
    }
    return v;
}

// Функция для сравнения двух бинарных строк по количеству единиц
bool compareVectors(std::vector<int>& v1, std::vector<int>& v2) {
    return countOnes(v1) > countOnes(v2);
}

// Функция для создания новой популяции на основе текущей
std::vector<std::vector<int>> createNewPopulation(std::vector<std::vector<int>>& population, int popSize, int n, int threshold) {
    std::vector<std::vector<int>> newPopulation;
    sort(population.begin(), population.end(), compareVectors);
    newPopulation.push_back(population[0]);
    for (int i = 1; i < popSize; i++) {
        bool found = false;
        for (int j = 0; j < newPopulation.size(); j++) {
            int hammingDistance = 0;
            for (int k = 0; k < n; k++) {
                if (population[i][k] != newPopulation[j][k]) {
                    hammingDistance++;
                }
            }
            if (hammingDistance < threshold) {
                found = true;
                break;
            }
        }
        if (!found) {
            newPopulation.push_back(population[i]);
        }
    }
    while (newPopulation.size() < popSize) {
        newPopulation.push_back(generateRandomVector(n));
    }
    return newPopulation;
}


//Алгоритм Genitor для нахождения максимального значения (OneMax)
// Фитнес-функция для проверки популяции
int fitness(std::vector<int>& individual) {
    int sum = 0;
    for (int i = 0; i < individual.size(); i++) {
        sum += individual[i];
    }
    return sum;
}

// Локальный оператор мутации
void mutate(std::vector<int>& individual, double mutation_rate) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < individual.size(); i++) {
        if (dis(gen) < mutation_rate) {
            individual[i] = 1 - individual[i];
        }
    }
}

// Локальный оператор кроссовера
std::vector<int> crossover2(std::vector<int>& parent1, std::vector<int>& parent2) {
    std::vector<int> child(parent1.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, parent1.size() - 1);
    int crossover_point = dis(gen);
    for (int i = 0; i < crossover_point; i++) {
        child[i] = parent1[i];
    }
    for (int i = crossover_point; i < child.size(); i++) {
        child[i] = parent2[i];
    }
    return child;
}

// Алгоритм Genitor
std::vector<std::vector<int>> genitor(int population_size, int num_generations, double mutation_rate) {
    // Инициализация популяции
    std::vector<std::vector<int>> population(population_size, std::vector<int>(100));
    for (int i = 0; i < population_size; i++) {
        for (int j = 0; j < 100; j++) {
            population[i][j] = rand() % 2;
        }
    }
    // Проверка фитнес-функцией
    std::vector<int> fitness_values(population_size);
    for (int i = 0; i < population_size; i++) {
        fitness_values[i] = fitness(population[i]);
    }
    // Сортируем популяцию по фитнес-функции
    std::vector<int> indices(population_size);
    for (int i = 0; i < population_size; i++) {
        indices[i] = i;
    }
    sort(indices.begin(), indices.end(), [&](int i, int j) { return fitness_values[i] > fitness_values[j]; });
    
    for (int generation = 0; generation < num_generations; generation++) {
        // Выбираем родителей
        std::vector<std::vector<int>> parents(population_size / 2, std::vector<int>(100));
        for (int i = 0; i < population_size / 2; i++) {
            int parent1_index = indices[rand() % (population_size / 2)];
            int parent2_index = indices[rand() % (population_size / 2)];
            parents[i] = crossover2(population[parent1_index], population[parent2_index]);
        }
        // Мутация получаемой популяции
        for (int i = 0; i < population_size / 2; i++) {
            mutate(parents[i], mutation_rate);
        }
        // Проверка фитнес-функцией получаемой популяции
        std::vector<int> offspring_fitness_values(population_size / 2);
        for (int i = 0; i < population_size / 2; i++) {
            offspring_fitness_values[i] = fitness(parents[i]);
        }
        // Заменя худших особей популяции на те, что получаем
        for (int i = 0; i < population_size / 2; i++) {
            population[indices[i]] = parents[i];
            fitness_values[indices[i]] = offspring_fitness_values[i];
        }
        sort(indices.begin(), indices.end(), [&](int i, int j) { return fitness_values[i] > fitness_values[j]; });
    }
    return population;
}


//Алгоритм с использованием функции тестирования Химмельбрау
double fitness(double x, double y) {
    return pow(pow(x, 2) + y - 11, 2) + pow(x + pow(y, 2) - 7, 2);
}

double random_double(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

std::vector<double> random_chromosome(double min, double max) {
    std::vector<double> chromosome;
    chromosome.push_back(random_double(min, max));
    chromosome.push_back(random_double(min, max));
    return chromosome;
}

std::vector<std::vector<double>> initialize_population(int population_size, double min, double max) {
    std::vector<std::vector<double>> population;
    for (int i = 0; i < population_size; i++) {
        population.push_back(random_chromosome(min, max));
    }
    return population;
}

std::vector<double> evaluate_population(std::vector<std::vector<double>> population) {
    std::vector<double> fitness_values;
    for (int i = 0; i < population.size(); i++) {
        fitness_values.push_back(fitness(population[i][0], population[i][1]));
    }
    return fitness_values;
}

std::vector<std::vector<double>> select_parents(std::vector<std::vector<double>> population, std::vector<double> fitness_values, int num_parents) {
    std::vector<std::vector<double>> parents;
    for (int i = 0; i < num_parents; i++) {
        int max_index = max_element(fitness_values.begin(), fitness_values.end()) - fitness_values.begin();
        parents.push_back(population[max_index]);
        fitness_values[max_index] = -1;
    }
    return parents;
}

std::vector<double> crossover3(std::vector<double> parent1, std::vector<double> parent2) {
    std::vector<double> child;
    child.push_back((parent1[0] + parent2[0]) / 2);
    child.push_back((parent1[1] + parent2[1]) / 2);
    return child;
}

std::vector<std::vector<double>> breed_population(std::vector<std::vector<double>> parents, int num_children) {
    std::vector<std::vector<double>> children;
    for (int i = 0; i < num_children; i++) {
        int parent1_index = i % parents.size();
        int parent2_index = (i + 1) % parents.size();  //
        children.push_back(crossover3(parents[parent1_index], parents[parent2_index]));
    }
    return children;
}

std::vector<std::vector<double>> mutate_population(std::vector<std::vector<double>> population, double mutation_rate, double min, double max) {
    for (int i = 0; i < population.size(); i++) {
        if (random_double(0, 1) < mutation_rate) {
            population[i][0] = random_double(min, max);
        }
        if (random_double(0, 1) < mutation_rate) {
            population[i][1] = random_double(min, max);
        }
    }
    return population;
}

std::vector<double> genetic_algorithm(int population_size, double min, double max, int num_generations, double mutation_rate, int num_parents, int num_children) {
    std::vector<std::vector<double>> population = initialize_population(population_size, min, max);
    for (int i = 0; i < num_generations; i++) {
        std::vector<double> fitness_values = evaluate_population(population);
        std::vector<std::vector<double>> parents = select_parents(population, fitness_values, num_parents);
        std::vector<std::vector<double>> children = breed_population(parents, num_children);
        population = mutate_population(children, mutation_rate, min, max);
    }
    std::vector<double> fitness_values = evaluate_population(population);
    int min_index = min_element(fitness_values.begin(), fitness_values.end()) - fitness_values.begin();
    return population[min_index];
}


// Eggholder 
double eggHolderEvaluation(const std::vector<double>& position) {
    double sum = 0;
    std::vector<double> modifiedPosition = position; //создаем копию для пропуска модификации вводимого вектора
    for (unsigned int i = 0; i < modifiedPosition.size(); ++i) {
        double temp = modifiedPosition[i];
        modifiedPosition[i] = std::max(0.0, temp + 47 * std::sin(i * 3.14159));
        sum += modifiedPosition[i] * modifiedPosition[i];
    }
    return sum;
}

// 
const int populationSize = 100;
const int maxIterations = 100;
double crossoverRate = 0.5;
double mutationRate = 0.01;

std::vector<double> generateRandomPosition() {
    std::vector<double> position(2);
    for (unsigned int i = 0; i < position.size(); ++i) {
        position[i] = -512 + 102 * std::rand() / (std::rand() + 1.0);
    }
    return position;
}

std::vector<double> crossover(const std::vector<double>& parent1, const std::vector<double>& parent2) {
    std::vector<double> child(2);
    for (unsigned int i = 0; i < child.size(); ++i) {
        child[i] = parent1[i] + (parent2[i] - parent1[i]) * crossoverRate;
    }
    return child;
}

std::vector<double> mutate(const std::vector<double>& position) {
    std::vector<double> mutatedPosition = position;
    for (unsigned int i = 0; i < mutatedPosition.size(); ++i) {
        mutatedPosition[i] += mutationRate * (std::rand() - 0.5);
    }
    return mutatedPosition;
}

double selection(const std::vector<double>& fitness) {
    double sum = 0;
    for (unsigned int i = 0; i < fitness.size(); ++i) {
        sum += fitness[i] / (1 + i);
    }
    return sum;
}


//Задача коммивояжера
// Функция для преобразования строки в список символов
std::vector<char> stringToChars(const std::string& str) {
    std::vector<char> chars;
    for (const auto& c : str) {
        chars.push_back(c);
    }
    return chars;
}

// Функция для оценки качества решения
int fitnessFunction(const std::vector<char>& individual) {
    // Здесь можно реализовать функцию оценки качества, например, количество совпадающих символов с коммивояжером
    // В данном примере мы используем простую функцию оценки качества
    int score = 0;
    for (const auto& c : individual) {
        if (c == 'a' || c == 'A') {
            score++;
        }
    }
    return score;
}

// Генетический алгоритм
void geneticAlgorithm(std::vector<char>& population, int numGenerations) {
    // Инициализируем population случайными символами
    std::random_device rd;
    std::mt19937 gen(rd());
    for (auto& ind : population) {
        std::generate(ind.begin(), ind.end(), [&gen]() {
            return 'a' + static_cast<char>((gen() % 26) + 97);
            });
    }

    // Основной цикл алгоритма
    for (int gen = 0; gen < numGenerations; ++gen) {
        // Выбор родителей
        std::vector<char> parents[2];
        for (int i = 0; i < 2; ++i) {
            std::generate(parents[i].begin(), parents[i].end(), [&gen, &population]() {
                return population[gen % population.size()];
                });
        }

        // Кроссовер
        std::vector<char> offspring[2];
        for (int i = 0; i < 2; ++i) {
            std::vector<char> child;
            for (int j = 0; j < population.size(); ++j) {
                child[j] = parents[0][j / 2] + (parents[1][j / 2] - parents[0][j / 2]) * static_cast<float>(gen) / numGenerations;
            }
            offspring[i] = child;
        }

        // Мутация
        for (int i = 0; i < 2; ++i) {
            std::generate(offspring[i].begin(), offspring[i].end(), [&gen, &population]() {
                return population[gen % population.size()] + static_cast<char>((gen() % 2) * 2 - 1);
                });
        }

        // Замена population на offspring
        population = offspring[0];

        // Оценка качества
        int bestFitness = fitnessFunction(offspring[0]);
        if (bestFitness > fitnessFunction(population[0])) {
            population = offspring[0];
        }
        else if (bestFitness > fitnessFunction(population[1])) {
            population = offspring[1];
        }
    }
}

