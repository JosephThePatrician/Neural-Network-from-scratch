#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <random>
#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include "array.h"
#include "network.h"

void readcsv(
        std::string filename,
        std::vector<array>& xtrain, std::vector<int>& ytrain,
        std::vector<array>& xtest, std::vector<int>& ytest,
        unsigned int length = 42000, double val_percent = 0.2){

    std::string line, val;
    unsigned int col_num{};

    std::ifstream trainFile(filename);

    std::getline(trainFile, line); // first line is just names

//    std::cout << line.length() << '\n';
//    std::cout << line << '\n';

    unsigned int train_shape = (length * (1 - val_percent));

    while (std::getline(trainFile, line, '\n')) {

        if (col_num < train_shape) {
            std::stringstream ss(line);

            std::getline(ss, val, ',');

            ytrain.push_back(std::stoi(val));

            xtrain.emplace_back(784);

            int i = 0;
            while (std::getline(ss, val, ',')) {
                xtrain[col_num][i] = (((float) std::stoi(val)) / 255);
                i++;
            }
        }

        else{
            std::stringstream ss(line);

            std::getline(ss, val, ',');

            ytest.push_back(std::stoi(val));

            xtest.emplace_back(784);

            int i = 0;
            while (std::getline(ss, val, ',')) {
                xtest[col_num - train_shape][i] = (((float) std::stoi(val)) / 255);
                i++;
            }
        }

        col_num++;
    }

    trainFile.close();
}

int main(){

    // read train.csv
    std::cout << "reading train.cvs" << '\n';
    std::vector<array> xtrain;
    std::vector<int> ytrain;

    std::vector<array> xtest;
    std::vector<int> ytest;

    readcsv(R"(C:\Users\Dmitry\CLionProjects\untitled\data\train.csv)",
            xtrain, ytrain, xtest, ytest);

    std::cout << "Train examples: " << xtrain.size() << "\nValidation examples: " << xtest.size() << '\n';

    std::vector<layer> layers = {
            layer(28 * 28, 16),
            layer(256, 128),
            layer(128, 64),
            layer(16, 10)
    };

    network net(layers);

    // hyperparameters
    unsigned int epochs = 5;

    float accuracy{};
    float loss{};

    // train loop
    for (int e = 0; e < epochs; ++e){

        std::cout << "------------------" << "epoch " << e << "------------------" << '\n';

        accuracy = 0;
        loss = 0;
        for (int i = 0; i < xtrain.size(); ++i){

            array predicted = net.predict(xtrain[i]);

            accuracy += (argmax(predicted) == ytrain[i]);

            array true_answers(10);
            true_answers[ytrain[i]] = 1;
            net.backpropagation(predicted, true_answers);

            loss += ((true_answers - predicted) * (true_answers - predicted)).sum();

            if ((i + 1) % 1000 == 0) {
//                for (int k = 0; k < predicted.size(); ++k)
//                    std::cout << true_answers[k] << ' ' << predicted[k] << ' ';
//                std::cout << '\n';
                std::cout << "step " << i + 1 << " accuracy " << accuracy / i << " loss " << loss / i << '\n';
            }
        }

        std::cout << "accuracy on train epoch " << e << " is " << accuracy / xtrain.size() << '\n';

        accuracy = 0;

        for (int i = 0; i < xtest.size(); ++i){
            array predicted = net.predict(xtest[i]);

            accuracy += (argmax(predicted) == ytest[i]);

            if ((i + 1) % 1000 == 0)
                std::cout << "step " << i + 1 << " accuracy " << accuracy / i << '\n';
        }

        std::cout << "accuracy on validation epoch " << e << " is " << accuracy / xtest.size() << '\n';
    }

    std::cout << "model has been trained and saved" << '\n';


    std::string filename = R"(C:\\Users\\Dmitry\\CLionProjects\\untitled\\models\\test.txt)";

    net.save(filename);

    return 0;
}