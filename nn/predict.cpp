#include <iostream>
#include <string>
#include "array.h"
#include "network.h"
#pragma once

int main(int argv, char** argc){

    if (argv > 2) {
        std::string model_name = argc[1];

//        std::cout << model_name << '\n';

        network net(model_name);

        std::string str_X = argc[2];

        std::stringstream ss_x(str_X);

        std::string val{};

        array X;

        while (std::getline(ss_x, val, ',')){
//            std::cout << val << ' ';
            X.push_back(std::stof(val) / 255.F);
        }

        std::cout << argmax(net.predict(X));

        return 0;
    }

    std::string model_name = R"(C:\\Users\\Dmitry\\CLionProjects\\untitled\\models\\model_full_256_128_64.txt)" ;

    network net(model_name);

    while (true) {
        array input(784);
        for (int i = 0; i < 784; ++i) {
            std::cin >> input[i];
            input[i] = input[i] / 255.F;
        }

        std::cout << argmax(net.predict(input)) << '\n';

    }
    return 0;
}