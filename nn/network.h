#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include "array.h"
#pragma once

class layer{
private:
    array neurons;
    array biases;
    matrix weights;
    array z;
    unsigned int n_in{}, n_out{};
public:
    layer(unsigned int n_in, unsigned int n_out) : n_in(n_in), n_out(n_out) {
        neurons.resize(n_in);

        biases.resize(n_out);
        biases.fill_random();

        weights.resize(n_out);
        for (auto &i : weights){
            i.resize(n_in);
            i.fill_random();
        }
    }

    layer(unsigned int n_in, unsigned int n_out, matrix weights, array biases) :
            n_in(n_in), n_out(n_out), weights(std::move(weights)), biases(std::move(biases)) {

        neurons.resize(n_in);
    }

    array forward(array neu){
//        assert(neu.size() == n_in);
        neurons = neu;
        z = dot(weights, neurons) + biases;
//        assert(z.size() == n_out);
        return z;
    }

    friend class network;
};


class network{
private:
    std::vector<layer> layers;
    float alpha = .001;

public:
    explicit network(std::vector<layer> &val, float alpha = 0.001) : layers(val), alpha(alpha){};

    explicit network(const std::string& filename){
        std::ifstream saved_model(filename);

        int n_in, n_out;
        std::string buffer;

        while (true) {
            // read sizes
            saved_model >> n_in;
            saved_model >> n_out;

            if (saved_model.eof()){break;}

            // read weights
            matrix weights(n_out);
            for (auto &w: weights) {
                w.resize(n_in);
                for (auto &i: w) {
                    saved_model >> i;
                }
            }

            // read biases
            array biases(n_out);
            for (auto &b: biases) {
                saved_model >> b;
            }

            // add layer
            layers.emplace_back(n_in, n_out, weights, biases);
        }
    };

    void set_alpha(float val){
        alpha = val;
    }

    array predict(array inp){
        // run full loop through layers and return predicted values

        array out = inp;

        for (int i = 0; i < layers.size() - 1; ++i){
            out = layers[i].forward(out);
            out = sigma(out); // activation function
        }

        // last layer
        out = layers[layers.size() - 1].forward(out);
        out = softmax(out);

        return out;
    }

    void backpropagation(const array& y){

        unsigned layers_size = layers.size();

        matrix da(layers.size());

        // de/dw
        std::vector<matrix> dw(layers_size);

        // de/db
        std::vector<array> db(layers_size);

        // neurons of all layers, except the last one
        std::vector<array> al(layers_size);
        for (int i = 0; i < layers_size; i++){
            al[i] = layers[i].neurons;
        }

        // z of all layers
        std::vector<array> zl(layers_size);
        for (int i = 0; i < layers_size; i++){
            zl[i] = layers[i].z;
        }

        // weights of all layers
        std::vector<matrix> wl(layers_size);
        for (int i = 0; i < layers_size; i++){
            wl[i] = layers[i].weights;
        }

        // for convenience
        layers_size--;

        // derivative of softmax for de/da of last layer
        da[layers_size] = zl[layers_size] - y;

        for (int curr = layers_size; curr > 0; curr--){
            auto val = da[curr] * dsigma(zl[curr]); // need it
            auto wT = wl[curr].T();
            da[curr-1] = dot(wT, val);
        }

        dw[layers_size] = dot(al[layers_size - 1], da[layers_size]);
        db[layers_size] = da[layers_size];

        for (int curr = layers_size - 1; curr >= 0; curr--){
            array dt = dsigma(zl[curr]) * da[curr];
            dw[curr] = dot(dt, al[curr]);
            db[curr] = dt;
        }

        for (int i = 0; i < layers.size(); ++i){
            layers[i].weights = layers[i].weights - dw[i] * alpha;
            layers[i].biases = layers[i].biases - db[i] * alpha;
        }
    }

    void backpropagation(array predicted, array& y){
        unsigned int layers_size = layers.size();

        matrix dt(layers.size()); // dE/dt
        matrix dh(layers.size()); // dE/dh

        std::vector<matrix> dw(layers_size); // dE/dw
        std::vector<array> db(layers_size); // dE/db

        // neurons of all layers, except the last one
        std::vector<array> h(layers_size);
        for (int i = 0; i < layers_size; i++){
            h[i] = layers[i].neurons;
        }

        // z of all layers (marked as t)
        std::vector<array> t(layers_size);
        for (int i = 0; i < layers_size; i++){
            t[i] = layers[i].z;
        }

        // weights of all layers
        std::vector<matrix> wl(layers_size);
        for (int i = 0; i < layers_size; i++){
            wl[i] = layers[i].weights.T();
        }

        // decrement for convenience
        layers_size--;

        // gradient of last layer
        dh[layers_size] = predicted;
        dt[layers_size] = predicted - y;
        dw[layers_size] = dot(h[layers_size], dt[layers_size]);
        db[layers_size] = dt[layers_size];

        // gradients of other layers
        for (int i = layers_size-1; i >= 0; --i){
            auto wT = wl[i + 1].T();
            dh[i] = dot(dt[i+1], wT);
            dt[i] = dh[i] * dsigma(t[i]);
            dw[i] = dot(h[i], dt[i]);
            db[i] = dh[i];
        }

        // update weights and biases
        for (int i = 0; i < layers.size(); ++i){
            layers[i].weights = layers[i].weights - dw[i].T() * alpha;
            layers[i].biases = layers[i].biases - db[i] * alpha;
        }

    }

    void save(const std::string& filename){

        // example of 1 layer being written:
        // n_in n_out\n
        // weights[0][0] weights[0][1] ... weights[0][n_in]\n
        // weights[1][0] weights[1][1] ... weights[1][n_in]\n
        // ...
        // weights[n_out][0] weights[n_out][1] ... weights[n_out][n_in]\n
        // bias[0] bias[1] ... bias[n_out]\n

        std::ofstream outfile (filename);

        for (auto& l : layers){
            // write sizes
            outfile << l.n_in << ' ' << l.n_out << '\n';

            // write weights
            for (auto& weight : l.weights){
                outfile << weight[0];
                for (int i = 1; i < weight.size(); ++i) {
                    outfile << ' ' << weight[i];
                }
                outfile << '\n';
            }

            // write biases
            outfile << l.biases[0];
            for (int i = 1; i < l.biases.size(); ++i) {
                outfile << ' ' << l.biases[i];
            }
            outfile << '\n';
        }
    }

//    void print(){
//        for (auto &i: layers) {
//            std::cout << i.get_n_in() << " -> " << i.get_n_out() << '\n';
//            std::cout << "----------------------------------------------------------\n";
//        }
//    }
};
