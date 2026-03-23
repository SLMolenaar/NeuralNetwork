#pragma once

#include "NeuralNetwork.h"

#include <SFML/Graphics.hpp>

class Visualizer {
public:
    Visualizer(const NeuralNetwork& network);

    void show();

private:
    const NeuralNetwork& network;

    static constexpr float windowWidth  = 900.f;
    static constexpr float windowHeight = 700.f;
    static constexpr float neuronRadius = 18.f;
    static constexpr float padding      = 60.f;

    sf::Vector2f neuronPosition(int layerIndex, int neuronIndex, int layerCount, int neuronsInLayer) const;

    void drawConnections(sf::RenderWindow& window) const;
    void drawNeurons(sf::RenderWindow& window, const sf::Font& font) const;
    void drawLabels(sf::RenderWindow& window, const sf::Font& font) const;
};
