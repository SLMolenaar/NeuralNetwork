#include "Visualizer.h"

#include <string>

Visualizer::Visualizer(const NeuralNetwork& network)
    : network(network)
{
}

sf::Vector2f Visualizer::neuronPosition(int layerIndex, int neuronIndex, int layerCount, int neuronsInLayer) const
{
    float xSpacing = (windowWidth - padding * 2) / (layerCount - 1);
    float x = padding + layerIndex * xSpacing;

    float totalHeight = (neuronsInLayer - 1) * (neuronRadius * 3);
    float startY = (windowHeight - totalHeight) / 2.f;
    float y = startY + neuronIndex * (neuronRadius * 3);

    return { x, y };
}

void Visualizer::drawConnections(sf::RenderWindow& window) const
{
    const std::vector<int>& layerSizes = network.getLayerSizes();
    int layerCount = (int)layerSizes.size();

    for (int i = 0; i < layerCount - 1; i++) {
        for (int j = 0; j < layerSizes[i]; j++) {
            for (int k = 0; k < layerSizes[i + 1]; k++) {
                sf::Vector2f from = neuronPosition(i,     j, layerCount, layerSizes[i]);
                sf::Vector2f to   = neuronPosition(i + 1, k, layerCount, layerSizes[i + 1]);

                sf::Vertex line[2] = {
                    sf::Vertex{ from, sf::Color(180, 180, 180, 80) },
                    sf::Vertex{ to,   sf::Color(180, 180, 180, 80) }
                };

                window.draw(line, 2, sf::PrimitiveType::Lines);
            }
        }
    }
}

void Visualizer::drawNeurons(sf::RenderWindow& window, const sf::Font& font) const
{
    const std::vector<int>& layerSizes = network.getLayerSizes();
    int layerCount = (int)layerSizes.size();

    sf::CircleShape circle(neuronRadius);
    circle.setOrigin({ neuronRadius, neuronRadius });

    for (int i = 0; i < layerCount; i++) {
        for (int j = 0; j < layerSizes[i]; j++) {
            sf::Vector2f pos = neuronPosition(i, j, layerCount, layerSizes[i]);

            if (i == 0)
                circle.setFillColor(sf::Color(29, 158, 117));
            else if (i == layerCount - 1)
                circle.setFillColor(sf::Color(216, 90, 48));
            else
                circle.setFillColor(sf::Color(127, 119, 221));

            circle.setOutlineColor(sf::Color(255, 255, 255, 40));
            circle.setOutlineThickness(1.f);
            circle.setPosition(pos);
            window.draw(circle);

            sf::Text label(font, std::to_string(j + 1), 11);
            label.setFillColor(sf::Color::White);
            sf::FloatRect bounds = label.getLocalBounds();
            label.setOrigin({ bounds.position.x + bounds.size.x / 2.f,
                              bounds.position.y + bounds.size.y / 2.f });
            label.setPosition(pos);
            window.draw(label);
        }
    }
}

void Visualizer::drawLabels(sf::RenderWindow& window, const sf::Font& font) const
{
    const std::vector<int>& layerSizes = network.getLayerSizes();
    int layerCount = (int)layerSizes.size();

    for (int i = 0; i < layerCount; i++) {
        std::string name;
        if (i == 0)
            name = "Input";
        else if (i == layerCount - 1)
            name = "Output";
        else
            name = "Hidden " + std::to_string(i);

        sf::Text label(font, name, 13);
        label.setFillColor(sf::Color(200, 200, 200));

        sf::Vector2f topNeuron = neuronPosition(i, 0, layerCount, layerSizes[i]);
        sf::FloatRect bounds = label.getLocalBounds();
        label.setOrigin({ bounds.position.x + bounds.size.x / 2.f, 0.f });
        label.setPosition({ topNeuron.x, topNeuron.y - neuronRadius - 24.f });
        window.draw(label);

        sf::Text countLabel(font, std::to_string(layerSizes[i]) + " neurons", 11);
        countLabel.setFillColor(sf::Color(140, 140, 140));
        sf::FloatRect countBounds = countLabel.getLocalBounds();
        countLabel.setOrigin({ countBounds.position.x + countBounds.size.x / 2.f, 0.f });
        countLabel.setPosition({ topNeuron.x, topNeuron.y - neuronRadius - 10.f });
        window.draw(countLabel);
    }
}

void Visualizer::show()
{
    sf::RenderWindow window(
        sf::VideoMode({ (unsigned int)windowWidth, (unsigned int)windowHeight }),
        "Neural Network"
    );
    window.setFramerateLimit(60);

    sf::Font font;
    if (!font.openFromFile("C:/Windows/Fonts/arial.ttf")) {
        // Fall back to no text if font not found
    }

    // View for zoom and pan
    sf::View view(sf::FloatRect({ 0.f, 0.f }, { windowWidth, windowHeight }));
    window.setView(view);

    bool isPanning = false;
    sf::Vector2i lastMousePos;

    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();

            // Zoom with scroll wheel
            if (const auto* scrolled = event->getIf<sf::Event::MouseWheelScrolled>()) {
                float zoomFactor = (scrolled->delta > 0) ? 0.9f : 1.1f;
                view.zoom(zoomFactor);
                window.setView(view);
            }

            // Pan with right or middle mouse button
            if (const auto* mousePressed = event->getIf<sf::Event::MouseButtonPressed>()) {
                if (mousePressed->button == sf::Mouse::Button::Right ||
                    mousePressed->button == sf::Mouse::Button::Middle) {
                    isPanning = true;
                    lastMousePos = sf::Mouse::getPosition(window);
                }
            }

            if (const auto* mouseReleased = event->getIf<sf::Event::MouseButtonReleased>()) {
                if (mouseReleased->button == sf::Mouse::Button::Right ||
                    mouseReleased->button == sf::Mouse::Button::Middle) {
                    isPanning = false;
                }
            }

            if (event->is<sf::Event::MouseMoved>() && isPanning) {
                sf::Vector2i currentMousePos = sf::Mouse::getPosition(window);
                sf::Vector2f delta(
                    (float)(lastMousePos.x - currentMousePos.x) * view.getSize().x / windowWidth,
                    (float)(lastMousePos.y - currentMousePos.y) * view.getSize().y / windowHeight
                );
                view.move(delta);
                window.setView(view);
                lastMousePos = currentMousePos;
            }

            // Reset view with R key
            if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
                if (keyPressed->code == sf::Keyboard::Key::R) {
                    view = sf::View(sf::FloatRect({ 0.f, 0.f }, { windowWidth, windowHeight }));
                    window.setView(view);
                }
            }
        }

        window.clear(sf::Color(30, 30, 30));

        drawConnections(window);
        drawNeurons(window, font);
        drawLabels(window, font);

        window.display();
    }
}