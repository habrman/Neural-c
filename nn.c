#include "nn.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

static uint8_t getClassification(OutputLayer* outputLayer);
static void backProagate(Network* network, int label);
static void updateHiddenNode(InputLayer* inputLayer, HiddenNode* node, double backPropValue);
static void updateOutputNode(HiddenLayer* hiddenLayer, OutputNode* node, double backPropValue);
static void feedForward(Network* network, Image* img);
static double sigmoidDerivative(double nodeOutput);
static double sigmoid(double value);
static void initWeights(int weight_size, double* weights);

void initNetwork(Network* network){
    //Initialize hidden layer
    HiddenLayer* hiddenLayer = &network->hiddenLayer;
    for(int hn=0; hn<HIDDEN_LAYER_SIZE; ++hn){
        HiddenNode* node = &hiddenLayer->nodes[hn];
        node->bias = rand()/(double)(RAND_MAX);
        initWeights(IMAGE_SIZE, node->weights);
    }

    //Initialize output layer
    OutputLayer* outputLayer = &network->outputLayer;
    for(int on=0; on<OUTPUT_SIZE; ++on){
        OutputNode* node = &outputLayer->nodes[on];
        node->bias = rand()/(double)(RAND_MAX);
        initWeights(HIDDEN_LAYER_SIZE, node->weights);
    }
}

void trainNetwork(Network* network){
    FILE *imageFile;
    FILE *labelFile;
    ImageFileHeader imageFileHeader;
    imageFile = openImageFile(TRAINING_SET_IMAGE_FILE_NAME, &imageFileHeader);
    labelFile = openLabelFile(TRAINING_SET_LABEL_FILE_NAME);

    for (int i=0; i<imageFileHeader.maxImages; i++){
        Image img;
        getImage(imageFile, &img);
        uint8_t label = getLabel(labelFile);

        feedForward(network, &img);
        backProagate(network, label);
    }
}

void testNetwork(Network *network){
    FILE *imageFile;
    FILE *labelFile;
    ImageFileHeader imageFileHeader;
    imageFile = openImageFile(TEST_SET_IMAGE_FILE_NAME, &imageFileHeader);
    labelFile = openLabelFile(TEST_SET_LABEL_FILE_NAME);

    int errCount = 0;
    for (int i=0; i<imageFileHeader.maxImages; i++){
        Image img;
        getImage(imageFile, &img);
        uint8_t lbl = getLabel(labelFile);
        feedForward(network, &img);

        uint8_t classification = getClassification(&network->outputLayer);
        if (classification!=lbl){
            errCount++;
        }
    }
    fclose(imageFile);
    fclose(labelFile);

    printf("Test Accuracy: %0.2f%%\n", ((double)(imageFileHeader.maxImages - errCount) / imageFileHeader.maxImages) * 100);
}

static void initWeights(int weight_size, double* weights){
    //Initialize weights between -0.7 and 0.7
    for(int w=0; w<weight_size; ++w){
        weights[w] = 0.7 * (rand()/(double)(RAND_MAX));
        if (w%2){
            weights[w] = -weights[w];
        }
    }
}

static double sigmoid(double value){
    return 1.0 / (1.0 + exp(-value));
}

static double sigmoidDerivative(double nodeOutput){
    return nodeOutput * (1- nodeOutput);
}

static void feedForward(Network* network, Image* img){
    //Populate the input layer with normalized input
    for(int i=0; i<IMAGE_SIZE; ++i)
    {
        network->inputLayer.output[i] = (double)(img->pixels[i] / 255.0);
    }

    //Propagate through the hidden layer
    for(int hn=0; hn<HIDDEN_LAYER_SIZE; ++hn){
        HiddenNode* node = &network->hiddenLayer.nodes[hn];
        node->output = node->bias;

        for(int w=0; w<IMAGE_SIZE; ++w){
            node->output += network->inputLayer.output[w] * node->weights[w];
        }
        node->output = sigmoid(node->output);
    }

    //Calculate network output
    for(int on=0; on<OUTPUT_SIZE; ++on){
        OutputNode* node = &network->outputLayer.nodes[on];
        node->output = node->bias;

        for(int w=0; w<HIDDEN_LAYER_SIZE; ++w){
            node->output += network->hiddenLayer.nodes[w].output * node->weights[w];
        }
        node->output = sigmoid(node->output);
    }
}

static void updateOutputNode(HiddenLayer* hiddenLayer, OutputNode* node, double backPropValue){
    for(int hn=0; hn<HIDDEN_LAYER_SIZE;++hn){
        HiddenNode* hiddenNode = &hiddenLayer->nodes[hn];
        node->weights[hn] += LEARNING_RATE * hiddenNode->output * backPropValue;
    }
    node->bias += LEARNING_RATE * backPropValue;
}

static void updateHiddenNode(InputLayer* inputLayer, HiddenNode* node, double backPropValue){
    for(int in=0; in<IMAGE_SIZE; ++in){
        node->weights[in] += LEARNING_RATE * inputLayer->output[in] * backPropValue;
    }
    node->bias += LEARNING_RATE * backPropValue;
}

static void backProagate(Network* network, int label){
    HiddenLayer* hiddenLayer = &network->hiddenLayer;
    OutputLayer* outputLayer = &network->outputLayer;

    for(int on=0; on<OUTPUT_SIZE; ++on){
        OutputNode* outputNode = &outputLayer->nodes[on];

        int nodeTarget = (on==label) ? 1:0;
        double errorDelta = nodeTarget - outputNode->output;
        double backPropValue = errorDelta * sigmoidDerivative(outputNode->output);

        outputNode->backPropValue = backPropValue;
        updateOutputNode(&network->hiddenLayer, outputNode, outputNode->backPropValue);
    }

    for(int hn=0; hn<HIDDEN_LAYER_SIZE; ++hn){
        HiddenNode* hiddenNode = &hiddenLayer->nodes[hn];

        double outputNodesBackPropSum = 0;

        for(int on=0; on<OUTPUT_SIZE; ++on){
            OutputNode* outputNode = &outputLayer->nodes[on];
            outputNodesBackPropSum += outputNode->backPropValue * outputNode->weights[hn];
        }

        double hiddenNodeBackPropValue = outputNodesBackPropSum * sigmoidDerivative(hiddenNode->output);
        updateHiddenNode(&network->inputLayer, hiddenNode, hiddenNodeBackPropValue);
    }
}

static uint8_t getClassification(OutputLayer* outputLayer){
    double maxOutput = 0;
    int maxIndex = 0;

    for(int on=0; on<OUTPUT_SIZE; ++on){
        double nodeOutput = outputLayer->nodes[on].output;
        if(nodeOutput > maxOutput){
            maxOutput = nodeOutput;
            maxIndex = on;
        }
    }
    return (uint8_t)maxIndex;
}
