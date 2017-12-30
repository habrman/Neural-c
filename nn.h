#include "dataLoader.h"
#include "config.h"

typedef struct HiddenNode_{
    double bias;
    double output;
    double weights[IMAGE_SIZE];
} HiddenNode;

typedef struct OutputNode_{
    double bias;
    double output;
    double backPropValue;
    double weights[HIDDEN_LAYER_SIZE];
} OutputNode;

typedef struct InputLayer_{
    double output[IMAGE_SIZE];
} InputLayer;

typedef struct HiddenLayer_{
    HiddenNode nodes[HIDDEN_LAYER_SIZE];
} HiddenLayer;

typedef struct OutputLayer_{
    OutputNode nodes[OUTPUT_SIZE];
} OutputLayer;

typedef struct Network_{
    InputLayer inputLayer;
    HiddenLayer hiddenLayer;
    OutputLayer outputLayer;
} Network;

void initNetwork(Network* network);
void trainNetwork(Network* network);
void testNetwork(Network *network);
