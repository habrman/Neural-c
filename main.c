#include "nn.h"

int main()
{
    Network network;
    initNetwork(&network);

    testNetwork(&network);
    for(int i=0; i<TRAINING_EPOCHS; ++i){
        printf("Training epoch %i/%i\n", i + 1, TRAINING_EPOCHS);
        trainNetwork(&network);
        testNetwork(&network);
    }

    return 0;
}
