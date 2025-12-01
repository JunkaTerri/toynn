#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define TNN_IMPLEMENTATION

#include "tnn.h"

typedef struct {
	uint8_t label;
	uint8_t pixels[784];
} Digit;


int load_dataset(Digit** dataset, char* filename) {
	FILE* f = fopen(filename, "rb");
	fseek(f, 0, SEEK_END);
	long filesize = ftell(f);
	int records = filesize / sizeof(Digit);
	*dataset = (Digit*)malloc(records*sizeof(Digit));
	fseek(f, 0, SEEK_SET);
	fread(*dataset, sizeof(Digit), records, f);
	fclose(f);
	return records;
}


void activate_network(Digit* digit, Network* net) {
	for (int i = 0; i < 784; i++) {
		net->layers[0].neurons[i].a = digit->pixels[i] / 255.f;
	}
}


int main() {
	Digit* test_set = 0;
	int total = load_dataset(&test_set, "test.bin");
	printf("Dataset loaded.\n");

	Network net = {
		{784, 16, 16, 10},
		4,
		0
	};

	init_network(&net);
	load_network(&net, "weights.bin");


	printf("networks initialized.\n");


	int correct = 0;

	for (int i = 0; i < total; i++) {
		activate_network(&test_set[i], &net);
		forward_pass(&net);
		int max = 0;
		for (int i = 0; i < 10; i++) {
			if (net.layers[net.lc-1].neurons[i].a > net.layers[net.lc-1].neurons[max].a) {
				max = i;
			}
		}
		if (max == test_set[i].label) {
		correct++;
		}
		printf("Ground Truth: %d\n", test_set[i].label);
		printf("Prediction  : %d\n", max);
		printf("------------------\n\n");
	}


	printf("Success : %d/%d\n", correct, total);
	printf("Accuracy: %.2f%%\n", correct / (float)total * 100.f);

	free(test_set);
	free_network(&net);
}
