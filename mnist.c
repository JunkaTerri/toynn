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


typedef struct {
	int total;
	int used;
	int* index;
} Bstate;


static inline int random_int(int min, int max) {
	return min + rand() % (max - min);
}

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

void get_next_batch(Bstate* state, int* batch, int bsize) {
	for (int i = 0; i < bsize; i++) {
		batch[i] = -1;
	}
	for (int i = 0; i < bsize; i++) {
		if (state->total - state->used == 0) {
			if (i == 0) {
				state->used = 0;
			} else {
				break;
			}
		}
		int r = random_int(state->used, state->total);
		int temp = state->index[r];
		state->index[r] = state->index[state->used];
		state->index[state->used++] = temp;
		batch[i] = temp;
	}
}


void activate_network(Digit* digit, Network* net) {
	for (int i = 0; i < 784; i++) {
		net->layers[0].neurons[i].a = digit->pixels[i] / 255.f;
	}
}



int main() {
	Digit* train_set = 0;
	Bstate state = {0};
	state.total = load_dataset(&train_set, "train.bin");
	state.index = malloc(state.total*sizeof(int));
	for (int i = 0; i < state.total; i++) {
		state.index[i] = i;
	}
	printf("Dataset loaded.\n");
	int bsize = 100;
	int batch[100];
	float truth[10] = {0};
	float eta = 1.f;

	int cycles = 20000;

	Network net = {
		{784, 16, 16, 10},
		4,
		0
	};

	Network delta = {
		{784, 16, 16, 10},
		4,
		0
	};

	Network delta_sum = {
		{784, 16, 16, 10},
		4,
		0
	};

	init_network(&net);
	init_network(&delta);
	init_network(&delta_sum);

	printf("networks initialized.\n");
	xavier_rfill_network(&net);
	printf("network randomized.\n");

	for (int i = 0; i < cycles; i++) {
		get_next_batch(&state, batch, bsize);
		printf("batch prepared.\n");
		for (int j = 0; j < bsize; j++) {
			int di = batch[j];
			activate_network(&train_set[di], &net);
			forward_pass(&net);
			truth[train_set[di].label] = 1.f;
			back_pass(&net, &delta, truth);
			truth[train_set[di].label] = 0.f;
			add_wab(&delta_sum, &delta);
		}
		descend(&net, &delta_sum, eta / (float)bsize);
		reset_network(&delta_sum);
		printf("Batch %d done.\n", i);
	}
	save_network(&net, "weights.bin");

	free(train_set);
	free_network(&net);
	free_network(&delta);
	free_network(&delta_sum);
}
