#ifndef TNN_H
#define TNN_H

typedef struct {
	float bias;
	float a; // activation of the current neuron
	int wc; // weight count of current neuron
	float* weights;
} Neuron;

typedef struct {
	int nc; // neuron count of current layer
	Neuron* neurons;
} Layer;

typedef struct {
	int arch[255];
	int lc;
	Layer* layers;
} Network;

inline static float random_uniform(float min, float max) {
	return min + ( max - min ) * rand() / (float)RAND_MAX;
}

inline static float sigmoid(float x) {
	return 1.0 / ( 1.0 + exp(-x) );
}

inline static float d_sigmoid(float x) {
	return x * (1 - x);
}

void init_network(Network* net);

void xavier_rfill_network(Network* net);

void free_network(Network* net);

void forward_pass(Network* net);

void back_pass(Network* net, Network* delta, float* truth);

void add_wab(Network* net, Network* delta);

void reset_network(Network* net);

void descend(Network* net, Network* delta, float rate);

void save_network(Network* net, char* filename);

void load_network(Network* net, char* filename);


#endif // TNN_H

#ifdef TNN_IMPLEMENTATION

#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

void init_network(Network* net) {
	net->layers = malloc(net->lc*sizeof(Layer));
	for (int i = 0; i < net->lc; i++) {
		net->layers[i] = (Layer) {
			.nc = net->arch[i],
			.neurons = (Neuron*)malloc(net->arch[i]*sizeof(Neuron))
		};
		if (i == 0) {
			continue;
		}
		for (int j = 0; j < net->layers[i].nc; j++) {
			net->layers[i].neurons[j] = (Neuron) {
				.wc = net->arch[i-1],
				.weights = (float*)malloc(net->arch[i-1]*sizeof(float))
			};
		}
	}
}


void xavier_rfill_network(Network* net) {
	for (int i = 1; i < net->lc; i++) {
		// use xavier (glorot) method to initialize weights
		float limit = sqrtf( 6.0f / ( net->layers[i-1].nc + net->layers[i].nc) );
		for (int j = 0; j < net->layers[i].nc; j++) {
			for (int k = 0; k < net->layers[i].neurons[j].wc; k++) {
				net->layers[i].neurons[j].weights[k] = random_uniform(-limit, limit);
			}
		}
	}
}

void free_network(Network* net) {
	for (int i = 0; i < net->lc; i++) {
		if (i == 0) {
			free(net->layers[i].neurons);
			continue;
		}
		for (int j = 0; j < net->layers[i].nc; j++) {
			free(net->layers[i].neurons[j].weights);
		}
		free(net->layers[i].neurons);
	}
	free(net->layers);
}


void forward_pass(Network* net) {
	for (int i = 1; i < net->lc; i++) {
		float z;
		for (int j = 0; j < net->layers[i].nc; j++) {
			z = 0;
			for (int k = 0; k < net->layers[i].neurons[j].wc; k++) {
				z += net->layers[i].neurons[j].weights[k] * net->layers[i-1].neurons[k].a;
			}
			z += net->layers[i].neurons[j].bias;
			net->layers[i].neurons[j].a = sigmoid(z);
		}
	}
}

void back_pass(Network* net, Network* delta, float* truth) {
	// maybe this loop can be added inside the next loop, but maybe too much unnecessary comparison?
	for (int j = 0, i = net->lc - 1; j < net->layers[i].nc; j++) {
		delta->layers[i].neurons[j].bias = 2.f*( net->layers[i].neurons[j].a - truth[j] )*d_sigmoid(net->layers[i].neurons[j].a);
		for (int k = 0; k < net->layers[i].neurons[j].wc; k++) {
			delta->layers[i].neurons[j].weights[k] = delta->layers[i].neurons[j].bias * net->layers[i-1].neurons[k].a;
		}
	}
	for (int i = net->lc - 2; i > 0; i--) {
		for (int j = 0; j < net->layers[i].nc; j++) {
			delta->layers[i].neurons[j].bias = 0;
			for (int k = 0; k < net->layers[i+1].nc; k++) {
				delta->layers[i].neurons[j].bias += delta->layers[i+1].neurons[k].bias*net->layers[i+1].neurons[k].weights[j];
			}
			delta->layers[i].neurons[j].bias *= d_sigmoid(net->layers[i].neurons[j].a);
			for (int k = 0; k < net->layers[i].neurons[j].wc; k++) {
				delta->layers[i].neurons[j].weights[k] = delta->layers[i].neurons[j].bias * net->layers[i-1].neurons[k].a;
			}
		}
	}
}

void add_wab(Network* net, Network* delta) {
	for (int i = 1; i < delta->lc; i++) {
		for (int j = 0; j < delta->layers[i].nc; j++) {
			net->layers[i].neurons[j].bias += delta->layers[i].neurons[j].bias;
			for (int k = 0; k < delta->layers[i].neurons[j].wc; k++) {
				net->layers[i].neurons[j].weights[k] += delta->layers[i].neurons[j].weights[k];
			}
		}
	}
}

void reset_network(Network* n) {
	for (int i = 1; i < n->lc; i++) {
		for (int j = 0; j < n->layers[i].nc; j++) {
			n->layers[i].neurons[j].bias = 0.f;
			memset(n->layers[i].neurons[j].weights, 0, n->layers[i].neurons[j].wc*sizeof(float));
		}
	}
}

void descend(Network* net, Network* delta, float rate) {
	for (int i = 1; i < net->lc; i++) {
		for (int j = 0; j < net->layers[i].nc; j++) {
			net->layers[i].neurons[j].bias -= delta->layers[i].neurons[j].bias*rate;
			for (int k = 0; k < net->layers[i].neurons[j].wc; k++) {
				net->layers[i].neurons[j].weights[k] -= delta->layers[i].neurons[j].weights[k]*rate;
			}
		}
	}
}


void save_network(Network* net, char* filename) {
	FILE* f = fopen(filename, "wb");
	uint8_t l = (uint8_t)net->lc;
	fwrite(&l, 1, 1, f);
	for (int i = 0; i < net->lc; i++) {
		fwrite(&(net->layers[i].nc), sizeof(int), 1, f);
	}
	for (int i = 1; i < net->lc; i++) {
		for (int j = 0; j < net->layers[i].nc; j++) {
			fwrite(net->layers[i].neurons[j].weights, sizeof(float), net->layers[i].neurons[j].wc, f);
			fwrite(&(net->layers[i].neurons[j].bias), sizeof(float), 1, f);
		}
	}
	fclose(f);
}


void load_network(Network* net, char* filename) {
	// very hacky save / load system for weights and biases
	/* [[1 byte layer count]
	 *  [4 byte int neuron count] * layer_count
	 *  [weights][bias]
	 *  [weights][bias]
	 *  .
	 *  .
	 *  .
	 *  [weights][bias]
	 * ]
	 */
	FILE* f = fopen(filename, "rb");
	uint8_t l;
	fread(&l, 1, 1, f);
	int* layer_length = (int*)malloc( (int)l*sizeof(int) );
	fread(layer_length, sizeof(int), (int)l, f);
	*net = (Network) {
		.arch = {0},
		.lc = (int)l,
		.layers = (Layer*)calloc((int)l, sizeof(Layer))
	};
	for (int i = 0; i < net->lc; i++) {
		net->arch[i] = layer_length[i];
	}
	init_network(net);
	for (int i = 1; i < net->lc; i++) {
		for (int j = 0; j < net->layers[i].nc; j++) {
			fread(net->layers[i].neurons[j].weights, sizeof(float), net->layers[i].neurons[j].wc, f);
			fread(&(net->layers[i].neurons[j].bias), sizeof(float), 1, f);
		}
	}
	free(layer_length);
	fclose(f);
}




#endif // TNN_IMPLEMENTATION
