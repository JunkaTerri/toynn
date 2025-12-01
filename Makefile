all:
	cc mnist.c -lm -o train -Wall -Wextra -O3
	cc infer.c -lm -o infer -Wall -Wextra -O3
