compile with `make`.

run `./train` to train the neural net, and `./infer` to run inference on test data.

`train.bin` and `test.bin` contain mnist data set as an array of 784 `uint8_t` pixels + 1 `uint8_t` label

```
[784 bytes pixel value] [1 byte lable] [784 bytes pixel value] [1 byte label] ...
```
