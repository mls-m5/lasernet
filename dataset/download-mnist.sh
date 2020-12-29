
echo downloading training images...
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -qO- \
    | gunzip - > train-images-idx3-ubyte

echo downloading training labels...
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -qO- \
    | gunzip - > train-labels-idx1-ubyte

echo downloading test images...
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -qO- \
    | gunzip - > t10k-images-idx3-ubyte

echo downloading test labels...
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -qO- \
    | gunzip - > t10k-labels-idx1-ubyte

