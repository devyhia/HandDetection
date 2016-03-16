from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

def load():
    _TRAIN = 1000
    _TEST = 500
    _VALIDATION = 100

    # print "Loading %d for Training, %d for Validation, and %d for Testing!" % (_TRAIN, _VALIDATION, _TEST)
    # We can now download and read the training and test set images and labels.

    print("Training: %d ..." % (_TRAIN))
    TrainData = map(lambda p: sio.loadmat('TrainData/Data_%s.mat' % (('%d') % (p)).zfill(7)), range(_TRAIN))
    print(" Done.\nValidation %d ..." % (_VALIDATION))

    # ValidationData = map(lambda p: sio.loadmat('ValidationData/Data_%s.mat' % (('%d') % (p)).zfill(7)), range(_VALIDATION))
    # print(" Done.\nTesting %d ..." % (_TEST))
    #
    # TestData = map(lambda p: sio.loadmat('TestData/Data_%s.mat' % (('%d') % (p)).zfill(7)), range(_TEST))
    # print(' Done.\n')

    def depth(data):
        d = np.array(map(lambda i: smisc.imresize(i['depth'], 0.25), data))
        d.shape = (len(d), 60*80)
        return d

    def label(data):
        d = np.array(map(lambda i: smisc.imresize(i['lbl'], 0.25), data))
        d.shape = (len(d), 60*80)
        return d

    X_train = depth(TrainData)
    y_train = label(TrainData)
    # X_test = depth(TestData)
    # y_test = label(TestData)
    # X_val = depth(ValidationData)
    # y_val = label(ValidationData)

    # We reserve the last 10000 training examples for validation.
    # X_train, X_val = X_train[:-10000], X_train[-10000:]
    # y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train #, X_val, y_val, X_test, y_test

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 60*80),  # 96x96 input pixels per batch
    hidden_num_units=60*80*3,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=60*80,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=400,  # we want to train this many epochs
    verbose=1,
    )

X, y = load()
net1.fit(X, y)
