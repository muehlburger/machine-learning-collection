from __future__ import print_function
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from text_gen_utils import *
import sys
import io

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

#model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])

# Run this cell to try with different inputs without having to re-train the model
generate_output()