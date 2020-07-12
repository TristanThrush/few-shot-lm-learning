# few-shot-lm-learning

First, add the necessary dependencies. If on openmind, most can be added with:
$ module add openmind/anaconda
$ module add nklab/pytorch

You should also install the transformers library, which can be done on openmind
with:
$ pip install --user transformers

If using openmind, you can run with:
$ sbatch train.sh

If on your own computer, you can run with:
$ python main.py

The code trains a pretrained bidirectional transformer on the data in
"train.txt" for several seeds. The data is tokenized as expected, except for
words of the form [V<number>] and [N<number>]. For these words, a new token is
added to the transformer's tokenizer, a new word embedding is created for the
token, and a new output neuron in the lm head is created. All of the weights
are frozen during training except the newly added weights.

After training, an evaluation is performed. The test data is found in
"test.txt" and lines of the file are structured as tuple with two sentences:

('Will the [MASK] be [V1]-[V10] onto the [MASK]?', 'Will the [MASK] be [V11]-[V20] onto the [MASK]?')

For each line in "test.txt", the tuned transformer takes in each of the
two sentences. Each sentence must have exactly one [V<number>] token, or
exactly one token of the form [V<number1>]-[V<number2>]. For several training
seeds, the probability from the lm head of the [V<number>] token is computed.
If there is a [V<number1>]-[V<number2>] token, then the average probability
from the lm head of all [V<number>] tokens from the range [number1, number2] is
computed. The probability values from several seeds on the two sentences are
used in a nonparametric test to determine if the median of the distribution of
probabilities from the first sentence is significantly different from that of
the second sentence. The test is Mood's median test
(https://en.wikipedia.org/wiki/Median_test). For each line in the test file,
the p-value is returned as well as the median probability score for the first
sentence followed by that of the second sentence. This way, we can determine if
the transformer assigns significantly higher probability to one type of nonce
word versus another type of nonce word, usually holding context constant.
