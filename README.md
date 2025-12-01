This is where we will explain our project.

Descriptive statistics:

Capitalization frequency,
Word count per label,
Avg sentence length per label,
Distribution of sentence length,
Frequency n-grams (2-10),
Sentiment frequency per label,
first/last word freq per label

Preprocessing steps:

Do not remove stop words - they vary per label,
Lowercase the data,
No data augmentation,
Large training set,
No significant class imbalance,
Remove unnecessary white space,
Tokenization,
Punctuation,
Keep but make sure it is a separate token,
Differentiate between … adn .,
Do not remove stop words,

Feature extraction:

Features:
Tf-idf vectorization,
Static embeddings - hyperparameter: window size,
Sentiment frequency,
Sentence length,
Punctuation,
first/last word frequency,
Bag of words (n-grams),
Parts of speech frequency,
L1 + L2 regularization,
Ablation study,

Model Architecture & Selection

Models:
Bi-lstm,
Random forest,
Naive bayes,
Logistic regression,
CNN,
NN,
SVM

Ensemble method options:
Stacking
Boosting

Each model alone

Pure stacking

Pure boosting - for each model

Combination of boost then stack

Combination of stack then boost

Training methodology

Cross entropy vs 0-1 Loss Function

k-fold cross validation

use validation set to test hyperparameters
