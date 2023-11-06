from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd

df = pd.read_csv('sub-dataset.csv')

print(df)

X_train = df['text'].tolist()
y_train = df['labels'].apply(eval).tolist()

X_test = ["Mỗi ngày thức dậy, em cảm thấy mình đang mắc kẹt trong một vòng lặp vô tận, không có lối ra. Những lúc như vậy, em thường cảm thấy mất đi sự hy vọng và thậm chí, cảm giác cô đơn, dù xung quanh em có nhiều người,"]
y_test = [['1', '2', '38']]

# Binarize the labels
mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train)
y_test_bin = mlb.transform(y_test)

# Create a pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', OneVsRestClassifier(SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))),
])

# Train the classifier
text_clf.fit(X_train, y_train_bin)

# Predict on new data
predicted = text_clf.predict(X_test)

# Convert binary predictions back to labels
predicted_labels = mlb.inverse_transform(predicted)
print(predicted_labels)
