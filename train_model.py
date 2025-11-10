import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# Load dataset
df = pd.read_csv("bbc.csv")
df.columns = df.columns.str.strip()  # remove leading/trailing spaces

# Automatically detect text and label columns
text_column = None
label_column = None

for col in df.columns:
    if any(keyword in col.lower() for keyword in ['headline', 'title', 'text']):
        text_column = col
    elif any(keyword in col.lower() for keyword in ['category', 'label', 'class']):
        label_column = col

if text_column is None or label_column is None:
    raise ValueError("Could not detect text or label column. Please check your CSV.")

print(f"Using '{text_column}' as text column and '{label_column}' as label column.")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df[label_column])
num_classes = len(le.classes_)

# Save label encoder
joblib.dump(le, "label_encoder.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df[text_column], y, test_size=0.2, random_state=42)

# Tokenize
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
max_len = 20
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# Save tokenizer
joblib.dump(tokenizer, "tokenizer.pkl")

# Convert labels to categorical
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# Build LSTM model
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=max_len),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(X_train_pad, y_train_cat, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(X_test_pad, y_test_cat)
print("Test Accuracy:", acc)

# Save model
model.save("news_category_model_dl.h5")
print("Model saved as news_category_model_dl.h5")
