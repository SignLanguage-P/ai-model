from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt

# 데이터 준비: LSTM에 맞게 데이터 형태 변환
# LSTM은 시퀀스 데이터를 처리하므로 데이터를 (samples, timesteps, features) 형태로 변환
sequence_length = 10  # 시퀀스 길이 (프레임 수)
num_features = X_train.shape[1]  # 랜드마크 데이터 크기 (63차원)

# 데이터를 시퀀스 형태로 리샘플링
def reshape_to_sequences(X, seq_length):
    return np.array([X[i:i + seq_length] for i in range(len(X) - seq_length + 1)])

X_train_seq = reshape_to_sequences(X_train, sequence_length)
X_test_seq = reshape_to_sequences(X_test, sequence_length)

# 라벨도 시퀀스에 맞춰 크기를 조정
y_train_seq = y_train[sequence_length - 1:]
y_test_seq = y_test[sequence_length - 1:]

# LSTM 모델 정의
model = models.Sequential([
    # 첫 번째 LSTM 레이어
    layers.LSTM(256, activation='tanh', return_sequences=True, kernel_regularizer=regularizers.l2(0.005), input_shape=(sequence_length, num_features)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    # 두 번째 LSTM 레이어
    layers.LSTM(128, activation='tanh', return_sequences=False, kernel_regularizer=regularizers.l2(0.005)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    # 출력(Dense) 레이어
    layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.005))
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks 설정
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)

# 학습
history = model.fit(
    X_train_seq, y_train_seq,
    epochs=80,
    batch_size=32,
    validation_data=(X_test_seq, y_test_seq),
    callbacks=[early_stopping, lr_scheduler]
)

model.save('p_signlanguage_model.h5')
