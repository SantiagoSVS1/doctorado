from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def get_callbacks():
    """
    Define y retorna una lista de callbacks para el entrenamiento del modelo.
    """
    # Callback para guardar el mejor modelo durante el entrenamiento
    checkpoint = ModelCheckpoint(
        "best_model.h5",  # Ruta donde se guardará el modelo
        monitor="val_loss",  # Métrica a monitorear
        save_best_only=True,  # Guardar solo el mejor modelo
        mode="min",  # Minimizar la métrica (en este caso, la pérdida)
        verbose=1
    )

    # Callback para reducir la tasa de aprendizaje si la pérdida no mejora
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",  # Métrica a monitorear
        factor=0.1,  # Factor por el cual se reduce la tasa de aprendizaje
        patience=5,  # Número de épocas sin mejora antes de reducir la tasa
        min_lr=1e-6,  # Tasa de aprendizaje mínima
        mode="min",  # Minimizar la métrica
        verbose=1
    )

    # Callback para detener el entrenamiento si no hay mejora
    early_stopping = EarlyStopping(
        monitor="val_loss",  # Métrica a monitorear
        patience=10,  # Número de épocas sin mejora antes de detener
        restore_best_weights=True,  # Restaurar los pesos del mejor modelo
        verbose=1
    )

    # Lista de callbacks
    callbacks = [checkpoint, reduce_lr, early_stopping]
    
    return callbacks