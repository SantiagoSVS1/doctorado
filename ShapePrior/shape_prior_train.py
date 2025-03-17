from models import ModelManager
from generator import DataGenerator
from callbacks import get_callbacks

modelmanager = ModelManager()

input_shape = (256, 256, 2) 
image_dir = "data/image"
label_dir = "data/label"
batch_size = 1
epochs = 50

model = modelmanager.unet(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

train_generator = DataGenerator(image_dir, label_dir, batch_size, input_shape)
val_generator = DataGenerator("data/image - copia", "data/label - copia", batch_size, input_shape)

callbacks = get_callbacks()

history = model.fit(
    train_generator,  
    epochs=epochs,
    callbacks=callbacks, 
    verbose=1
)

# Evaluar el modelo
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

model.save("model.h5")