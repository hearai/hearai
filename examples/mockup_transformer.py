import numpy as np
import torch.nn as nn

from torch.optim import Adam

from mockups.transformer_mockups import generate_mockup_input, generate_mockup_output
from models.transformers.pretrained_transformer import PretrainedTransformer


batch_size = 4
input_shape = (7, 512,)
value_range = (0., 1.)
num_classes = 2

feature_extractor_output = generate_mockup_input(
    batch_size=batch_size, input_shape=input_shape, value_range=value_range
)
labels = generate_mockup_output(batch_size=batch_size, num_classes=num_classes)

# simulate model training
checkpoint = 'facebook/hubert-large-ls960-ft'
encoder_list = [64, 10, 2]
model = PretrainedTransformer(checkpoint=checkpoint,
                              encoder_list=encoder_list,
                              add_softmax=True)
loss_fn = nn.functional.cross_entropy
optimizer = Adam(model.parameters(), lr=1e-4)
epochs = 1000

for epoch in range(epochs):
    X, y_true = feature_extractor_output, labels

    y_pred = model(X)
    loss = loss_fn(y_pred, y_true)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch % 1) == 0:
        print(f'Epoch {epoch}/{epochs}, loss: {loss}')

print(f"Predicted labels: {np.argmax(model(X).detach().numpy(), axis=1)}")
print(f"True labels: {np.argmax(y_true, axis=1)}")
