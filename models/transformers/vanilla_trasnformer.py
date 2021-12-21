import timm
import torch.nn as nn
import torch.optim


class TransformerModel(nn.Module):
    """Basic transformer model"""

    def __init__(self,
                 input_size: int = 128,
                 output_size: int = 1000):
        """
        Dummy example of __init__ function of basic transformer model. Does nothing as transformer is not even implemented.
        
        Args:
            input_size (int, optional): Input size fot the transformer model. Should be equal to the output size of the feature_extrator. Defaults to 128.
            output_size (int, optional): Output size. Should be equal to the selected representation size. Defaults to 1000.
        """
        super().__init__()
        self.__fully_connected = nn.Linear(input_size, output_size)
        self.__output_layer = nn.Softmax(dim=1)
        # this is a dummy example but in practice this will be longer 

    def forward(self, input, **kwargs):
        x = self.__fully_connected(input)
        x = self.__output_layer(x)
        return x


if __name__ == '__main__':
    import numpy as np
    import torch.nn as nn

    from hearai.mockups.transformer_mockups import generate_mockup_input, generate_mockup_output

    batch_size = 2
    input_shape = (128,)
    value_range = (0., 1.)
    num_classes = 10

    feature_extractor_output = generate_mockup_input(batch_size=batch_size,
                                                     input_shape=input_shape,
                                                     value_range=value_range)
    labels = generate_mockup_output(batch_size=batch_size,
                                    num_classes=num_classes)

    # simulate model training
    model = TransformerModel(input_size=128, output_size=10)
    loss_fn = nn.functional.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 1000

    for epoch in range(epochs):
        X, y_true = feature_extractor_output, labels

        y_pred = model(X)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (epoch % 10) == 0:
            print(f'Epoch {epoch}/{epochs}, loss: {loss}')

    print(f'Predicted labels: {np.argmax(model(X).detach().numpy(), axis=1)}')
    print(f'True labels: {np.argmax(y_true, axis=1)}')




