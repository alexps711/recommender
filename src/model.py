from turtle import forward
import torch
import torch.nn as nn


class Model(nn.Module):
    """Matrix Factorization model."""

    def __init__(self, num_users, num_events, num_factors=100):
        """Initialize the model with the user and event embeddings.

        Args:
            num_users (int): The number of users to consider.
            num_events (int): The number of events to consider.
            num_factors (int, optional): The size of the embeddings. Defaults to 100.
        """
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, num_factors)
        self.event_embeddings = nn.Embedding(num_events, num_factors)

    def __forward(self, user_id, event_id):
        user_id = self.user_embeddings(user_id)
        event_id = self.event_embeddings(event_id)
        return (user_id * event_id).sum(dim=1)

    def predict(self, user_id, event_id):
        """Calculate the dot product of the user and event embeddings.

        Args:
            user_id (int): the id (row) of the user.
            event_id (int): the id (row) of the event.

        Returns:
            int: The prediction (dot product).
        """
        return self.__forward(user_id, event_id)
    
    def train():
        """Train the model."""
        # TODO: implement
        
    def validate():
        """Validate the model."""
        # TODO: implement
    
    
