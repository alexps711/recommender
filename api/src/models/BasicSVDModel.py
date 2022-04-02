from models import Model 
import numpy as np

class BasicSVDModel():
    """This is a baseline matrix factorization model."""

    def __init__(self, num_users, num_events, num_factors=5):
        """Initialize the model with the user and event embeddings.

        Args:
            num_users (int): The number of users to consider.
            num_events (int): The number of events to consider.
            num_factors (int, optional): The size of the embeddings. Defaults to 5.
        """
        super().__init__()

    def __forward(self, user_id, event_id):
        utilMat = np.array(train)
        # the nan or unavailable entries are masked
        mask = np.isnan(utilMat)
        masked_arr = np.ma.masked_array(utilMat, mask)
        item_means = np.mean(masked_arr, axis=0)
        # nan entries will replaced by the average rating for each item
        utilMat = masked_arr.filled(item_means)
        x = np.tile(item_means, (utilMat.shape[0],1))
        # The magic happens here. U and V are user and item features
        U, s, V= np.linalg.svd(utilMat, full_matrices=False)
        s=np.diag(s)
        s_root= np.sqrt(s)
        Usk=np.dot(U,s_root)
        skV=np.dot(s_root,V)
        UsV = np.dot(Usk, skV)
        UsV = UsV + x
        return UsV

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