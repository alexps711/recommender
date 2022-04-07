import numpy as np


class FunkSVDModel():
    """
    Based on Simon Funk's implementation at the Netflix Prize of 2006.
    This model is based on the principles of SVD to compute ratings for events.
    """

    def __init__(self):
        self.EVENT_MATRIX = None
        self.USER_MATRIX = None
        self.latent_features_guess = 5
        self.learning_rate = 0.02
        self.steps = 500
        self.regularization_penalty = 0.02

    def train(self, user_x_product):
        print('Training SVD model...')
        return self.__forward(user_x_product)

    def predict(self, row_index):
        return np.dot(self.USER_MATRIX[row_index, :], self.EVENT_MATRIX.T)

    def get_models(self):
        return self.USER_MATRIX, self.EVENT_MATRIX

    def __forward(self, rating_matrix):
        rating_matrix = np.array(rating_matrix)

        # Build user - latent factors matrix
        user_rows = len(rating_matrix)
        USER_MATRIX = np.random.rand(user_rows, self.latent_features_guess)

        # Build events - latent factors matrix
        event_rows = len(rating_matrix[0])
        EVENT_MATRIX = np.random.rand(event_rows, self.latent_features_guess).T

        error = 0
        for step in range(self.steps):
            print("Step: " + str(step))
            # iterate each cell in r
            for i in range(user_rows):
                for j in range(len(rating_matrix[i])):
                    if rating_matrix[i][j] > 0:
                        # get the eij (error) side of the equation
                        eij = rating_matrix[i][j] - \
                            np.dot(USER_MATRIX[i, :], EVENT_MATRIX[:, j])

                        for k in range(self.latent_features_guess):
                            USER_MATRIX[i][k] = USER_MATRIX[i][k] + self.learning_rate * \
                                (2 * eij *
                                 EVENT_MATRIX[k][j] - self.regularization_penalty * USER_MATRIX[i][k])

                            EVENT_MATRIX[k][j] = EVENT_MATRIX[k][j] + self.learning_rate * \
                                (2 * eij *
                                 USER_MATRIX[i][k] - self.regularization_penalty * EVENT_MATRIX[k][j])

            # Measure error
            error = self.__error(rating_matrix, USER_MATRIX, EVENT_MATRIX)
            print("Error: " + str(error))

        self.EVENT_MATRIX = EVENT_MATRIX.T
        self.USER_MATRIX = USER_MATRIX


    def __error(self, rating_matrix, USER_MATRIX, EVENT_MATRIX):
        error = 0
        for i in range(len(rating_matrix)):
            for j in range(len(rating_matrix[i])):
                if rating_matrix[i][j] > 0:

                    # loss function error sum( (y-y_hat)^2 )
                    error = error + \
                        pow(rating_matrix[i][j]-np.dot(USER_MATRIX[i, :],
                                              EVENT_MATRIX[:, j]), 2)

                    # add regularization
                    for latent_feature in range(self.latent_features_guess):
                        # error + ||P||^2 + ||Q||^2
                        error = error + \
                            (self.regularization_penalty/2) * \
                            (pow(USER_MATRIX[i][latent_feature], 2) +
                             pow(EVENT_MATRIX[latent_feature][j], 2))
        return error
