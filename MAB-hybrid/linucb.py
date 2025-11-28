import numpy as np

class LinUCBAgent:
    """
    Disjoint Linear Upper Confidence Bound (LinUCB) Agent.
    
    References:
        Li et al., "A Contextual-Bandit Approach to Personalized News Article Recommendation", WWW 2010.
        (Algorithm 1)
    """
    def __init__(self, n_arms, n_features, alpha=0.1):
        """
        Args:
            n_arms (int): Number of distinct actions (fusion weights).
            n_features (int): Dimension of the context vector.
            alpha (float): Exploration hyperparameter. Higher alpha = more exploration.
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        
        # Initialize disjoint matrices for each arm
        # A: Covariance matrix (d x d), initialized to Identity for Ridge Regularization
        # b: Reward-weighted feature vector (d x 1), initialized to zeros
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, context_vector):
        """
        Selects an arm based on the Upper Confidence Bound (UCB) of the estimated reward.
        
        Args:
            context_vector (np.array): Shape (n_features,)
            
        Returns:
            int: Index of the selected arm.
        """
        p = np.zeros(self.n_arms)
        
        for a in range(self.n_arms):
            # 1. Compute the inverse of A (Ridge Regression covariance)
            # In production, use np.linalg.solve or update inverse iteratively for speed
            A_inv = np.linalg.inv(self.A[a])
            
            # 2. Compute the estimated coefficient (theta)
            # theta = A^-1 * b
            theta = A_inv @ self.b[a]
            
            # 3. Calculate the standard deviation (uncertainty width)
            # std = sqrt(x.T * A^-1 * x)
            std_dev = np.sqrt(context_vector.T @ A_inv @ context_vector)
            
            # 4. Calculate UCB
            # Prediction + Exploration Bonus
            p[a] = theta @ context_vector + self.alpha * std_dev
            
        # Tie-breaking: randomly choose among max if multiple arms share the same score
        # (np.argmax usually takes the first occurrence, which is fine here)
        return np.argmax(p)

    def update(self, arm, context_vector, reward):
        """
        Updates the internal matrices A and b for the specific arm that was chosen.
        
        Args:
            arm (int): The arm index that was selected.
            context_vector (np.array): The feature vector observed.
            reward (float): The actual reward (NDCG) received.
        """
        # Outer product of context vector (d x d)
        self.A[arm] += np.outer(context_vector, context_vector)
        
        # Update bias vector
        self.b[arm] += reward * context_vector