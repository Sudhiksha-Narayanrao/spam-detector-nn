import numpy as np

# Sigmoid activation function
#maps any value to a value between 0 and 1. We use it to convert numbers to probabilities
#One of the desirable properties of a sigmoid function is that its output can be used to create its derivative.
# If the sigmoid's output is a variable "out"
#then the derivative is simply out * (1-out). This is very efficient.
#just think about the deriv. as the slope of the sigmoid function at a given point

def nonlin(x, deriv=False):   # this means that by default, deriv will be False unless someone specifically passes True when calling this function.
    if deriv:                 # this is asking:“Did the user ask for the derivative instead of the regular sigmoid function?” If yes, it returns the derivative of the sigmoid; derivative of sigmoid is for backpropagation (learning mechanism an NN uses to adjust its weights based on how wrong its predictions were)
        return x * (1 - x)  # Derivative for backpropagation
    return 1 / (1 + np.exp(-x))  # Sigmoid: squashes values between 0 and 1


# Input dataset (X)
# Each row is one email
# Features: [contains_offer, has_link, all_caps, contains_money, is_sender_unknown, has_attachment]

X = np.array([
    [0, 0, 0, 0, 0, 0],  # normal email: no spammy features
    [1, 1, 1, 1, 1, 0],  # spam: offer + link + CAPS + money + unknown sender
    [0, 1, 0, 1, 1, 1],  # suspicious: link + money + unknown sender + attachment
    [1, 0, 1, 0, 0, 0],  # offer + CAPS (spammy wording, no link)
    [0, 1, 0, 0, 1, 1],  # unknown sender + attachment (phishy)
    [1, 1, 1, 1, 1, 1],  # full-on spam: all 6 features present
])

# there’s no strict threshold of how many 1s = spam
# the model learns the importance (weight) of each feature
# more spammy features -> higher probability of being spam


# Output labels (y)
# 1 = Spam, 0 = Not Spam
y = np.array([[0, 1, 0, 1, 0, 1]]).T


# Initialize weights
np.random.seed(1)    #This sets the random number generator in NumPy to a specific starting point (called a seed;which is 1 here)
                     #so that the random numbers it produces are always the same every time you run the code.

# 6 inputs -> 1 output neuron
syn0 = 2 * np.random.random((6, 1)) - 1  #Create a 6x1 column matrix of random numbers between -1 & 1 and store it in syn0 (weight matrix)
                                        #using the random() function from the random module of NumPy


# Training loop
for iter in range(10000):
    # Forward pass
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    # l0 is just the input layer.
    # np.dot(l0, syn0) performs matrix multiplication of input X with weights syn0.
    # nonlin(...) applies the sigmoid activation to this result.
    # Resulting l1 is the network's prediction/output.

    # Error
    l1_error = y - l1

    # Backpropagation
    l1_delta = l1_error * nonlin(l1, True) #Gradient = Error × Sensitivity
                                                  #nonlin(l1, True) → the derivative of sigmoid
                                                  #which tells us how sensitive the output is to changes in weights

    # Update weights
    syn0 += np.dot(l0.T, l1_delta)  #New weights = old weights + amount to shift



# After training: predictions
print("Output After Training:\n")
for i, output in enumerate(l1): #enumerate(l1) gives you-> i: the index (email number) and output: the actual predicted value (e.g. 0.87)
    label = "Spam" if output > 0.5 else "Not Spam"
    print(f"Email {i+1} prediction: {output[0]:.4f} -> {label}") #i+1 makes the index human-friendly (starting from 1 instead of 0)



# Function to predict new emails
def predict_email(features):
    """
    features: list of 6 binary values [offer, link, all_caps, money, unknown_sender, attachment]
    returns: probability and label
    """
    input_data = np.array(features).reshape(1, 6) #.reshape(1, 6) turns it into the shape expected by the neural network:
                                                  #1 row (one email), 6 columns (6 features)
    prediction = nonlin(np.dot(input_data, syn0))[0][0] #[0][0] is used to pull the raw number out of the 2D array
    label = "Spam" if prediction > 0.5 else "Not Spam"
    print(f"\nNew Email Prediction -> {prediction:.4f} -> {label}")



# Example: [offer=1, link=0, all_caps=1, money=1, unknown sender=0, attachment=0]
predict_email([1, 0, 1, 1, 0, 0])

# You can try others like:
# predict_email([0, 1, 0, 0, 0, 0])
# predict_email([1, 1, 1, 1, 1, 0])
