# Spam Detector Neural Network (NumPy)

This is a simple neural network I built from scratch using just NumPy. The goal was to train it to detect whether an email is spam or not based on a few yes/no features.


## What this project does

It’s a single-layer neural network that takes in six binary features (like whether the email contains a link or certain keywords) and predicts whether it’s spam. I wrote the forward propagation and backpropagation manually, just using NumPy. No TensorFlow or PyTorch — this was mainly to learn how the math behind neural networks works.

The model outputs a number between 0 and 1, and we treat anything above 0.5 as spam.


## Features used

Each email is described using 6 binary features (`1` = yes, `0` = no):

- `contains_offer` – Words like “offer”, “free”, “win”
- `has_link` – If there’s a hyperlink in the message
- `all_caps` – If it uses a lot of ALL CAPS
- `contains_money` – Mentions of money or discounts
- `is_sender_unknown` – If it’s from an unfamiliar address
- `has_attachment` – If the email has an attachment


## Files in this repo

- `spam_detector.py` – The complete neural network code
- `README.md` – This file
- `.gitignore` – Ignores Python cache files, etc.
- `LICENSE` – MIT license (open to use)


## How to run it

Make sure you have Python and NumPy installed:

pip install numpy

Then just run the script:

python spam_detector.py

It will train the model and print out predictions for some sample emails. You can also test new email inputs by calling the predict_email() function at the bottom of the file.

## Sample output

Email 1 prediction: 0.1034 -> Not Spam  
Email 2 prediction: 0.8987 -> Spam  
...  
New Email Prediction -> 0.9321 -> Spam

## Author
Sudhiksha Narayanaraopeta
