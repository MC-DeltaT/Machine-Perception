Machine Perception Assignment 2: House Number Recognition
Author: Reece Jones


Project Files:
    data/correct_numbers.py - Correct house numbers for training and validation images, for pipeline evaluation.
    generated/recognition_model.xml - Saved SVM model for digit recognition.
    src/ - Python source code.
        pipeline/
            digit_descriptor.py - Definition of the feature descriptor used for digit recognition.
            number_extract.py - Functions for extracting the house numbers from images.
            pipeline.py - Pipeline class for convenience.
            recognition_model.py - Defines the SVM used for digit recognition.
        eval_pipeline.py - Evaluates the performance of the pipeline.
        eval_recognition_model.py - Evaluates the performance of the digit recognition SVM.
        extract_house_numbers.py - Does the main assignment task.
        recognition_training_data.py - Loads the digit recognition training dataset.
        train_recognition_model.py - Trains and saves the digit recognition SVM.
    task.sh - As required by the assignment specification.
    training_eval.sh - Evaluates the performance of the pipeline on the training images.
    validation_eval.sh - Evaluates the performance of the pipeline on the validation images.
