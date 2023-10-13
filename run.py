from eric_core import Eric
import os

if __name__ == "__main__":
    # Is there a saved, trained model?
    existingModel = os.path.exists('./model')

    # Instantiate ERIC and train if no model exists
    eric = Eric(training=not existingModel)

    # List of values to test
    testEmails = [
        "jack@jackwhitworth.com",
        "hello, world!"
    ]

    # Test the list and save results
    predictions = eric.predict(testEmails)
    
    # Output the results in a human-friendly way
    for i in range(0, len(predictions)):
        if predictions[i]:
            print(f"{testEmails[i]} is an email.")
        else:
            print(f"{testEmails[i]} is not an email.")