from core import Eric

if __name__ == "__main__":
    eric = Eric(
        training=True,
        dataPath="data/eric_standard.csv"
    )

    while True:
        userInput = input("Potential Email: ")
        
        if userInput.lower == 'quit':
            break
        
        prediction = eric.predict(userInput)
        print(f"This is {'' if prediction else 'not '}email address")
        