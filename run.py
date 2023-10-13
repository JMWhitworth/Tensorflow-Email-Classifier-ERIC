from eric_core import Eric

if __name__ == "__main__":
    eric = Eric(training=True)
    print(eric.predict(["jack@jackwhitworth.com"]))
