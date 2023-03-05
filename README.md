# ERIC - Email | Retriever | In | Code
TF Keras Model for text classification.




This can be called from a subfolder as a module as follows:

```
from core import Eric

if __name__ == "__main__":
    eric = Eric(
        training=True,
        dataPath="data/eric_standard.csv"
    )
    eric.predict("Is this an email?")
    eric.predict("oristhis@example.com")
```
Outputs:
```
0
1
```

On first run you must set training to True and pass valid CSV data for the training. The pre-included data was gathered via simple webscraping, and is not individually owned by me. If your contact info is on there and you would like it removed please notify me and I'll remove it immediately.

CSV format:
```
label,target
thisisvalid@emails.com,1
notanemail,0
```

Once trained it'll save the model and you can then change training to False and use as required.
