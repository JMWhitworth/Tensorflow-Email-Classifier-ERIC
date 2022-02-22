# ERIC - Email | Retriever | In | Code
TF Keras Model for binary word classification.




This can be called from a subfolder as a module as follows:

```
import eric

Eric = eric.Eric(
    training=False,
    max_length=max_email_length,
    model_path="eric/model",
    training_data="eric/data/train.csv",
    testing_data="eric/data/test.csv")

print(Eric.predict(["testemail@testingemails.com"])[0])
```

On first run you must set training to True and pass valid CSV data for the training. The pre-included data was gathered via simple webscraping, and is not individually owned by me. If your contact info is on there and you would like it removed please notify me and I'll remove it immediately.

CSV format:
```
label,target
thisisvalid@emails.com,1
notanemail,0
```

Once trained it'll save the model and you can then change training to False and use as required.
