# ERIC
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
