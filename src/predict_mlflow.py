import mlflow

model_name = "fastttext"
version = 1

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{version}"
)

list_libs = ["vendeur d'huitres", "boulanger", "COIFFEUR", "coiffeur, & 98789"]

test_data = {
    "query": list_libs,
    "k": 5
}

results = model.predict(test_data)
print(results)