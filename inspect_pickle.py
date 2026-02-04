import pickle

try:
    with open("model/fake_news_model.pkl", "rb") as f:
        data = pickle.load(f)
    print(f"Type: {type(data)}")
    if isinstance(data, (list, tuple)):
        print(f"Length: {len(data)}")
        for i, item in enumerate(data):
            print(f"Item {i} type: {type(item)}")
    else:
        print("Data is not a list or tuple.")
except Exception as e:
    print(f"Error: {e}")
