import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


veri = pd.read_csv("skulls3.csv")


X = veri[["skull_length", "skull_width"]]
y = veri["face_height"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


derinlikler = [1, 2, 3, 5, 10]

egitim_hatalari = []
test_hatalari = []


for derinlik in derinlikler:

    model = DecisionTreeRegressor(
        max_depth=derinlik,
        random_state=42
    )

    model.fit(X_train, y_train)

    
    egitim_tahminleri = model.predict(X_train)

    
    test_tahminleri = model.predict(X_test)

    
    egitim_mae = mean_absolute_error(
        y_train, egitim_tahminleri
    )

    test_mae = mean_absolute_error(
        y_test, test_tahminleri
    )

    egitim_hatalari.append(egitim_mae)
    test_hatalari.append(test_mae)

    print("Derinlik:", derinlik)
    print("Eğitim MAE:", round(egitim_mae, 2))
    print("Test MAE:", round(test_mae, 2))
    print("----------------------")
    
plt.plot(
    derinlikler,
    egitim_hatalari,
    marker="o",
    label="Eğitim MAE"
)

plt.plot(
    derinlikler,
    test_hatalari,
    marker="o",
    label="Test MAE"
)

plt.xlabel("Karar ağacı derinliği")
plt.ylabel("MAE")
plt.title("Derinliğe Göre Model Hataları")
plt.legend()
plt.grid(True)
plt.show()