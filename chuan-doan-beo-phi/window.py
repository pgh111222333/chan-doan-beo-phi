import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import joblib
import pandas as pd

# Khai báo các biến Entry tương ứng cho các trường nhập liệu
ncp_var = tk.DoubleVar()  
caec_var = tk.StringVar()
smoke_var = tk.StringVar()
ch2o_var = tk.DoubleVar()
scc_var = tk.StringVar()
faf_var = tk.DoubleVar()
tue_var = tk.DoubleVar()
calc_var = tk.StringVar()
mtrans_var = tk.StringVar()

# Load model and dependencies
model = joblib.load('D:\\chuan-doan-beo-phi\\model.pkl')
scaler = joblib.load('D:\\chuan-doan-beo-phi\\scaler.pkl')
label_encoders = joblib.load('D:\\chuan-doan-beo-phi\\label_encoders.pkl')

X_columns = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
    'CALC', 'MTRANS'
]

categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 
                       'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

translation_mapping = {
    'Gender': {'Nam': 'Male', 'Nữ': 'Female'},
    'family_history_with_overweight': {'Có': 'yes', 'Không': 'no'},
    'FAVC': {'Có': 'yes', 'Không': 'no'},
    'CAEC': {'Không': 'no', 'Thỉnh thoảng': 'Sometimes', 'Thường xuyên': 'Frequently', 'Luôn luôn': 'Always'},
    'SMOKE': {'Có': 'yes', 'Không': 'no'},
    'SCC': {'Có': 'yes', 'Không': 'no'},
    'CALC': {'Không': 'no', 'Thỉnh thoảng': 'Sometimes', 'Thường xuyên': 'Frequently', 'Luôn luôn': 'Always'},
    'MTRANS': {'Ô tô': 'Automobile', 'Xe máy': 'Motorbike', 'Xe đạp': 'Bike', 
               'Phương tiện công cộng': 'Public_Transportation', 'Đi bộ': 'Walking'}
}

def predict_obesity(input_list):
    if len(input_list) != len(X_columns):
        raise ValueError(f"Expected {len(X_columns)} features, but got {len(input_list)}")

    input_dict = {col: [val] for col, val in zip(X_columns, input_list)}
    input_df = pd.DataFrame(input_dict)

    for col in categorical_columns:
        if col in input_df.columns:
            input_df[col] = label_encoders[col].transform(input_df[col])

    input_normalized = scaler.transform(input_df)
    prediction_proba = model.predict_proba(input_normalized)
    obesity_indices = [2, 3, 4]
    obesity_probability = sum(prediction_proba[0][index] for index in obesity_indices) * 100

    return obesity_probability

def provide_advice(probability):
    if probability < 20:
        return "Bạn có nguy cơ béo phì rất thấp. Hãy duy trì lối sống lành mạnh, tập thể dục thường xuyên và ăn uống cân đối!"
    elif probability < 50:
        return "Nguy cơ béo phì của bạn ở mức trung bình. Hãy tăng cường hoạt động thể chất, hạn chế thức ăn nhiều đường và calo!"
    elif probability < 80:
        return "Nguy cơ béo phì của bạn cao. Hãy cân nhắc chế độ ăn lành mạnh, giảm lượng calo dư thừa, và tập thể dục thường xuyên!"
    else:
        return "Nguy cơ béo phì rất cao. Nên gặp chuyên gia dinh dưỡng để được tư vấn cụ thể, đồng thời thay đổi ngay lối sống!"

# Create main window
root = tk.Tk()
root.title("Ứng dụng Dự Đoán Béo Phì 🌟")
root.geometry("500x600")
root.config(bg="#f5f5f5")

# Add a heading label
heading = tk.Label(root, text="Chào mừng đến với ứng dụng Dự đoán Béo phì", font=("Helvetica", 16, "bold"), bg="#f5f5f5")
heading.pack(pady=20)

# Create form frame
form_frame = tk.Frame(root, bg="#f5f5f5")
form_frame.pack(pady=20)

# Gender field
gender_var = tk.StringVar()
gender_label = tk.Label(form_frame, text="Giới tính:", font=("Helvetica", 12), bg="#f5f5f5")
gender_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
gender_menu = ttk.Combobox(form_frame, textvariable=gender_var, values=list(translation_mapping['Gender'].keys()), state="readonly")
gender_menu.grid(row=0, column=1, padx=10, pady=5)

# Age field (Entry widget)
age_var = tk.IntVar(value=25)
age_label = tk.Label(form_frame, text="Tuổi:", font=("Helvetica", 12), bg="#f5f5f5")
age_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
age_entry = tk.Entry(form_frame, textvariable=age_var, font=("Helvetica", 12))
age_entry.grid(row=1, column=1, padx=10, pady=5)

# Height field (Entry widget)
height_var = tk.DoubleVar(value=1.7)
height_label = tk.Label(form_frame, text="Chiều cao (m):", font=("Helvetica", 12), bg="#f5f5f5")
height_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
height_entry = tk.Entry(form_frame, textvariable=height_var, font=("Helvetica", 12))
height_entry.grid(row=2, column=1, padx=10, pady=5)

# Weight field (Entry widget)
weight_var = tk.IntVar(value=70)
weight_label = tk.Label(form_frame, text="Cân nặng (kg):", font=("Helvetica", 12), bg="#f5f5f5")
weight_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
weight_entry = tk.Entry(form_frame, textvariable=weight_var, font=("Helvetica", 12))
weight_entry.grid(row=3, column=1, padx=10, pady=5)

# Add more fields (Family history, FAVC, etc.) in a similar way...

# Add a predict button
predict_button = tk.Button(root, text="Dự đoán Béo Phì", font=("Helvetica", 14, "bold"), command= on_predict)
predict_button.pack(pady=20)

# Start the application
root.mainloop()
