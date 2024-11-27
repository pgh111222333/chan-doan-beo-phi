import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd

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
               'Phương tiện công cộng': 'Public_Transportation', 'Đi bộ': 'Walking'},
    'TUE': {'Hiếm khi': 'Rarely', 'Thỉnh thoảng': 'Sometimes', 'Thường xuyên': 'Frequently', 'Luôn luôn': 'Always'},
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

def on_predict_button_click():
    try:
        input_features = [
            translation_mapping['Gender'][gender_var.get()],
            int(age_var.get()),
            float(height_var.get()),
            float(weight_var.get()),
            translation_mapping['family_history_with_overweight'][family_history_var.get()],
            translation_mapping['FAVC'][favc_var.get()],
            int(fcvc_var.get()),
            int(ncp_var.get()),
            translation_mapping['CAEC'][caec_var.get()],
            translation_mapping['SMOKE'][smoke_var.get()],
            int(ch2o_var.get()),
            translation_mapping['SCC'][scc_var.get()],
            int(faf_var.get()),
            translation_mapping['TUE'][tue_var.get()],
            translation_mapping['CALC'][calc_var.get()],
            translation_mapping['MTRANS'][mtrans_var.get()]
        ]
        
        result = predict_obesity(input_features)
        result_label.config(text=f"🔍 Tỉ lệ bị béo phì: {result:.2f}%")
        progress_var.set(result / 100)

        # Provide advice
        advice = provide_advice(result)
        advice_label.config(text=f"💡 Lời khuyên: {advice}")
    except ValueError as e:
        messagebox.showerror("Lỗi", f"Lỗi: {e}")

# Set up the main window
root = tk.Tk()
root.title("Ứng dụng Dự Đoán Béo Phì 🌟")
root.geometry("600x600")

# Personal Information Frame
personal_frame = ttk.LabelFrame(root, text="Thông tin cá nhân", padding="10")
personal_frame.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

gender_var = tk.StringVar()
gender_label = ttk.Label(personal_frame, text="Giới tính")
gender_label.grid(row=0, column=0, sticky="w")
gender_menu = ttk.OptionMenu(personal_frame, gender_var, *translation_mapping['Gender'].keys())
gender_menu.grid(row=0, column=1)

age_var = tk.StringVar()
age_label = ttk.Label(personal_frame, text="Tuổi")
age_label.grid(row=1, column=0, sticky="w")
age_entry = ttk.Entry(personal_frame, textvariable=age_var)
age_entry.grid(row=1, column=1)

height_var = tk.StringVar()
height_label = ttk.Label(personal_frame, text="Chiều cao (m)")
height_label.grid(row=2, column=0, sticky="w")
height_entry = ttk.Entry(personal_frame, textvariable=height_var)
height_entry.grid(row=2, column=1)

weight_var = tk.StringVar()
weight_label = ttk.Label(personal_frame, text="Cân nặng (kg)")
weight_label.grid(row=3, column=0, sticky="w")
weight_entry = ttk.Entry(personal_frame, textvariable=weight_var)
weight_entry.grid(row=3, column=1)

family_history_var = tk.StringVar()
family_history_label = ttk.Label(personal_frame, text="Gia đình có người bị béo phì")
family_history_label.grid(row=4, column=0, sticky="w")
family_history_menu = ttk.OptionMenu(personal_frame, family_history_var, *translation_mapping['family_history_with_overweight'].keys())
family_history_menu.grid(row=4, column=1)

# Daily Habits Frame
habits_frame = ttk.LabelFrame(root, text="Thói quen hằng ngày", padding="10")
habits_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

favc_var = tk.StringVar()
favc_label = ttk.Label(habits_frame, text="Thường xuyên sử dụng thực phẩm có Calo cao")
favc_label.grid(row=0, column=0, sticky="w")
favc_menu = ttk.OptionMenu(habits_frame, favc_var, *translation_mapping['FAVC'].keys())
favc_menu.grid(row=0, column=1)

fcvc_var = tk.StringVar()
fcvc_label = ttk.Label(habits_frame, text="Tần suất ăn trái cây (1-3)")
fcvc_label.grid(row=1, column=0, sticky="w")
fcvc_entry = ttk.Entry(habits_frame, textvariable=fcvc_var)
fcvc_entry.grid(row=1, column=1)

ncp_var = tk.StringVar()
ncp_label = ttk.Label(habits_frame, text="Số bữa ăn mỗi ngày (1-4)")
ncp_label.grid(row=2, column=0, sticky="w")
ncp_entry = ttk.Entry(habits_frame, textvariable=ncp_var)
ncp_entry.grid(row=2, column=1)

caec_var = tk.StringVar()
caec_label = ttk.Label(habits_frame, text="Tiêu thụ thực phẩm giữa các bữa ăn")
caec_label.grid(row=3, column=0, sticky="w")
caec_menu = ttk.OptionMenu(habits_frame, caec_var, *translation_mapping['CAEC'].keys())
caec_menu.grid(row=3, column=1)

# Additional Factors Frame
factors_frame = ttk.LabelFrame(root, text="Các yếu tố khác", padding="10")
factors_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")

smoke_var = tk.StringVar()
smoke_label = ttk.Label(factors_frame, text="Hút thuốc")
smoke_label.grid(row=0, column=0, sticky="w")
smoke_menu = ttk.OptionMenu(factors_frame, smoke_var, *translation_mapping['SMOKE'].keys())
smoke_menu.grid(row=0, column=1)

ch2o_var = tk.StringVar()
ch2o_label = ttk.Label(factors_frame, text="Số lượng nước uống hàng ngày")
ch2o_label.grid(row=1, column=0, sticky="w")
ch2o_entry = ttk.Entry(factors_frame, textvariable=ch2o_var)
ch2o_entry.grid(row=1, column=1)

scc_var = tk.StringVar()
scc_label = ttk.Label(factors_frame, text="Có tham gia các chương trình giảm béo?")
scc_label.grid(row=2, column=0, sticky="w")
scc_menu = ttk.OptionMenu(factors_frame, scc_var, *translation_mapping['SCC'].keys())
scc_menu.grid(row=2, column=1)

faf_var = tk.StringVar()
faf_label = ttk.Label(factors_frame, text="Mức độ hoạt động thể chất")
faf_label.grid(row=3, column=0, sticky="w")
faf_entry = ttk.Entry(factors_frame, textvariable=faf_var)
faf_entry.grid(row=3, column=1)

tue_var = tk.StringVar()
tue_label = ttk.Label(factors_frame, text="Mức độ căng thẳng")
tue_label.grid(row=4, column=0, sticky="w")
tue_menu = ttk.OptionMenu(factors_frame, tue_var, *translation_mapping['TUE'].keys())
tue_menu.grid(row=4, column=1)

calc_var = tk.StringVar()
calc_label = ttk.Label(factors_frame, text="Tiêu thụ thức ăn vặt")
calc_label.grid(row=5, column=0, sticky="w")
calc_menu = ttk.OptionMenu(factors_frame, calc_var, *translation_mapping['CALC'].keys())
calc_menu.grid(row=5, column=1)

mtrans_var = tk.StringVar()
mtrans_label = ttk.Label(factors_frame, text="Phương tiện di chuyển")
mtrans_label.grid(row=6, column=0, sticky="w")
mtrans_menu = ttk.OptionMenu(factors_frame, mtrans_var, *translation_mapping['MTRANS'].keys())
mtrans_menu.grid(row=6, column=1)

# Result and Progress
result_label = ttk.Label(root, text="🔍 Tỉ lệ béo phì: Chưa có kết quả")
result_label.grid(row=3, column=0, pady=10)

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

advice_label = ttk.Label(root, text="💡 Lời khuyên: Chưa có kết quả")
advice_label.grid(row=5, column=0, pady=10)

# Predict Button
predict_button = ttk.Button(root, text="Dự đoán", command=on_predict_button_click)
predict_button.grid(row=6, column=0, pady=20)

root.mainloop()
