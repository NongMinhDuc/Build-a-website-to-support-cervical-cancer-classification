import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load model
model_load = tf.keras.models.load_model('Xception.keras')

# Ensure session state initialization
if 'history' not in st.session_state:
    st.session_state.history = []

# Mapping labels
label_map = {
    0: "ASC_H",
    1: "ASC_US",
    2: "HSIL",
    3: "LSIL",
    4: "SCC"
}

# Function to predict image class
def predict_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.image.resize(img_array, [224, 224])
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  
    predictions = model_load.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

# Save history
def save_to_history(image, predicted_class, note):
    st.session_state.history.append({'Ảnh': image.name, 'Kết quả': predicted_class, 'Ghi chú': note})

# Function to display history in sidebar
def display_history():
    #st.sidebar.title("Lịch sử tra cứu ảnh")
    logo = Image.open("G:/document/UDCNPTPM/Phân loại ung thư/avata.jpg")  
    st.sidebar.image(logo, width=100)
    st.sidebar.markdown("### Tên nhóm: NHÓM 3 XÂY DỰNG WEB HỖ TRỢ PHÂN LOẠI UNG THƯ CỔ TỬ CUNG")
    st.sidebar.markdown("1. Nguyễn Thái Học")
    st.sidebar.markdown("2. Lý Quốc Huy")
    st.sidebar.markdown("3. NÔng Minh Đức")
    st.sidebar.markdown("4. Lê Duy Quang")
   
    st.sidebar.title("Lịch sử tra cứu ảnh")
    history = st.session_state.get('history', [])
    for idx, entry in enumerate(reversed(history[-10:])):
        label = label_map.get(entry['Kết quả'], 'Unknown')
        st.sidebar.markdown(f"**[+] Ảnh số : {idx+1}**")
        st.sidebar.write(f"Tên ảnh: {entry['Ảnh']}")
        st.sidebar.write(f"Kết quả: {label}")
        st.sidebar.write(f"Ghi chú: {entry['Ghi chú']}")
        st.sidebar.markdown("---")

# Main Streamlit app
st.title("Phân loại ảnh")

upload_file = st.file_uploader("Chọn ảnh tải lên", type=["jpg", "png"])

if upload_file is not None:
    image = Image.open(upload_file)
    st.image(image, caption='Ảnh đã tải lên.', width=100)
    st.write("Kích thước ảnh: ", image.size)

    # Predict image
    st.write("Đang phân loại ảnh...")
    try:
        predicted_class = predict_image(image)
        label = label_map.get(predicted_class, 'Unknown')
        st.write(f"Ảnh thuộc lớp: {label}")
    except Exception as e:
        st.error(f"Lỗi trong quá trình dự đoán. {e}")

    # Note
    note = st.text_area("Ghi chú")
    st.write("Ghi chú của bạn:", note)

    # Save to history
    if st.button('Lưu kết quả'):
        save_to_history(upload_file, predicted_class, note)
        st.success('Đã lưu kết quả thành công!')

# Hiển thị ra màn hình
display_history()
