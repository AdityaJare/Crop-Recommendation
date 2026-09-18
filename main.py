import streamlit as st 
import pandas as pd 
import joblib

st.set_page_config(page_title="Crop Recommender", page_icon="🌾")

model=joblib.load("crop_prediction.pkl")

st.title("Crop Recommendation System")
st.write("Enter your soil and climate conditions to get a best crop recommendation suitable for your land and environment.")

#creating UI columns and widgets

col1, col2 = st.columns(2)
with col1:
    n_val = st.slider("Nitrogen Value :", 0, 140, 40)  

    p_val = st.slider("Phosphorus Value :", 5, 145, 40)

    k_val = st.slider("Potassium Value :", 5, 205, 40)
with col2:
    temperature = st.slider("Temperature :", 8, 45, 27)

    humidity = st.slider("Humidity : ", 12, 100, 45)

    ph_value = st.slider("pH value :", 3, 10, 6)


rainfall = st.slider("Rainfall : ",20, 300, 100)

input_values=pd.DataFrame(
    [[n_val, p_val, k_val, temperature, humidity, ph_value, rainfall]],
    columns=["N","P","K","temperature","humidity","ph","rainfall"]
    )
 
col3, col4 = st.columns(2)

crop_images= {
    "rice":"https://images.unsplash.com/photo-1651981350249-6173caeeb660?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "maize":"https://images.unsplash.com/photo-1554402100-8d1d9f3dff80?q=80&w=687&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "chickpea":"https://plus.unsplash.com/premium_photo-1675237624857-7d995e29897d?q=80&w=687&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    'kidneybeans':"https://media.istockphoto.com/id/541972824/photo/red-bean-on-rural-cookware.jpg?s=2048x2048&w=is&k=20&c=OpULNERAZYq-8NBJdY_PuFUqnacgjWRRD82YdlYUif0=",
    'pigeonpeas':"https://plus.unsplash.com/premium_photo-1663844169236-ff32474d1dc8?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NXx8cGlnZW9ucGVhc3xlbnwwfHwwfHx8MA%3D%3D",
    'mothbeans':"https://media.istockphoto.com/id/2160783041/photo/image-of-full-frame-display-of-dried-moth-beans-nutritious-legumes-protein-rich-high-fiber.jpg?s=2048x2048&w=is&k=20&c=YZlfDMQGXcAptv75uGX7iROGDCGje9-JKziivaBWeiI=",
    'mungbean':"https://images.unsplash.com/photo-1594900799266-0e56587ba586?q=80&w=1001&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    'blackgram':"https://media.istockphoto.com/id/2285933191/photo/image-of-black-turtle-beans-with-nuts-and-seeds-dish-healthy-eating-snack-rich-in-plant-based.jpg?s=2048x2048&w=is&k=20&c=N3WNLeTB0lYGJoXtH1UPi8Dhgr3rAwTWEOws_AONIPI=",
    'lentil':"https://plus.unsplash.com/premium_photo-1671130295987-13d3b3b4e9dc?q=80&w=687&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    'pomegranate':"https://images.unsplash.com/photo-1571347586843-69826a524206?q=80&w=1964&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    'banana':"https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?q=80&w=2080&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    'mango':"https://images.unsplash.com/photo-1601493700631-2b16ec4b4716?q=80&w=735&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    'grapes':"https://images.unsplash.com/photo-1537640538966-79f369143f8f?q=80&w=1173&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    'watermelon':"https://images.unsplash.com/photo-1720239278431-bf2a0c838180?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    'muskmelon':"https://images.unsplash.com/photo-1788511192198-7d98514ff1be?q=80&w=735&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    'apple':"https://plus.unsplash.com/premium_photo-1661322640130-f6a1e2c36653?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    'orange':"https://images.unsplash.com/photo-1597714026720-8f74c62310ba?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    'papaya':"https://images.unsplash.com/photo-1581242335635-ce8631489ac5?q=80&w=687&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    'coconut':"https://images.unsplash.com/photo-1537191072641-5e19cc173c6a?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    'cotton':"https://images.unsplash.com/photo-1705147289293-ce4d1a8f5a4c?q=80&w=687&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    'jute':"https://images.unsplash.com/photo-1610428011552-734764c290ee?q=80&w=735&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    'coffee':"https://plus.unsplash.com/premium_photo-1675435644687-562e8042b9db?q=80&w=749&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
}

if st.button("Predict Crop"):
    prediction = model.predict(input_values)
    
    with col3:
        st.success(f"Predicted Crop is {prediction[0]}")

    with col4:
        st.image(crop_images[prediction[0]], width=300)
    

