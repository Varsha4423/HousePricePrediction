import numpy as np 
import pickle
import stremlit as st 

with open('Model','rb') as f:
    model=pickle.load(f)

with open('Scaler','rb') as f:
    scaler=pickle.load(f)    


location_mapping = {0: 'AGIRIPALLI',
 1: 'Ajit Singh Nagar',
 2: 'Andhra Prabha Colony Road',
 3: 'Ashok Nagar',
 4: 'Auto Nagar',
 5: 'Ayyappa Nagar',
 6: 'Bandar Road',
 7: 'Benz Circle',
 8: 'Bharathi Nagar',
 9: 'Bhavanipuram',
 10: 'Chennai Vijayawada Highway',
 11: 'Currency Nagar',
 12: 'Devi Nagar',
 13: 'Edupugallu',
 14: 'Enikepadu',
 15: 'G Konduru',
 16: 'Gandhi Nagar',
 17: 'Gannavaram',
 18: 'Gollapudi',
 19: 'Gollapudi1',
 20: 'Gosala',
 21: 'Governor Peta',
 22: 'Gudavalli',
 23: 'Gunadala',
 24: 'Guntupalli',
 25: 'Guru Nanak Colony',
 26: 'Ibrahimpatnam',
 27: 'Jaggayyapet',
 28: 'Kanchikacherla',
 29: 'Kandrika',
 30: 'Kanigiri Gurunadham Street',
 31: 'Kankipadu',
 32: 'Kanuru',
 33: 'Kesarapalle',
 34: 'Kesarapalli',
 35: 'LIC Colony',
 36: 'Labbipet',
 37: 'Madhuranagar',
 38: 'Mangalagiri',
 39: 'Milk Factory Road',
 40: 'Moghalrajpuram',
 41: 'Mylavaram',
 42: 'MylavaramKuntamukkalaVellaturuVijayawada Road',
 43: 'Nandigama',
 44: 'Nidamanuru',
 45: 'Nunna',
 46: 'Nuzividu',
 47: 'Nuzvid Road',
 48: 'Nuzvid To Vijayawada Road',
 49: 'PNT Colony',
 50: 'Pamarru',
 51: 'Patamata',
 52: 'Payakapuram',
 53: 'Pedapulipaka Tadigadapa Road',
 54: 'Penamaluru',
 55: 'Poranki',
 56: 'Punadipadu',
 57: 'Rajiv Bhargav Colony',
 58: 'Rama Krishna Puram',
 59: 'Ramalingeswara Nagar',
 60: 'Ramavarapadu',
 61: 'Ramavarapadu Ring',
 62: 'SURAMPALLI',
 63: 'Satyanarayanapuram Main Road',
 64: 'Satyaranayana Puram',
 65: 'Sri Ramachandra Nagar',
 66: 'Srinivasa Nagar Bank Colony',
 67: 'Subba Rao Colony 2nd Cross Road',
 68: 'Tadepalligudem',
 69: 'Tadigadapa',
 70: 'Tadigadapa Donka Road',
 71: 'Tarapet',
 72: 'Telaprolu',
 73: 'Tulasi Nagar',
 74: 'Vaddeswaram',
 75: 'Vidhyadharpuram',
 76: 'Vijayawada Airport Road',
 77: 'Vijayawada Guntur Highway',
 78: 'Vijayawada Nuzvidu Road',
 79: 'Vijayawada Road',
 80: 'Vuyyuru',
 81: 'chinnakakani',
 82: 'currency nagar',
 83: 'krishnalanka',
 84: 'kunchanapalli',
 85: 'ramavarappadu'} 

 status_mapping = {0: 'New', 1: 'Ready to move', 2: 'Resale', 3: 'Under Construction'}  

 facing_mapping = {0: 'East',
 1: 'None',
 2: 'North',
 3: 'NorthEast',
 4: 'NorthWest',
 5: 'South',
 6: 'SouthEast',
 7: 'SouthWest',
 8: 'West'} 

 property_type_mapping = {0: 'Apartment',
 1: 'Independent Floor',
 2: 'Independent House',
 3: 'Residential Plot',
 4: 'Studio Apartment',
 5: 'Villa'}


 #fun for prediction

 def predict(bed,bath,loc,size,status,face,t):
    selected_location = location_mapping[loc]
    selected_status = status_mapping[status]
    selected_facing = facing_mapping[face]
    selected_type = property_type_mapping[t]

    input_data = np array([[bed,bath,selected_location,size,selected_status,selected_facing,selected_type]])

    input_df = scaler.transform(input_data)

    return model.predict(input_df)[0]

if __name__ =='__main__':
    st.header('House Price Prediction')

    bed = st.slider('No of Bedrooms',max_value = 10,min_value = 1,value = 3)
    bath = st.slider('No of Bathrooms',max_value = 10,min_value = 1,value = 3)  
    loc = st.selectbox('selected location', list(location_mapping.value())) 