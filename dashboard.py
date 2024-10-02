import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Fungsi untuk memuat data
@st.cache
def load_data(day_file, hour_file):
    day_data = pd.read_csv(day_file)
    hour_data = pd.read_csv(hour_file)
    return day_data, hour_data

# Fungsi untuk analisis regresi
def regression_analysis(X, y):
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    return model

# Fungsi untuk clustering
def kmeans_clustering(data):
    kmeans = KMeans(n_clusters=3)
    data['cluster'] = kmeans.fit_predict(data[['casual', 'registered']])
    return data

# Fungsi utama untuk Streamlit
def main():
    st.title("Analisis Penggunaan Sepeda Berdasarkan Data Berbagi Sepeda")

    # Load data
    day_data, hour_data = load_data('day.csv', 'hour.csv')
    
    # Menampilkan data mentah jika di centang
    if st.checkbox("Tampilkan data mentah harian"):
        st.write(day_data.head())

    if st.checkbox("Tampilkan data mentah jam-jaman"):
        st.write(hour_data.head())
    
    # Penjelasan awal atau pengantar
    st.subheader("Pengantar")
    st.write("""
    Data ini merupakan data berbagi sepeda yang mencakup penggunaan sepeda secara harian dan jam-jaman. 
    Analisis ini mencakup berbagai aspek, termasuk faktor-faktor yang mempengaruhi penggunaan sepeda, 
    tren penggunaan berdasarkan waktu, serta pengelompokan pengguna dengan algoritma clustering.
    """)

    # Statistik Deskriptif
    st.subheader("Statistik Deskriptif")
    st.write("Data Harian:")
    st.write(day_data[['temp', 'hum', 'windspeed', 'cnt']].describe())
    
    st.write("Data Jam-jaman:")
    st.write(hour_data[['temp', 'hum', 'windspeed', 'cnt']].describe())
    
    # Analisis Regresi
    st.subheader("Analisis Regresi (Model Sederhana)")
    st.write("Model regresi digunakan untuk melihat pengaruh variabel suhu (temp), kelembapan (hum), dan kecepatan angin (windspeed) terhadap jumlah penggunaan sepeda (cnt).")
    
    X_day = day_data[['temp', 'hum', 'windspeed']]
    y_day = day_data['cnt']
    model_day = regression_analysis(X_day, y_day)
    st.write(model_day.summary())

    st.subheader("Analisis Regresi (Model Diperluas)")
    st.write("Model regresi ini menggunakan variabel tambahan, termasuk kondisi cuaca (weathersit), hari kerja (workingday), dan hari libur (holiday).")
    
    X_extended = day_data[['temp', 'hum', 'windspeed', 'weathersit', 'workingday', 'holiday']]
    y_extended = day_data['cnt']
    model_extended = regression_analysis(X_extended, y_extended)
    st.write(model_extended.summary())
    
    # Visualisasi tren penggunaan sepeda
    st.subheader("Tren Penggunaan Sepeda")
    st.write("""
    Berikut adalah visualisasi tren penggunaan sepeda berdasarkan musim, bulan, dan hari dalam seminggu.
    Tren ini memberikan gambaran mengenai kapan penggunaan sepeda meningkat atau menurun.
    """)
    
    season_trend = day_data.groupby('season')['cnt'].mean()
    month_trend = day_data.groupby('mnth')['cnt'].mean()
    weekday_trend = day_data.groupby('weekday')['cnt'].mean()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].bar(season_trend.index, season_trend.values, color='lightblue')
    axes[0].set_title("Average Bike Usage by Season")
    axes[0].set_xticks([1, 2, 3, 4])
    axes[0].set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])
    
    axes[1].bar(month_trend.index, month_trend.values, color='lightgreen')
    axes[1].set_title("Average Bike Usage by Month")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    axes[2].bar(weekday_trend.index, weekday_trend.values, color='lightcoral')
    axes[2].set_title("Average Bike Usage by Weekday")
    axes[2].set_xticks(range(7))
    axes[2].set_xticklabels(['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Clustering
    st.subheader("K-Means Clustering Pengguna Kasual dan Terdaftar")
    st.write("""
    Pengelompokan dilakukan menggunakan algoritma K-Means untuk mengidentifikasi pola penggunaan sepeda oleh pengguna kasual dan terdaftar.
    Hasil clustering memberikan informasi mengenai kelompok pengguna yang sering menggunakan sepeda pada waktu-waktu tertentu.
    """)
    
    data_clustered = kmeans_clustering(day_data)
    fig, ax = plt.subplots()
    sns.scatterplot(x='casual', y='registered', hue='cluster', data=data_clustered, palette='Set1', ax=ax)
    plt.title('K-Means Clustering of Casual and Registered Users')
    st.pyplot(fig)

    # Kesimpulan
    st.subheader("Kesimpulan")
    st.write("""
    Berdasarkan analisis yang dilakukan, dapat disimpulkan bahwa suhu memiliki pengaruh besar dalam meningkatkan penggunaan sepeda, 
    sedangkan kelembapan dan kecepatan angin cenderung menurunkan penggunaan sepeda. Selain itu, tren penggunaan sepeda lebih tinggi 
    pada musim panas dan hari kerja. Pengelompokan menggunakan K-Means menunjukkan adanya tiga kelompok utama pengguna sepeda: 
    pengguna kasual, pengguna terdaftar, dan pengguna dengan volume tinggi pada jam-jam sibuk.
    """)

if __name__ == '__main__':
    main()
