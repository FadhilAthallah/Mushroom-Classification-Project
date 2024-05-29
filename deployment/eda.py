import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load data
df = pd.read_csv("mushroom_cleaned.csv")

def eda_page():

    st.title("Eksploratory Data Analysis")
    st.write('Analisa data pada dataframe untuk lebih memahami isi dari data')

# Dsitribusi class Jamur

    # Calculate the percentage of each class
    class_counts = df['class'].value_counts()
    total_samples = class_counts.sum()
    edible_percentage = (class_counts[0] / total_samples) * 100
    poisonous_percentage = (class_counts[1] / total_samples) * 100

    # Display the title using Markdown
    st.markdown("<h2><strong>Distribusi Class Jamur</strong></h2>", unsafe_allow_html=True)

    # Create the pie plot
    fig, ax = plt.subplots()
    ax.pie(class_counts, labels=['Edible', 'Poisonous'], colors=['lightgreen', 'red'], autopct='%1.1f%%', startangle=140)
    ax.set_title('Distribution of Mushroom Classes')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # Show the plot in Streamlit
    st.pyplot(fig)

    # Display the explanation
    st.write("**Penjelasan**:")
    st.write(f"Berdasarkan pie plot, persebaran data edible berjumlah {edible_percentage:.1f}% dari total data dan poisonous berjumlah {poisonous_percentage:.1f}% dari total data. Dari sini didapati bahwa jamur yang dapat dikonsumsi berjumlah lebih banyak dari jamur beracun pada dataframe ini.")

# Code untuk class terhadap semua fitur categorical 

    st.markdown("<h2><strong>Hubungan kolom class terhadap semua kolom categorical</strong></h2>", unsafe_allow_html=True)
    st.write("Berdasarkan jumlah unique valuenya, diketahui bahwa kolom `cap_shape`, `gill-attachment`, `gill-color`, `stem-color`, `season`, dan `class` merupakan data ordinal. Pada bagian ini ingin diketahui bagaimana hubungan class terhadap kolom-kolom tersebut")
    # Define categorical features
    categorical = ['cap-shape', 'gill-attachment', 'gill-color', 'stem-color', 'season']

    # Calculate the number of plots needed based on the number of categorical features
    num_plots = len(categorical)

    # Calculate the number of rows and columns for the subplots
    num_rows = (num_plots - 1) // 3 + 1
    num_cols = min(num_plots, 3)

    # Create a figure and axis array for subplots with larger figsize
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 6 * num_rows))

    # Flatten the axes array to simplify iteration
    axes = axes.flatten()

    for i, x in enumerate(categorical):
        # Crosstab and plot
        cross_table = pd.crosstab(df[x], df['class'])
        cross_table.plot(kind='bar', stacked=False, color=['red', 'lightgreen'], ax=axes[i])

        # Add labels and title
        axes[i].set_xlabel(x)
        axes[i].set_ylabel('Count')
        axes[i].set_title(f"Relationship between {x} and class")
        axes[i].legend(title='target', labels=['Poisonous', 'Edible'])  # Custom legend labels

        # Rotate x-axis labels for better readability
        axes[i].tick_params(axis='x', rotation=45)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display plots in Streamlit
    st.pyplot(fig)

    st.write("**Penjelasan**:")
    markdown_text = """
    Berdasarkan hasil dari visualisasi hubungan `class` dengan seluruh kolom kategori didapatkan :

    * `cap-shape` : Berdarkan bentuk cap-nya, tiap bentuk jamur membentuk sebuah pola terhadap class-nya dimana rata-rata bentuk jamur memiliki total jamur edible lebih banyak dari pada jamur yang beracun, Kecuali jamur dengan bentuk 6. Jamur bentuk 6 merupakan jamur terbanyak pada data dimana jumlah jamur beracunnya lebih banyak jika dibandingkan dengan jamur yang edible pada bentuk tersebut. 
    * `gil-attachment` : pada penhubung gil jamur, terbentuk pola yang sama seperti pada cap-shape. Akan tetapi khusus untuk gill pada tipe 4 jamur beracun lebih banyak ketimbang jamur edible
    * `gil-color` : Seluruh warna memiliki jamur edible yang lebih banyak ketimbang jamur poisonous, kecuali pada tiga bentuk warna dimana jamur poisonous lebih banyak. Yaitu pada warna jamur 0, 3, dan 10
    * `stem-color` :Pada stem colour ini terdapat dua data yang menarik. Yang pertama adalah stem dengan warna 6, dimana jamur edible memiliki jumlah yang jauh lebih banyak dibandingkan dengan jamur poisonous. Hal yang sama terdapat juga pada jamur dengan warna 11, dimana jumlah jamur poosonous jauh lebih banyak dibandingkan jamur ediblenya
    * `season` : Pada season 0.02 dan 1.8 jamur cenderung merupakan jamur beracun, akan tetapi pada 0.8 dan 0.94 jamur edible jauh lebih banyak dibandinkan dengan jamur beracun
    * Dari seluruh data ini asumsi sementara adalah bahwa `cap-shape`, `gil-attachment`, `gil-color`, `stem-color`, `season` memiliki korelasi dengan `class` dimana tiap fitur tersebut memiliki sebuah pola terhadap `class`. Untuk korelasi yang lebih tepat sendiri harus dilakukan uji korelasi pada saat proses data engineering nanti agar fitur yang digunakan lebih akurat dalam menghasilkan prediksi pada model
    """
    st.markdown(markdown_text)

# Code hubungan stem height terhadap stem width

    st.markdown("<h2><b>Hubugan `stem-height` terhadap `stem-width`</b></h2>", unsafe_allow_html=True)


    # Scatter plot
    scatter_fig, ax = plt.subplots()
    sns.scatterplot(x='stem-width', y='stem-height', data=df, ax=ax)
    plt.xlabel('stem-width')
    plt.ylabel('stem-height')
    plt.title("Relationship between stem-width and stem-height")

    # Adding regression line
    sns.regplot(x='stem-width', y='stem-height', data=df, scatter=False, color='red', ax=ax)

    # Display plot in Streamlit
    st.pyplot(scatter_fig)
    
    st.write("**Penjelasan**:")
    markdown_text = """
    * Berdasarkan garis regresi yang terbentuk, stem-width dan stem-height memiliki hubungan lurus yang lemah, hal dapat terjadi karena jika sebuah jamur itu sehat jamur tersebut dapat memiliki panjang dan lebar batang yang panjang dan besar
    """
    st.markdown(markdown_text)

# Code hubungan stem height terhadap cap diameter
    st.markdown("<h2><b>Hubugan `stem-height` terhadap `cap-diameter`</b></h2>", unsafe_allow_html=True)

    sns.scatterplot(x='cap-diameter', y='stem-height', data=df)
    plt.xlabel('cap-diameter')
    plt.ylabel('stem-height')
    plt.title("Relationship between cap-diameter and stem-height")

    # Adding regression line
    sns.regplot(x='cap-diameter', y='stem-height', data=df, scatter=False, color='red')

    # Display plot in Streamlit
    st.pyplot(scatter_fig)
    
    st.write("**Penjelasan**:")
    markdown_text = """
    * Berdasarkan garis regresi yang terbentuk, `cap-diameter` dan `stem-height` memiliki hubungan linear yang lemah, menunjukkan bahwa diameter dari topi jamur akan semakin besar ketika tinggi batang jamur semakin tinggi juga     """
    st.markdown(markdown_text)

# code korelasi antar kolom numerikal

    st.markdown("<h2><b>Hubungan seluruh fitur dan target pada dataframe</b></h2>", unsafe_allow_html=True)
    
    # Calculate correlation matrix
    correlation_matrix = df.corr()

    # Create the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

    # Add title
    plt.title('Correlation Heatmap of Numerical Columns')

    # Display plot in Streamlit
    st.pyplot(plt)
        
    st.write("**Penjelasan**:")
    markdown_text = """
    * Rata-rata fitur memiliki hubungan yang lemah dengan fitur lainnya, akan tetapi terdapat korelasi yang kuat yaitu pada `stem-width` dan `cap-diameter`. korelasi ini menunjukkan bahwa panjang dari batang pada jamur memiliki pengaruh terhadap diameter topi dari jamur itu tersendiri    """
    st.markdown(markdown_text)


