import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="Smart Kasa Asistanı | Akıllı Çapraz Satış",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODEL VE VERİ SETİ YÜKLEME ---
@st.cache_resource
def load_model():
    model_path = "smart_kasa_model.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Model yüklenirken hata oluştu: {e}")
            return None
    else:
        st.error("Model dosyası ('smart_kasa_model.pkl') bulunamadı.")
        return None

@st.cache_data
def load_dataset():
    data_path = "musteri-davranis-seti.csv"
    if os.path.exists(data_path):
        try:
            return pd.read_csv(data_path)
        except Exception as e:
            st.error(f"Veri seti yüklenirken hata oluştu: {e}")
            return None
    else:
        st.error("Veri seti dosyası ('musteri-davranis-seti.csv') bulunamadı.")
        return None

model = load_model()
df_raw = load_dataset()

# --- BAŞLIK VE AÇIKLAMA ---
st.title("🛍️ Smart Kasa Asistanı")
st.markdown("""
Bu uygulama, kasa anında müşterinin demografik verilerini, geçmiş alışveriş davranışlarını ve sepet içeriğini analiz ederek 
**en yüksek kabul edilme olasılığına sahip çapraz satış (öneri) ürününü** tahmin eder.
""")

st.divider()

# --- YAN MENÜ (SIDEBAR) / INPUT ALANLARI ---
st.sidebar.header("📋 Müşteri & İşlem Bilgileri")

# Demografik ve Mağaza Bilgileri
cinsiyet = st.sidebar.selectbox("Cinsiyet", ["Kadın", "Erkek"])
yas_grubu = st.sidebar.selectbox("Yaş Grubu", ["18-24", "25-34", "35-49", "50+"])
magaza_tipi = st.sidebar.selectbox("Mağaza Tipi", ["Cadde", "AVM"])
mevsim = st.sidebar.selectbox("Mevsim", ["İlkalbahar", "Yaz", "Sonbahar", "Kış"])
islem_saati = st.sidebar.selectbox("İşlem Saati Dilimi", ["Sabah", "Öğle", "Akşam", "Gece"])
hafta_durumu = st.sidebar.selectbox("Hafta İçi / Hafta Sonu", ["Hafta İçi", "Hafta Sonu"])

st.sidebar.divider()
st.sidebar.header("📊 Genel Alışveriş Skorları")

sadakat_puani = st.sidebar.slider("Sadakat Puanı (0-100)", 0.0, 100.0, 50.0)
promosyon_skoru = st.sidebar.slider("Promosyon Hassasiyeti Skoru (0-1)", 0.0, 1.0, 0.5)
gecen_gun = st.sidebar.number_input("Son Alışverişten Geçen Gün Sayısı", min_value=0, max_value=365, value=15)
coklu_kategori_skoru = st.sidebar.slider("Çoklu Kategori Alım Skoru (0-1)", 0.0, 1.0, 0.5)
sepet_tutari = st.sidebar.number_input("Sepet Tutarı (TL)", min_value=0.0, max_value=10000.0, value=250.0, step=10.0)

st.sidebar.divider()
st.sidebar.header("🎁 Önerilecek Ürün Bilgisi")

# Veri setinden Önerilen Kategori ve Ürün opsiyonlarını alma
kategori_listesi = [
    "Aksesuar & Güzellik Aletleri", "Cilt Bakımı", "El & Ayak Bakımı", 
    "Erkek Bakım", "Güneş & Bronzlaşma", "Makyaj", "Parfüm", 
    "Saç Bakımı", "Vücut Bakımı & Banyo"
]
onerilen_kategori = st.sidebar.selectbox("Önerilen Kategori", kategori_listesi)

# Kategoriye göre ürün seçimi
urun_haritasi = {
    "Makyaj": ["Allık", "Aydınlatıcı", "BB/CC Krem", "Bronzer", "Dudak Parlatıcısı", "Eyeliner", "Fondöten", "Göz Farı", "Kapatıcı", "Maskara", "Oje", "Pudra", "Ruj"],
    "Cilt Bakımı": ["Aloe Vera Jeli", "Cilt Serumu", "Göz Çevresi Kremi", "Kağıt Maske", "Misel Su", "Temizleme Jeli", "Tonik", "Yüz Güneş Kremi", "Yüz Kremi"],
    "El & Ayak Bakımı": ["Aseton", "Ayak Kremi", "Ayak Törpüsü", "El Kremi"],
    "Erkek Bakım": ["Erkek Deodorant", "Sakal Yağı", "Tıraş Bıçağı", "Tıraş Köpüğü", "Tıraş Sonrası Losyon"],
    "Güneş & Bronzlaşma": ["Bronzlaştırıcı Yağ", "Vücut Güneş Kremi"],
    "Parfüm": ["EDP Parfüm", "EDT Parfüm", "Vücut Spreyi (Body Mist)"],
    "Saç Bakımı": ["Kuru Şampuan", "Saç Bakım Yağı", "Saç Kremi", "Saç Maskesi", "Şampuan"],
    "Vücut Bakımı & Banyo": ["Banyo Bombası", "Duş Jeli", "Vücut Losyonu", "Vücut Peelingi"],
    "Aksesuar & Güzellik Aletleri": ["Cımbız", "Gua Sha Taşı", "Makyaj Fırçası", "Makyaj Pamuğu", "Makyaj Süngeri"]
}

mevcut_urunler = urun_haritasi.get(onerilen_kategori, ["Genel Ürün"])
onerilen_urun = st.sidebar.selectbox("Önerilen Ürün", mevcut_urunler)

# --- ANA EKRAN VE TAHMİN ALANI ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💡 Çapraz Satış Öneri Tahmini")
    st.write("Seçilen müşteri profili ve önerilen ürün kombinasyonuna göre satın alma olasılığını hesaplamak için aşağıdaki butona basınız.")

    if st.button("🚀 Öneri Başarısını Tahmin Et", use_container_width=True):
        if model is not None:
            # Modelin beklediği tüm 53 özelliği sözlük olarak oluşturma
            input_dict = {
                # Kategorik Değişkenler
                "Cinsiyet": cinsiyet,
                "Yas_Grubu": yas_grubu,
                "Magaza_Tipi": magaza_tipi,
                "Mevsim": mevsim,
                "Islem_Saati_Dilimi": islem_saati,
                "Hafta_Ici_Hafta_Sonu": hafta_durumu,
                "Onerilen_Kategori": onerilen_kategori,
                "Onerilen_Urun": onerilen_urun,

                # Genel Sayısal Değişkenler
                "Sadakat_Puani": sadakat_puani,
                "Promosyon_Hassasiyeti_Skoru": promosyon_skoru,
                "Gecen_Gun_Sayisi": gecen_gun,
                "Coklu_Kategori_Alim_Skoru": coklu_kategori_skoru,
                "Sepet_Tutari_TL": sepet_tutari,

                # Kategori Bazlı Detay Değişkenler (Varsayılan nötr değerlerle dolduruldu)
                "Makyaj_Alisveris_Sayisi": 1, "Gecen_Gun_Makyaj": 15, "Makyaj_Ort_Alim_Araligi": 20, "Makyaj_Tuketim_Orani": 0.5,
                "Cilt_Bakimi_Alisveris_Sayisi": 1, "Gecen_Gun_Cilt_Bakimi": 15, "Cilt_Bakimi_Ort_Alim_Araligi": 20, "Cilt_Bakimi_Tuketim_Orani": 0.5,
                "Agiz_Bakimi_Alisveris_Sayisi": 1, "Gecen_Gun_Agiz_Bakimi": 15, "Agiz_Bakimi_Ort_Alim_Araligi": 20, "Agiz_Bakimi_Tuketim_Orani": 0.5,
                "Parfum_Alisveris_Sayisi": 1, "Gecen_Gun_Parfum": 15, "Parfum_Ort_Alim_Araligi": 20, "Parfum_Tuketim_Orani": 0.5,
                "Sac_Bakimi_Alisveris_Sayisi": 1, "Gecen_Gun_Sac_Bakimi": 15, "Sac_Bakimi_Ort_Alim_Araligi": 20, "Sac_Bakimi_Tuketim_Orani": 0.5,
                "Vucut_Bakimi_Banyo_Alisveris_Sayisi": 1, "Gecen_Gun_Vucut_Bakimi_Banyo": 15, "Vucut_Bakimi_Banyo_Ort_Alim_Araligi": 20, "Vucut_Bakimi_Banyo_Tuketim_Orani": 0.5,
                "El_Ayak_Bakimi_Alisveris_Sayisi": 1, "Gecen_Gun_El_Ayak_Bakimi": 15, "El_Ayak_Bakimi_Ort_Alim_Araligi": 20, "El_Ayak_Bakimi_Tuketim_Orani": 0.5,
                "Gunes_Bronzlasma_Alisveris_Sayisi": 1, "Gecen_Gun_Gunes_Bronzlasma": 15, "Gunes_Bronzlasma_Ort_Alim_Araligi": 20, "Gunes_Bronzlasma_Tuketim_Orani": 0.5,
                "Aksesuar_Guzellik_Aletleri_Alisveris_Sayisi": 1, "Gecen_Gun_Aksesuar_Guzellik_Aletleri": 15, "Aksesuar_Guzellik_Aletleri_Ort_Alim_Araligi": 20, "Aksesuar_Guzellik_Aletleri_Tuketim_Orani": 0.5,
                "Erkek_Bakim_Alisveris_Sayisi": 1, "Gecen_Gun_Erkek_Bakim": 15, "Erkek_Bakim_Ort_Alim_Araligi": 20, "Erkek_Bakim_Tuketim_Orani": 0.5
            }

            input_df = pd.DataFrame([input_dict])

            try:
                # Olasılık Tahmini
                prediction_proba = model.predict_proba(input_df)[0][1]
                prediction_class = model.predict(input_df)[0]

                st.markdown("---")
                st.metric("Öneriyi Kabul Etme Olasılığı", f"%{prediction_proba * 100:.1f}")

                if prediction_class == 1 or prediction_proba >= 0.5:
                    st.success(f"✅ **Tavsiye Edilir:** Müşterinin **{onerilen_urun}** teklifini kabul etme olasılığı yüksek!")
                else:
                    st.warning(f"⚠️ **Tavsiye Edilmez:** Müşterinin **{onerilen_urun}** teklifini kabul etme olasılığı düşük.")

            except Exception as e:
                st.error(f"Tahmin yapılırken hata oluştu: {e}")

with col2:
    st.subheader("📌 Özet Bilgiler")
    st.info(f"""
    **Seçilen Profil:**
    - **Cinsiyet / Yaş:** {cinsiyet}, {yas_grubu}
    - **Konum / Zaman:** {magaza_tipi}, {islem_saati}
    - **Sepet Tutarı:** {sepet_tutari} TL
    - **Hedef Ürün:** {onerilen_urun} ({onerilen_kategori})
    """)
