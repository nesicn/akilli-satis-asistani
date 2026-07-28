import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Scikit-learn versiyon uyumluluk yaması (Gerekirse)
import sklearn.compose._column_transformer as ct
if not hasattr(ct, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    ct._RemainderColsList = _RemainderColsList

st.set_page_config(
    page_title="Akıllı Kasa Asistanı",
    page_icon="🛍️",
    layout="wide"
)

# 1. Model ve Veri Yükleme
@st.cache_resource
def load_model_and_data():
    with open('smart_kasa_model.pkl', 'rb') as f:
        model = pickle.load(f)
    df = pd.read_csv('musteri_davranis_seti.csv')
    return model, df

try:
    model, df = load_model_and_data()
except Exception as e:
    st.error(f"Dosyalar yüklenirken hata oluştu: {e}")
    st.stop()

# Başlık ve Açıklama
st.title("🛍️ Akıllı Kasa Asistanı (Smart Checkout AI)")
st.caption("Kasa anında kişiselleştirilmiş çapraz satış ve teklif kabul olasılığı tahmin sistemi")

# Sekme Yapısı
tab1, tab2 = st.tabs(["🎯 Kasa Anı Canlı Tahmin", "📊 Veri ve Model Analitiği"])

with tab1:
    st.subheader("Müşteri Seçimi ve Teklif Tahmini")
    
    # Müşteri Seçimi
    customer_list = df['Musteri_ID'] + " - " + df['Ad_Soyad']
    selected_customer_str = st.selectbox("İşlem Yapılacak Müşteriyi Seçin:", customer_list)
    selected_id = selected_customer_str.split(" - ")[0]
    
    # Müşteri Verisini Çekme
    cust_data = df[df['Musteri_ID'] == selected_id].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Müşteri Adı", cust_data['Ad_Soyad'])
        st.write(f"**Cinsiyet / Yaş:** {cust_data['Cinsiyet']} | {cust_data['Yas_Grubu']}")
        st.write(f"**Sadakat Puanı:** {cust_data['Sadakat_Puani']}")
    with col2:
        st.metric("Mevcut Sepet Tutarı", f"{cust_data['Sepet_Tutari_TL']:.2f} TL")
        st.write(f"**Mağaza / Zaman:** {cust_data['Magaza_Tipi']} | {cust_data['Islem_Saati_Dilimi']}")
        st.write(f"**Promosyon Hassasiyeti:** %{cust_data['Promosyon_Hassasiyeti_Skoru']*100:.0f}")
    with col3:
        st.subheader("Önerilen Ürün")
        st.success(f"**{cust_data['Onerilen_Kategori']}**")
        st.info(f"👉 **{cust_data['Onerilen_Urun']}**")

    st.markdown("---")
    
    # Tahmin Modeli Girdisi Hazırlığı
    feature_cols = [c for c in df.columns if c not in ['Musteri_ID', 'Ad_Soyad', 'Teklif_Kabul']]
    input_df = pd.DataFrame([cust_data[feature_cols]])
    
    # Model Tahmini
    if st.button("🚀 Kasa Anında Teklif Kabul Olasılığını Hesapla", type="primary"):
        try:
            proba = model.predict_proba(input_df)[0][1]
            pred = model.predict(input_df)[0]
            
            st.subheader("Tahmin Sonucu")
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric("Teklif Kabul Olasılığı", f"%{proba*100:.1f}")
                st.progress(float(proba))
            
            with res_col2:
                if proba >= 0.5:
                    st.success("✅ **TEKLİFİ SUNUN:** Müşterinin bu öneriyi kabul etme olasılığı yüksek!")
                else:
                    st.warning("⚠️ **TEKLİFİ DEĞİŞTİRİN:** Müşterinin bu öneriyi kabul etme olasılığı düşük.")
                    
        except Exception as e:
            st.error(f"Tahmin yapılırken bir hatayla karşılaşıldı: {e}")

with tab2:
    st.subheader("Genel Veri Seti Metrikleri")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Toplam Kayıt", len(df))
    m2.metric("Ortalama Sepet Tutarı", f"{df['Sepet_Tutari_TL'].mean():.2f} TL")
    m3.metric("Genel Teklif Kabul Oranı", f"%{df['Teklif_Kabul'].mean()*100:.1f}")
    m4.metric("Kategori Sayısı", df['Onerilen_Kategori'].nunique())
    
    st.markdown("---")
    st.write("### Kategori Bazlı Teklif Kabul Dağılımı")
    cat_summary = df.groupby('Onerilen_Kategori')['Teklif_Kabul'].agg(['count', 'mean']).reset_index()
    cat_summary.columns = ['Önerilen Kategori', 'Toplam Teklif Sayısı', 'Kabul Oranı']
    cat_summary['Kabul Oranı'] = (cat_summary['Kabul Oranı'] * 100).round(1).astype(str) + '%'
    st.dataframe(cat_summary, use_container_width=True)
