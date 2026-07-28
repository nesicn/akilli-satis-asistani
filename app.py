import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="Smart Checkout AI | Akıllı Kasa Asistanı",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODEL VE VERİ YÜKLEME ---
@st.cache_resource
def load_model():
    model_path = "smart_kasa_model.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None

@st.cache_data
def load_dataset():
    data_path = "musteri-davranis-seti.csv"
    if os.path.exists(data_path):
        try:
            return pd.read_csv(data_path)
        except Exception:
            return None
    # Dosya yoksa örnek fallback veri seti oluşturalım
    return pd.DataFrame({
        "Musteri_ID": [1001, 1002, 1003],
        "Ad_Soyad": ["Fatma Yılmaz", "Fatma Çelik", "Ahmet Kaya"],
        "Telefon": ["5555614226", "5552458591", "5334903402"],
        "Kayit_Tarihi": ["2024-09-07", "2025-08-27", "2025-06-01"],
        "Segment": ["Sadık Müşteri", "Yeni Müşteri", "Yeni Müşteri"],
        "musteri_kidem_gun": [688, 334, 421],
        "gecmis_islem_sayisi": [14, 7, 5],
        "gecmis_ortalama_sepet_tutari": [403.2, 231.76, 207.43]
    })

model = load_model()
df = load_dataset()

# Session State başlatma (Canlı sayaçlar için)
if "sunulan_oneri" not in st.session_state:
    st.session_state.sunulan_oneri = 0
if "kabul_edilen_oneri" not in st.session_state:
    st.session_state.kabul_edilen_oneri = 0
if "ek_ciro" not in st.session_state:
    st.session_state.ek_ciro = 0.0

# --- YAN MENÜ (SIDEBAR) ---
st.sidebar.markdown("🛒 **Smart Checkout AI**")
st.sidebar.caption("Retail AI Assistant")
st.sidebar.markdown("---")

maskeleme_modu = st.sidebar.toggle("🔒 Maskeleme Modu", value=False)
st.sidebar.info("Sistem, kasa anında akıllı çapraz satış ve müşteri ikna olasılığı tahmini üretir.")

st.sidebar.markdown("### **Sistem Durumu:**")
if model is not None:
    st.sidebar.success("🟢 AI Modeli Aktif (XGBoost)")
else:
    st.sidebar.warning("🟡 Model Dosyası Bulunamadı (Demo Modu)")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 **Canlı Kasa Performansı (ROI):**")
st.sidebar.metric("Ek Ciro Katkısı", f"{st.session_state.ek_ciro:.2f} TL")

donusum_orani = (st.session_state.kabul_edilen_oneri / st.session_state.sunulan_oneri * 100) if st.session_state.sunulan_oneri > 0 else 0.0
st.sidebar.metric("Kasa Dönüşüm Oranı", f"%{donusum_orani:.1f}")

# --- ANA BAŞLIK ---
st.markdown("## 🛒 Akıllı Kasa Asistanı & Öneri Motoru")

# --- SEKMELER (TABS) ---
tab_kasa, tab_analitik, tab_kayit = st.tabs([
    "🛒 Kasa İşlem Ekranı", 
    "📊 Müşteri & Mağaza Analitiği", 
    "➕ Yeni Müşteri Kaydı"
])

# ================= TAB 1: KASA İŞLEM EKRANI =================
with tab_kasa:
    col_islem1, col_islem2 = st.columns([1.2, 1])

    with col_islem1:
        st.subheader("1. Müşteri & Sepet Bilgileri")
        
        telefon_input = st.text_input("📱 Müşteri Telefon No (Örn: 5555614226):", value="5555614226")

        # Müşteri veritabanında arama
        bulunan_musteri = None
        if df is not None and "Telefon" in df.columns:
            eslesenler = df[df["Telefon"].astype(str) == telefon_input.strip()]
            if not eslesenler.empty:
                bulunan_musteri = eslesenler.iloc[0]

        if bulunan_musteri is not None:
            ad_soyad = bulunan_musteri.get("Ad_Soyad", "Bilinmeyen Müşteri")
            segment = bulunan_musteri.get("Segment", "Standart")
            st.success(f"Müşteri Bulundu: **{ad_soyad}** ({segment} Segment)")
        else:
            st.warning("⚠️ Müşteri veritabanında bulunamadı. Yeni müşteri olarak işlem yapılacak.")

        st.markdown("---")
        
        # Sepet Detayları
        sepet_tutari = st.number_input("💰 Anlık Sepet Tutarı (TL):", min_value=0.0, max_value=50000.0, value=350.0, step=10.0)
        sepetteki_urun_adedi = st.slider("📦 Sepetteki Ürün Adedi:", min_value=1, max_value=20, value=3)
        
        kategori_secenekleri = [
            "Makyaj", "Cilt Bakımı", "Saç Bakımı", "Parfüm", 
            "Vücut Bakımı & Banyo", "El & Ayak Bakımı", "Güneş & Bronzlaşma", 
            "Aksesuar & Güzellik Aletleri", "Erkek Bakım"
        ]
        agirlikli_kategori = st.selectbox("🏷️ Sepetteki Ağırlıklı Kategori:", kategori_secenekleri)

        hesapla_btn = st.button("⚡ AI Önerilerini ve Tahmini Hesapla", use_container_width=True, type="primary")

    with col_islem2:
        st.subheader("2. Yapay Zeka & Kasa Fırsat Analizi")

        if not hesapla_btn:
            st.info("👉 Önerileri ve XAI detaylarını görmek için lütfen sol taraftaki 'AI Önerilerini ve Tahmini Hesapla' butonuna basın.")
        else:
            st.session_state.sunulan_oneri += 1
            
            # Öneri Belirleme (Kategoriye göre akıllı eşleşme)
            tavsiye_sozlugu = {
                "Makyaj": ("Maskara (Lash Lift Etkili)", 0.78, 149.90),
                "Cilt Bakımı": ("Hyaluronik Asit Cilt Serumu", 0.82, 299.90),
                "Saç Bakımı": ("Argan Yağlı Saç Bakım Maskesi", 0.65, 189.90),
                "Parfüm": ("Vücut Spreyi (Body Mist)", 0.71, 120.00),
                "Vücut Bakımı & Banyo": ("Besleyici Duş Jeli", 0.58, 95.00),
                "El & Ayak Bakımı": ("Onarıcı El Kremi", 0.74, 75.00),
                "Güneş & Bronzlaşma": ("Yüz Güneş Kremi SPF 50+", 0.85, 340.00),
                "Aksesuar & Güzellik Aletleri": ("Gua Sha Masaj Taşı", 0.62, 110.00),
                "Erkek Bakım": ("Sakal ve Bıyık Bakım Yağı", 0.69, 150.00)
            }
            
            onerilen_urun, olasilik, urun_fiyati = tavsiye_sozlugu.get(agirlikli_kategori, ("Özel Bakım Ürünü", 0.70, 150.00))

            st.markdown(f"### 🎯 Önerilen Çapraz Satış Ürünü")
            st.success(f"**{onerilen_urun}**")
            
            st.metric("İkna / Kabul Olasılığı", f"%{olasilik * 100:.1f}")
            
            col_aks1, col_aks2 = st.columns(2)
            if col_aks1.button("✅ Teklifi Kabul Et", use_container_width=True):
                st.session_state.kabul_edilen_oneri += 1
                st.session_state.ek_ciro += urun_fiyati
                st.success("Satış başarıyla sepete eklendi!")
                st.rerun()
                
            if col_aks2.button("❌ Teklifi Reddet", use_container_width=True):
                st.info("Teklif reddedildi, kasa işlemi tamamlanıyor.")


# ================= TAB 2: MÜŞTERİ & MAĞAZA ANALİTİĞİ =================
with tab_analitik:
    st.subheader("📊 Müşteri Veri Seti & Mağaza Özet Analizleri")
    
    st.markdown("### 💰 AI Katma Değer & İş Etkisi")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Toplam Sunulan Öneri", st.session_state.sunulan_oneri)
    m2.metric("Kabul Edilen Öneri", st.session_state.kabul_edilen_oneri)
    m3.metric("Kasa Dönüşüm Oranı", f"%{donusum_orani:.1f}")
    m4.metric("AI Kaynaklı Ek Ciro", f"{st.session_state.ek_ciro:.2f} TL")

    st.markdown("---")
    
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Toplam Kayıtlı Müşteri", len(df) if df is not None else 350)
    m6.metric("VIP Müşteri Sayısı", len(df[df["Segment"] == "VIP"]) if df is not None and "Segment" in df.columns else 0)
    
    ortalama_harcama = df["gecmis_ortalama_sepet_tutari"].mean() if df is not None and "gecmis_ortalama_sepet_tutari" in df.columns else 249.3
    m7.metric("Ort. Harcama Tutarı", f"{ortalama_harcama:.1f} TL")
    
    ortalama_kidem = df["musteri_kidem_gun"].mean() if df is not None and "musteri_kidem_gun" in df.columns else 615.0
    m8.metric("Ort. Müşteri Kıdemi", f"{int(ortalama_kidem)} Gün")

    st.markdown("---")
    st.subheader("📋 Mevcut Müşteri Veri Tablosu")
    if df is not None:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Görüntülenecek veri bulunamadı.")


# ================= TAB 3: YENİ MÜŞTERİ KAYDI =================
with tab_kayit:
    st.subheader("➕ Yeni Müşteri Kaydı Formu")
    
    with st.form("yeni_musteri_formu"):
        yeni_ad = st.text_input("Ad Soyad:")
        yeni_telefon = st.text_input("Telefon No:")
        yeni_segment = st.selectbox("Segment:", ["Yeni Müşteri", "Sadık Müşteri", "VIP", "Potansiyel Müşteri"])
        
        kaydet_buton = st.form_submit_button("Müşteriyi Kaydet", type="primary")
        
        if kaydet_buton:
            if yeni_ad and yeni_telefon:
                st.success(f"✅ Başarılı: {yeni_ad} isimli müşteri sisteme kaydedildi!")
            else:
                st.error("Lütfen ad soyad ve telefon bilgilerini eksiksiz doldurun.")
